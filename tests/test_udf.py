from typing import TypedDict

import numpy as np
import pytest
from libertem.api import Context
from libertem.executor.inline import InlineJobExecutor

from libertem_parallax.udf.base import BaseParallaxUDF
from libertem_parallax.udf.parallax import ParallaxUDF, parallax_accumulate_cpu
from libertem_parallax.udf.parallax_phase_flip import parallax_phase_flip_accumulate_cpu


class GeometryKwargs(TypedDict):
    shape: tuple[int, int, int, int]
    scan_sampling: tuple[float, float]
    reciprocal_sampling: tuple[float, float] | None
    energy: float
    semiangle_cutoff: float
    upsampling_factor: int
    aberration_coefs: dict[str, float] | None


SIMPLE_GEOMETRY_KWARGS: GeometryKwargs = {
    "shape": (7, 8, 8, 8),
    "scan_sampling": (1.0, 1.0),
    "reciprocal_sampling": (1.0, 1.0),
    "energy": 1.2e7,  # such that lambda * 1e3 = 1
    "semiangle_cutoff": 2.5,
    "upsampling_factor": 1,
    "aberration_coefs": {"C10": 1e3},
}


@pytest.fixture
def simple_geometry():
    shape = SIMPLE_GEOMETRY_KWARGS["shape"]
    return shape, BaseParallaxUDF.preprocess_geometry(**SIMPLE_GEOMETRY_KWARGS)


class TestPreprocessGeometryErrors:
    def test_raises_if_both_samplings_given(self):
        with pytest.raises(ValueError, match="Specify only one"):
            kwargs = SIMPLE_GEOMETRY_KWARGS.copy()
            BaseParallaxUDF.preprocess_geometry(**kwargs, angular_sampling=(1, 1))

    def test_raises_if_no_sampling_given(self):
        with pytest.raises(ValueError, match="must be specified"):
            kwargs = SIMPLE_GEOMETRY_KWARGS.copy()
            kwargs["reciprocal_sampling"] = None
            BaseParallaxUDF.preprocess_geometry(
                **kwargs,
            )

    def test_raises_if_shape_not_length_4(self):
        with pytest.raises(ValueError, match="shape` must have length 4"):
            kwargs = SIMPLE_GEOMETRY_KWARGS.copy()
            kwargs["shape"] = (64, 64, 64)  # ty:ignore[invalid-assignment]
            BaseParallaxUDF.preprocess_geometry(**kwargs)

    def test_angular_sampling_is_canonicalized(self):
        kwargs = SIMPLE_GEOMETRY_KWARGS.copy()
        kwargs["reciprocal_sampling"] = None
        angular_sampling = (2.0, 2.0)  # mrad

        pre = BaseParallaxUDF.preprocess_geometry(
            **kwargs, angular_sampling=angular_sampling
        )

        reciprocal_sampling = pre.reciprocal_sampling

        expected = (
            angular_sampling[0] / pre.wavelength / 1e3,
            angular_sampling[1] / pre.wavelength / 1e3,
        )

        np.testing.assert_allclose(reciprocal_sampling, expected)

    def test_aberrations_None(self):
        kwargs = SIMPLE_GEOMETRY_KWARGS.copy()
        kwargs["aberration_coefs"] = None
        pre = BaseParallaxUDF.preprocess_geometry(**kwargs)

        expected_shifts = np.zeros_like(pre.shifts)

        np.testing.assert_allclose(
            pre.shifts,
            expected_shifts,
        )


class TestParallaxUDF:
    @pytest.fixture(autouse=True)
    def setup_geometry(self, simple_geometry):
        self.shape, self.pre = simple_geometry
        self.sy, self.sx = self.pre.upsampled_scan_gpts
        self.qy, self.qx = self.pre.gpts

    DETECTOR_ORIENTATIONS = [
        ("Q_rows,Q_cols", 0.0, False),
        ("Q_rows,reversed Q_cols", 0.0, True),
        ("reversed Q_rows,Q_cols", np.pi, True),
        ("reversed Q_rows,reversed Q_cols", np.pi, False),
        ("Q_cols,Q_rows", np.pi / 2, True),
        ("Q_cols,reversed Q_rows", np.pi / 2, False),
        ("reversed Q_cols,Q_rows", -np.pi / 2, False),
        ("reversed Q_cols,reversed Q_rows", -np.pi / 2, True),
    ]

    @pytest.mark.parametrize("desc, rotation_adjust, flip_cols", DETECTOR_ORIENTATIONS)
    def test_orientations(self, desc, rotation_adjust, flip_cols):
        shape = self.shape
        dataset = np.zeros(shape, dtype=np.float64)

        bf_flat_idx = 36
        iy = bf_flat_idx // shape[-1]
        ix = bf_flat_idx % shape[-1]
        dataset[0, 0, iy, ix] = 1.0

        if "reversed Q_rows" in desc:
            dataset = dataset[..., ::-1, :]
        if "reversed Q_cols" in desc:
            dataset = dataset[..., :, ::-1]
        if "Q_cols" in desc.split(",")[0]:
            dataset = dataset.swapaxes(-2, -1)

        ctx = Context(executor=InlineJobExecutor())
        ds = ctx.load("memory", data=dataset)

        udf = ParallaxUDF.from_parameters(
            rotation_angle=rotation_adjust,
            detector_flip_cols=flip_cols,
            **SIMPLE_GEOMETRY_KWARGS,
        )
        result = ctx.run_udf(dataset=ds, udf=udf)
        out_actual = result["reconstruction"].data  # ty:ignore[invalid-argument-type, not-subscriptable]

        expected_result = (
            np.array(
                [
                    [20, -1, -1, 0, 0, 0, -1, -1],
                    [-1, -1, -1, 0, 0, 0, -1, -1],
                    [-1, -1, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [-1, -1, 0, 0, 0, 0, 0, -1],
                    [-1, -1, -1, 0, 0, 0, -1, -1],
                ]
            )
            / 21
        )

        np.testing.assert_allclose(out_actual, expected_result, rtol=1e-12, atol=0)

    def test_parallax_single_bf_pixel(self):
        frames = np.zeros((1, self.qy, self.qx), dtype=np.float64)
        bf_flat_idx = self.pre.bf_flat_inds[0]
        iy = bf_flat_idx // self.qx
        ix = bf_flat_idx % self.qx
        frames[0, iy, ix] = 1.0
        coords = np.array([[0, 0]], dtype=np.int64)
        out = np.zeros((self.sy, self.sx), dtype=np.float64)

        parallax_accumulate_cpu(
            frames, self.pre.bf_flat_inds, self.pre.shifts, coords, out
        )

        M = len(self.pre.bf_flat_inds)
        expected = np.zeros_like(out)
        expected[0, 0] = (M - 1) / M
        for dy, dx in self.pre.shifts[1:]:
            oy = dy % self.sy
            ox = dx % self.sx
            expected[oy, ox] -= 1 / M

        np.testing.assert_allclose(out, expected, rtol=0, atol=0)

    def test_phase_flip_uniform_kernel_is_zero(self):
        frames = np.random.rand(2, self.qy, self.qx)
        coords = np.array([[0, 0], [1, 2]], dtype=np.int64)
        M = len(self.pre.bf_flat_inds)
        bf_rows = self.pre.bf_flat_inds // self.qx
        bf_cols = self.pre.bf_flat_inds % self.qx

        unique_offsets = np.array([0], dtype=np.int64)
        grouped_kernel = np.ones((1, M), dtype=np.float64)
        out = np.zeros((self.sy, self.sx), dtype=np.float64)

        parallax_phase_flip_accumulate_cpu(
            frames, bf_rows, bf_cols, coords, unique_offsets, grouped_kernel, out
        )
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    def test_phase_flip_reduces_to_parallax(self):
        frames = np.zeros((1, self.qy, self.qx), dtype=np.float64)
        bf_flat_idx = self.pre.bf_flat_inds[0]
        iy = bf_flat_idx // self.qx
        ix = bf_flat_idx % self.qx
        frames[0, iy, ix] = 1.0
        coords = np.array([[0, 0]], dtype=np.int64)

        out_parallax = np.zeros((self.sy, self.sx), dtype=np.float64)
        parallax_accumulate_cpu(
            frames, self.pre.bf_flat_inds, self.pre.shifts, coords, out_parallax
        )

        bf_rows = self.pre.bf_flat_inds // self.qx
        bf_cols = self.pre.bf_flat_inds % self.qx
        offsets = np.array(
            [((dy % self.sy) * self.sx + (dx % self.sx)) for dy, dx in self.pre.shifts],
            dtype=np.int64,
        )

        unique_offsets, inv = np.unique(offsets, return_inverse=True)
        U = len(unique_offsets)
        M = len(offsets)
        grouped_kernel = np.zeros((U, M))
        for m in range(M):
            grouped_kernel[inv[m], m] = 1.0

        out_phase = np.zeros((self.sy, self.sx), dtype=np.float64)
        parallax_phase_flip_accumulate_cpu(
            frames, bf_rows, bf_cols, coords, unique_offsets, grouped_kernel, out_phase
        )

        np.testing.assert_allclose(out_phase, out_parallax, rtol=1e-12, atol=1e-12)

    def test_edge_wrapping(self):
        frames = np.zeros((1, self.qy, self.qx), dtype=np.float64)
        # Put BF pixels near bottom-right corner
        for idx in self.pre.bf_flat_inds[:3]:
            iy = idx // self.qx
            ix = idx % self.qx
            frames[0, iy, ix] = 1.0
        coords = np.array([[self.sy - 1, self.sx - 1]], dtype=np.int64)
        out = np.zeros((self.sy, self.sx), dtype=np.float64)
        parallax_accumulate_cpu(
            frames, self.pre.bf_flat_inds, self.pre.shifts, coords, out
        )
        # Ensure output is non-zero and wraps
        assert np.any(out != 0.0)
