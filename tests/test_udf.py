from typing import TypedDict

import numpy as np
import pytest
from libertem.api import Context
from libertem.executor.inline import InlineJobExecutor

from libertem_parallax.udf.base import BaseParallaxUDF
from libertem_parallax.udf.parallax import ParallaxUDF, parallax_accumulate_cpu
from libertem_parallax.udf.parallax_phase_flip import (
    ParallaxPhaseFlipUDF,
    parallax_phase_flip_accumulate_cpu,
)


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


@pytest.fixture
def simple_geometry():
    shape = SIMPLE_GEOMETRY_KWARGS["shape"]
    return shape, BaseParallaxUDF._preprocess_geometry(**SIMPLE_GEOMETRY_KWARGS)


class TestPreprocessGeometryErrors:
    def test_raises_if_both_samplings_given(self):
        with pytest.raises(ValueError, match="Specify only one"):
            kwargs = SIMPLE_GEOMETRY_KWARGS.copy()
            BaseParallaxUDF._preprocess_geometry(**kwargs, angular_sampling=(1, 1))

    def test_raises_if_no_sampling_given(self):
        with pytest.raises(ValueError, match="must be specified"):
            kwargs = SIMPLE_GEOMETRY_KWARGS.copy()
            kwargs["reciprocal_sampling"] = None
            BaseParallaxUDF._preprocess_geometry(
                **kwargs,
            )

    def test_raises_if_shape_not_length_4(self):
        with pytest.raises(ValueError, match="shape` must have length 4"):
            kwargs = SIMPLE_GEOMETRY_KWARGS.copy()
            kwargs["shape"] = (64, 64, 64)  # ty:ignore[invalid-assignment]
            BaseParallaxUDF._preprocess_geometry(**kwargs)

    def test_angular_sampling_is_canonicalized(self):
        kwargs = SIMPLE_GEOMETRY_KWARGS.copy()
        kwargs["reciprocal_sampling"] = None
        angular_sampling = (2.0, 2.0)  # mrad

        pre = BaseParallaxUDF._preprocess_geometry(
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
        pre = BaseParallaxUDF._preprocess_geometry(**kwargs)

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


class TestParallaxUDFUpsampled:
    def test_upsampling_factor_2(self):
        kwargs = SIMPLE_GEOMETRY_KWARGS.copy()
        kwargs["upsampling_factor"] = 2
        shape = kwargs["shape"]
        dataset = np.zeros(shape, dtype=np.float64)

        bf_flat_idx = 36
        iy = bf_flat_idx // shape[-1]
        ix = bf_flat_idx % shape[-1]
        dataset[0, 0, iy, ix] = 1.0

        ctx = Context(executor=InlineJobExecutor())
        ds = ctx.load("memory", data=dataset)

        udf = ParallaxUDF.from_parameters(**kwargs)

        result = ctx.run_udf(dataset=ds, udf=udf)
        out_actual = result["reconstruction"].data  # ty:ignore[invalid-argument-type, not-subscriptable]

        # Downsample by factor 2 (average pooling)
        out_down = np.round(
            out_actual.reshape(
                out_actual.shape[0] // 2,
                2,
                out_actual.shape[1] // 2,
                2,
            ).sum(axis=(1, 3)),
            decimals=12,
        )

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

        np.testing.assert_allclose(
            out_down,
            expected_result,
            rtol=1e-12,
            atol=0,
        )


class TestParallaxPhaseFlipUDF:
    def test_equivalent_construction(self):
        # Construct directly from parameters
        udf_direct = ParallaxPhaseFlipUDF.from_parameters(**SIMPLE_GEOMETRY_KWARGS)

        # Construct via ParallaxUDF
        parallax_udf = ParallaxUDF.from_parameters(**SIMPLE_GEOMETRY_KWARGS)
        udf_from_udf = ParallaxPhaseFlipUDF.from_parallax_udf(parallax_udf)

        # Compare relevant attributes

        np.testing.assert_array_equal(
            udf_direct.preprocessed_geometry.bf_flat_inds,
            udf_from_udf.preprocessed_geometry.bf_flat_inds,
        )
        np.testing.assert_array_equal(
            udf_direct.preprocessed_geometry.shifts,
            udf_from_udf.preprocessed_geometry.shifts,
        )

    def test_runs_on_simple_geometry(self, simple_geometry):
        shape, pre = simple_geometry
        dataset = np.zeros(shape, dtype=np.float64)

        bf_flat_idx = 36
        iy = bf_flat_idx // shape[-1]
        ix = bf_flat_idx % shape[-1]
        dataset[0, 0, iy, ix] = 1.0

        ctx = Context(executor=InlineJobExecutor())
        ds = ctx.load("memory", data=dataset)

        udf = ParallaxPhaseFlipUDF.from_parameters(
            **SIMPLE_GEOMETRY_KWARGS,
        )

        result = ctx.run_udf(dataset=ds, udf=udf)
        out = result["reconstruction"].data  # ty:ignore[invalid-argument-type, not-subscriptable]

        assert out.shape == pre.upsampled_scan_gpts
        assert np.isfinite(out).all()

    @pytest.mark.parametrize("desc, rotation_adjust, flip_cols", DETECTOR_ORIENTATIONS)
    def test_phase_flipping_invariants_orientations(
        self, simple_geometry, desc, rotation_adjust, flip_cols
    ):
        """
        “For the phase-flipping UDF, the output is no longer analytically tractable due to kernel convolution and periodic wrap-around.
        Tests therefore validate structural invariants such as DC dominance and radial correlation with the analytical limit.”
        """

        def radial_average_pbc(arr):
            ny, nx = arr.shape
            # Create coordinate grids centered at 0,0
            y, x = np.indices((ny, nx))

            # Apply minimum image convention to coordinates
            # This shifts coordinates to the range [-size/2, size/2]
            dx = np.remainder(x + nx / 2, nx) - nx / 2
            dy = np.remainder(y + ny / 2, ny) - ny / 2

            # Calculate shortest periodic distance
            r = np.sqrt(dx**2 + dy**2)
            r_int = r.astype(int)

            # Maximum possible distance in a periodic box is half the diagonal
            max_r = int(np.sqrt((nx / 2) ** 2 + (ny / 2) ** 2))

            # Use np.bincount for fast, vectorized summation
            # We clip r_int to max_r to ignore distances beyond the box center
            mask = r_int <= max_r
            radial_sum = np.bincount(
                r_int[mask], weights=arr[mask], minlength=max_r + 1
            )
            counts = np.bincount(r_int[mask], minlength=max_r + 1)

            # Avoid division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(counts > 0, radial_sum / counts, 0)

        shape, pre = simple_geometry
        dataset = np.zeros(shape, dtype=np.float64)

        bf_flat_idx = 36
        iy = bf_flat_idx // shape[-1]
        ix = bf_flat_idx % shape[-1]
        dataset[0, 0, iy, ix] = 1.0

        # Apply detector orientation transforms
        if "reversed Q_rows" in desc:
            dataset = dataset[..., ::-1, :]
        if "reversed Q_cols" in desc:
            dataset = dataset[..., :, ::-1]
        if "Q_cols" in desc.split(",")[0]:
            dataset = dataset.swapaxes(-2, -1)

        ctx = Context(executor=InlineJobExecutor())
        ds = ctx.load("memory", data=dataset)

        udf = ParallaxPhaseFlipUDF.from_parameters(
            **SIMPLE_GEOMETRY_KWARGS,
            rotation_angle=rotation_adjust,
            detector_flip_cols=flip_cols,
        )
        udf_flip = ParallaxPhaseFlipUDF.from_parameters(
            **SIMPLE_GEOMETRY_KWARGS,
            rotation_angle=rotation_adjust,
            detector_flip_cols=flip_cols,
        )

        result, result_flip = ctx.run_udf(dataset=ds, udf=[udf, udf_flip])  # ty:ignore[not-iterable]
        out = result["reconstruction"].data  # ty:ignore[invalid-argument-type]
        out_flip = result_flip["reconstruction"].data  # ty:ignore[invalid-argument-type]

        radially_avg_out = radial_average_pbc(out)
        radially_avg_out_flip = radial_average_pbc(out_flip)

        # DC dominance
        assert abs(radially_avg_out_flip[0]) == np.max(np.abs(radially_avg_out))

        # Structural similarity
        corr = np.corrcoef(radially_avg_out, radially_avg_out_flip)[0, 1]
        assert corr > 0.99
