from typing import TypedDict

import numpy as np
import pytest
from libertem.api import Context
from libertem.executor.inline import InlineJobExecutor

from libertem_parallax.udf.parallax import ParallaxUDF, parallax_accumulate_cpu
from libertem_parallax.udf.parallax_phase_flip import (
    parallax_phase_flip_accumulate_cpu,
)


class GeometryKwargs(TypedDict):
    shape: tuple[int, int, int, int]
    scan_sampling: tuple[float, float]
    reciprocal_sampling: tuple[float, float]
    energy: float
    semiangle_cutoff: float
    upsampling_factor: int
    aberration_coefs: dict[str, float]


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
    return ParallaxUDF.preprocess_geometry(**SIMPLE_GEOMETRY_KWARGS)


# docstring table
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
def test_parallax_udf_orientations(desc, rotation_adjust, flip_cols):
    """
    Test ParallaxUDF on all 8 possible detector orientations.
    Note -- detector shape needs to be square and odd, to ensure indexing is exact and the output
    isn't rolled by one pixel.
    """

    # --- 1. Prepare toy dataset ---
    shape: tuple[int, int, int, int] = SIMPLE_GEOMETRY_KWARGS["shape"]

    gpts = shape[-2:]
    dataset = np.zeros(shape, dtype=np.float64)

    # Set a single asymmetric pixel
    bf_flat_idx = 36
    iy = bf_flat_idx // gpts[-1]
    ix = bf_flat_idx % gpts[-1]
    dataset[0, 0, iy, ix] = 1.0

    if "reversed Q_rows" in desc:
        dataset = dataset[..., ::-1, :]
    if "reversed Q_cols" in desc:
        dataset = dataset[..., :, ::-1]
    if "Q_cols" in desc.split(",")[0]:
        dataset = dataset.swapaxes(-2, -1)

    # --- 2. Setup LiberTEM context ---
    ctx = Context(executor=InlineJobExecutor())
    ds = ctx.load("memory", data=dataset)

    # --- 3. Parallax UDF parameters ---
    udf = ParallaxUDF.from_parameters(
        rotation_angle=rotation_adjust,
        detector_flip_cols=flip_cols,
        **SIMPLE_GEOMETRY_KWARGS,
    )

    # --- 4. Run UDF ---
    result = ctx.run_udf(dataset=ds, udf=udf)
    out_actual = result["reconstruction"].data  # ty:ignore[not-subscriptable, invalid-argument-type]

    # --- 5. Expected output (precomputed fractions from BF shifts & mean subtraction) ---
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

    # --- 6. Compare ---
    np.testing.assert_allclose(out_actual, expected_result, rtol=1e-12, atol=0)


def test_parallax_single_bf_pixel(simple_geometry):
    pre = simple_geometry
    sy, sx = pre.upsampled_scan_gpts
    qy, qx = pre.gpts

    # --- dataset ---
    frames = np.zeros((1, qy, qx), dtype=np.float64)

    bf_flat_idx = pre.bf_flat_inds[0]
    iy = bf_flat_idx // qx
    ix = bf_flat_idx % qx
    frames[0, iy, ix] = 1.0

    coords = np.array([[0, 0]], dtype=np.int64)
    out = np.zeros((sy, sx), dtype=np.float64)

    # --- run ---
    parallax_accumulate_cpu(
        frames,
        pre.bf_flat_inds,
        pre.shifts,
        coords,
        out,
    )

    M = len(pre.bf_flat_inds)

    # --- expected ---
    expected = np.zeros_like(out)
    expected[0, 0] = (M - 1) / M

    for dy, dx in pre.shifts[1:]:
        oy = dy % sy
        ox = dx % sx
        expected[oy, ox] -= 1 / M

    np.testing.assert_allclose(out, expected, rtol=0, atol=0)


def test_phase_flip_uniform_kernel_is_zero(simple_geometry):
    pre = simple_geometry
    sy, sx = pre.upsampled_scan_gpts
    qy, qx = pre.gpts

    frames = np.random.rand(2, qy, qx)
    coords = np.array([[0, 0], [1, 2]], dtype=np.int64)

    M = len(pre.bf_flat_inds)
    bf_rows = pre.bf_flat_inds // qx
    bf_cols = pre.bf_flat_inds % qx

    unique_offsets = np.array([0], dtype=np.int64)
    grouped_kernel = np.ones((1, M), dtype=np.float64)

    out = np.zeros((sy, sx), dtype=np.float64)

    parallax_phase_flip_accumulate_cpu(
        frames,
        bf_rows,
        bf_cols,
        coords,
        unique_offsets,
        grouped_kernel,
        out,
    )

    np.testing.assert_allclose(out, 0.0, atol=1e-12)


def test_phase_flip_reduces_to_parallax(simple_geometry):
    pre = simple_geometry
    sy, sx = pre.upsampled_scan_gpts
    qy, qx = pre.gpts

    frames = np.zeros((1, qy, qx), dtype=np.float64)

    bf_flat_idx = pre.bf_flat_inds[0]
    iy = bf_flat_idx // qx
    ix = bf_flat_idx % qx
    frames[0, iy, ix] = 1.0

    coords = np.array([[0, 0]], dtype=np.int64)

    # --- Parallax ---
    out_parallax = np.zeros((sy, sx), dtype=np.float64)
    parallax_accumulate_cpu(
        frames,
        pre.bf_flat_inds,
        pre.shifts,
        coords,
        out_parallax,
    )

    # --- Phase-flip equivalent ---
    bf_rows = pre.bf_flat_inds // qx
    bf_cols = pre.bf_flat_inds % qx

    # # Unique offsets = parallax shifts
    # unique_offsets = np.array(
    #     [dy * sx + dx for dy, dx in pre.shifts],
    #     dtype=np.int64,
    # )

    # # Identity kernel: K[u, m] = δ_{u,m}
    # M = len(pre.bf_flat_inds)
    # grouped_kernel = np.eye(M, dtype=np.float64)

    offsets = np.array(
        [((dy % sy) * sx + (dx % sx)) for dy, dx in pre.shifts],
        dtype=np.int64,
    )

    unique_offsets, inv = np.unique(offsets, return_inverse=True)

    U = len(unique_offsets)
    M = len(offsets)

    grouped_kernel = np.zeros((U, M))
    for m in range(M):
        grouped_kernel[inv[m], m] = 1.0

    out_phase = np.zeros((sy, sx), dtype=np.float64)

    parallax_phase_flip_accumulate_cpu(
        frames,
        bf_rows,
        bf_cols,
        coords,
        unique_offsets,
        grouped_kernel,
        out_phase,
    )

    np.testing.assert_allclose(
        out_phase,
        out_parallax,
        rtol=1e-12,
        atol=1e-12,
    )
