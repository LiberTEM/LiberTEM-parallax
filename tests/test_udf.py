import numpy as np
import pytest
from libertem.api import Context
from libertem.executor.inline import InlineJobExecutor

from libertem_parallax.udf.parallax import ParallaxUDF

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
    shape = (7, 8, 8, 8)
    scan_sampling = (1, 1)
    reciprocal_sampling = (1, 1)
    energy = 1.2e7
    semiangle_cutoff = 2.5
    upsampling_factor = 1
    aberration_coefs = {"C10": 1e3}

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
        shape=shape,
        scan_sampling=scan_sampling,
        reciprocal_sampling=reciprocal_sampling,
        energy=energy,
        semiangle_cutoff=semiangle_cutoff,
        upsampling_factor=upsampling_factor,
        aberration_coefs=aberration_coefs,
        rotation_angle=rotation_adjust,
        detector_flip_cols=flip_cols,
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
