from dataclasses import dataclass
from typing import cast

import numpy as np
from libertem.common.shape import Shape
from libertem.udf import UDF

from libertem_parallax.utils import (
    electron_wavelength,
    polar_coordinates,
    quadratic_aberration_cartesian_gradients,
    spatial_frequencies,
)


@dataclass(frozen=True)
class PreprocessedGeometry:
    bf_flat_inds: np.ndarray
    shifts: np.ndarray
    wavelength: float
    gpts: tuple[int, int]
    reciprocal_sampling: tuple[float, float]
    sampling: tuple[float, float]
    upsampled_scan_gpts: tuple[int, int]
    upsampled_scan_sampling: tuple[float, float]
    upsampling_factor: int
    aberration_coefs: dict[str, float]


class BaseParallaxUDF(UDF):
    """
    Base class for parallax-based UDFs.

    Provides common preprocessing class method for:
    - reciprocal-space sampling
    - bright-field mask & flat indices
    - integer parallax shifts from aberration phase gradients
    """

    @classmethod
    def _preprocess_geometry(
        cls,
        shape: tuple[int, int, int, int] | Shape,
        scan_sampling: tuple[float, float],
        energy: float,
        semiangle_cutoff: float,
        reciprocal_sampling: tuple[float, float] | None = None,
        angular_sampling: tuple[float, float] | None = None,
        aberration_coefs: dict[str, float] | None = None,
        rotation_angle: float | None = None,
        upsampling_factor: int = 1,
        detector_flip_cols: bool = False,
    ):
        """
        Precomputes:
        - reciprocal-space sampling
        - the bright-field mask defined by `semiangle_cutoff`
        - integer parallax shifts from aberration phase gradients

        Exactly one of `reciprocal_sampling` or `angular_sampling` must be
        specified. Angular sampling is interpreted in mrad.

        Parameters
        ----------
        shape:
            Acquisition shape of length 4.
            First two are scan dimensions. Last two are signal dimensions.
        scan_sampling
            Scan sampling in real space.
        energy
            Electron beam energy in eV.
        semiangle_cutoff
            Bright-field semiangle cutoff in mrad.
        reciprocal_sampling
            Reciprocal-space sampling in 1/Å.
        angular_sampling
            Angular sampling in mrad (convenience alternative).
        aberration_coefs
            Polar aberration coefficients dictionary.
        rotation_angle
            Optional rotation of reciprocal coordinates, in radians.
        upsampling_factor
            Integer upsampling factor for the scan grid.
        detector_flip_cols
            Controls detector ordering.

        Detector ordering conventions
        -----------------------------

        Internally, all geometry is computed assuming a canonical detector ordering
        (Q_rows, Q_cols). Differences in how detector data are stored on disk or
        streamed (transposes and flips) are handled by a combination of:

        - `rotation_angle` (applied in reciprocal-space geometry)
        - `detector_flip_cols` (applied as a zero-copy view on the data)

        No other data reordering is performed.

        The table below lists how common detector orderings should be mapped to these
        parameters. All rotations are applied by *adding* the indicated angle to
        `rotation_angle`.

        Raw detector layout                  rotation_angle adjustment   detector_flip_cols
        -----------------------------------------------------------------------------------
        (Q_rows, Q_cols)                     0                           False
        (Q_rows, reversed Q_cols)            0                           True
        (reversed Q_rows, Q_cols)            +π                          True
        (reversed Q_rows, reversed Q_cols)   +π                          False
        (Q_cols, Q_rows)                     +π/2                        True
        (Q_cols, reversed Q_rows)            +π/2                        False
        (reversed Q_cols, Q_rows)            −π/2                        False
        (reversed Q_cols, reversed Q_rows)   −π/2                        True

        Notes
        -----
        - `detector_flip_cols=True` corresponds to `frames[..., ::-1]` and is always
        applied as a view (no copy).
        - Row flips are *not* applied directly; they are represented by a π rotation
        combined with a column flip.
        - This parameterization is sufficient to represent all transpose/flip
        combinations without modifying the geometry code path.
        - Continuous relative rotation between detector and scan coordinates should
        be expressed via `rotation_angle` in addition to the adjustments above.
        """

        # ---- Sampling ----
        wavelength = electron_wavelength(energy)

        if reciprocal_sampling is not None and angular_sampling is not None:
            raise ValueError(
                "Specify only one of `reciprocal_sampling` or `angular_sampling`, not both."
            )

        if reciprocal_sampling is None and angular_sampling is None:
            raise ValueError(
                "One of `reciprocal_sampling` or `angular_sampling` must be specified."
            )

        # Canonicalize to reciprocal sampling
        if reciprocal_sampling is None:
            assert angular_sampling is not None
            reciprocal_sampling = (
                angular_sampling[0] / wavelength / 1e3,
                angular_sampling[1] / wavelength / 1e3,
            )

        if len(shape) != 4:
            raise ValueError(f"`shape` must have length 4, not {len(shape)}.")

        scan_gpts = (shape[0], shape[1])
        gpts = (shape[-2], shape[-1])

        sampling = (
            1.0 / reciprocal_sampling[0] / gpts[0],
            1.0 / reciprocal_sampling[1] / gpts[1],
        )

        upsampled_scan_gpts = (
            scan_gpts[0] * upsampling_factor,
            scan_gpts[1] * upsampling_factor,
        )

        upsampled_scan_sampling = (
            scan_sampling[0] / upsampling_factor,
            scan_sampling[1] / upsampling_factor,
        )

        # ---- Parallax shifts ----
        if aberration_coefs is None:
            aberration_coefs = {}

        kxa, kya = spatial_frequencies(
            gpts,
            sampling,
            rotation_angle=rotation_angle,
        )

        k, phi = polar_coordinates(kxa, kya)

        # ---- BF indices ----
        bf_mask = k * wavelength * 1e3 <= semiangle_cutoff
        inds_i, inds_j = np.where(bf_mask)

        inds_i_fft = (inds_i - gpts[0] // 2) % gpts[0]
        inds_j_fft = (inds_j - gpts[1] // 2) % gpts[1]

        if rotation_angle is not None:
            # FFT parity correction:
            # For even-sized grids, fftshift centers the origin between pixels.
            # Rotations by odd multiples of π/2 change which side of that half-pixel
            # the rotated coordinates fall on, requiring a one-pixel correction.

            n_rot = int(np.round(rotation_angle / (np.pi / 2))) % 4

            if n_rot == 1:  # +π/2
                if gpts[1] % 2 == 0:
                    inds_j_fft = (inds_j_fft - 1) % gpts[1]
            elif n_rot == 2:  # π
                if gpts[0] % 2 == 0:
                    inds_i_fft = (inds_i_fft - 1) % gpts[0]
                if gpts[1] % 2 == 0:
                    inds_j_fft = (inds_j_fft - 1) % gpts[1]
            elif n_rot == 3:  # -π/2
                if gpts[0] % 2 == 0:
                    inds_i_fft = (inds_i_fft - 1) % gpts[0]

        bf_flat_inds = (inds_i_fft * gpts[1] + inds_j_fft).astype(np.int32)

        dx, dy = quadratic_aberration_cartesian_gradients(
            k * wavelength,
            phi,
            aberration_coefs,
        )

        grad_k = np.stack(
            (dx[inds_i, inds_j], dy[inds_i, inds_j]),
            axis=-1,
        )

        shifts = np.round(grad_k / (2 * np.pi) / upsampled_scan_sampling).astype(
            np.int32
        )

        return PreprocessedGeometry(
            bf_flat_inds=bf_flat_inds,
            shifts=shifts,
            wavelength=wavelength,
            gpts=gpts,
            reciprocal_sampling=reciprocal_sampling,
            sampling=sampling,
            upsampled_scan_gpts=upsampled_scan_gpts,
            upsampled_scan_sampling=upsampled_scan_sampling,
            upsampling_factor=upsampling_factor,
            aberration_coefs=aberration_coefs,
        )

    def get_result_buffers(self):
        return {
            "reconstruction": self.buffer(
                kind="single",
                dtype=np.float64,
                extra_shape=self.preprocessed_geometry.upsampled_scan_gpts,
            )
        }

    @property
    def preprocessed_geometry(self):
        pre = cast(PreprocessedGeometry, self.params.preprocessed_geometry)
        return pre
