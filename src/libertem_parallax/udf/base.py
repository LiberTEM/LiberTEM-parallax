from dataclasses import dataclass

import numpy as np
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
    upsampled_gpts: tuple[int, int]
    upsampled_sampling: tuple[float, float]


class BaseParallaxUDF(UDF):
    """
    Base class for parallax-based UDFs.

    Provides common preprocessing class method for:
    - reciprocal-space sampling
    - bright-field mask & flat indices
    - integer parallax shifts from aberration phase gradients
    """

    @classmethod
    def preprocess_geometry(
        cls,
        gpts: tuple[int, int],
        scan_gpts: tuple[int, int],
        scan_sampling: tuple[float, float],
        energy: float,
        semiangle_cutoff: float,
        reciprocal_sampling: tuple[float, float] | None = None,
        angular_sampling: tuple[float, float] | None = None,
        aberration_coefs: dict[str, float] | None = None,
        rotation_angle: float | None = None,
        upsampling_factor: int = 1,
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
        gpts
            Number of detector pixels (ny, nx).
        scan_gpts
            Scan grid size (Ny, Nx).
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

        ny, nx = gpts
        sampling = (
            1.0 / reciprocal_sampling[0] / ny,
            1.0 / reciprocal_sampling[1] / nx,
        )

        upsampled_gpts = (
            scan_gpts[0] * upsampling_factor,
            scan_gpts[1] * upsampling_factor,
        )

        upsampled_sampling = (
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

        inds_i_fft = (inds_i - ny // 2) % ny
        inds_j_fft = (inds_j - nx // 2) % nx
        bf_flat_inds = (inds_i_fft * nx + inds_j_fft).astype(np.int32)

        dx, dy = quadratic_aberration_cartesian_gradients(
            k * wavelength,
            phi,
            aberration_coefs,
        )

        grad_k = np.stack(
            (dx[inds_i, inds_j], dy[inds_i, inds_j]),
            axis=-1,
        )

        shifts = np.round(grad_k / (2 * np.pi) / upsampled_sampling).astype(np.int32)

        return PreprocessedGeometry(
            bf_flat_inds=bf_flat_inds,
            shifts=shifts,
            wavelength=wavelength,
            upsampled_gpts=upsampled_gpts,
            upsampled_sampling=upsampled_sampling,
        )
