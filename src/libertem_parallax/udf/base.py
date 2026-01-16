from dataclasses import dataclass

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
    upsampled_scan_gpts: tuple[int, int]
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
        shape: tuple[int, int, int, int] | Shape,
        scan_sampling: tuple[float, float],
        energy: float,
        semiangle_cutoff: float,
        reciprocal_sampling: tuple[float, float] | None = None,
        angular_sampling: tuple[float, float] | None = None,
        aberration_coefs: dict[str, float] | None = None,
        rotation_angle: float | None = None,
        upsampling_factor: int = 1,
        detector_transpose: bool = False,
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
        if detector_transpose:
            gpts = (shape[-1], shape[-2])
        else:
            gpts = (shape[-2], shape[-1])

        sampling = (
            1.0 / reciprocal_sampling[0] / gpts[0],
            1.0 / reciprocal_sampling[1] / gpts[1],
        )

        upsampled_scan_gpts = (
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

        inds_i_fft = (inds_i - gpts[0] // 2) % gpts[0]
        inds_j_fft = (inds_j - gpts[1] // 2) % gpts[1]
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

        shifts = np.round(grad_k / (2 * np.pi) / upsampled_sampling).astype(np.int32)

        return PreprocessedGeometry(
            bf_flat_inds=bf_flat_inds,
            shifts=shifts,
            wavelength=wavelength,
            gpts=gpts,
            upsampled_scan_gpts=upsampled_scan_gpts,
            upsampled_sampling=upsampled_sampling,
        )

    def get_result_buffers(self):
        return {
            "reconstruction": self.buffer(
                kind="single",
                dtype=np.float64,
                extra_shape=self.params.upsampled_scan_gpts,  # ty:ignore[invalid-argument-type]
            )
        }
