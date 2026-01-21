import numba
import numpy as np
from libertem.common.shape import Shape

from libertem_parallax.utils import (
    suppress_nyquist_frequency,
)

from .base import BaseParallaxUDF


@numba.njit(fastmath=True, nogil=True, cache=True)  # pragma: no cover
def parallax_accumulate_cpu(
    frames,  # (T, sy, sx) float32/64
    bf_flat_inds,  # (M,) int32
    shifts,  # (M, 2) int32
    coords,  # (T, 2) int64
    out,  # (Ny, Nx) float64
):
    """
    Accumulate real-valued parallax signal into `out`.

    For each frame:
    - compute the mean over BF pixels specified by `bf_flat_inds`
    - subtract this mean from each BF pixel
    - accumulate the shifted values into `out` using integer shifts

    All accumulation is performed in-place into `out`.
    """
    # shapes
    T = frames.shape[0]
    M = shifts.shape[0]
    Ny, Nx = out.shape
    sx = frames.shape[2]

    # loop over frames
    for t in range(T):
        frame = frames[t]
        yt, xt = coords[t]

        # Compute mean over BF pixels
        mean = 0.0
        for m in range(M):
            flat_idx = bf_flat_inds[m]
            iy = flat_idx // sx
            ix = flat_idx % sx
            mean += frame[iy, ix]
        mean /= M

        # Accumulate shifted, mean-subtracted values
        for m in range(M):
            flat_idx = bf_flat_inds[m]
            iy = flat_idx // sx
            ix = flat_idx % sx

            val = frame[iy, ix] - mean

            dy, dx = shifts[m]
            oy = (yt + dy) % Ny
            ox = (xt + dx) % Nx

            out[oy, ox] += val


class ParallaxUDF(BaseParallaxUDF):
    """
    User-Defined Function for streaming parallax reconstruction.

    Accumulates mean-subtracted bright-field intensities into a
    real-space parallax reconstruction using integer pixel shifts
    derived from probe aberrations.

    Instances must be constructed via `from_parameters()`.
    Direct instantiation is not recommended to ensure
    consistent preprocessing and streaming-safe configuration.
    """

    def __init__(
        self,
        preprocessed_geometry,
        detector_flip_cols,
        suppress_Nyquist_noise,
        **kwargs,
    ):
        super().__init__(
            preprocessed_geometry=preprocessed_geometry,
            detector_flip_cols=detector_flip_cols,
            suppress_Nyquist_noise=suppress_Nyquist_noise,
            **kwargs,
        )

    @classmethod
    def from_parameters(
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
        suppress_Nyquist_noise: bool = True,
        detector_flip_cols: bool = False,
        **kwargs,
    ):
        """
        Construct a ParallaxUDF from acquisition parameters.

        This constructor computes:
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
            Polar aberration coefficients in Angstroms and radians using abTEM conventions:
            https://abtem.readthedocs.io/en/latest/user_guide/walkthrough/contrast_transfer_function.html#phase-aberrations
        rotation_angle
            Active, counter-clockwise rotation in radians of the detector frequency grid to match
            the spatial frequency grid. See ``libertem_parallax.utils.spatial_frequencies?``
            https://github.com/LiberTEM/LiberTEM-parallax/blob/main/src/libertem_parallax/utils.py#L59
        upsampling_factor
            Integer upsampling factor for the scan grid.
        suppress_Nyquist_noise
            Whether to suppress Nyquist-frequency artifacts at merge time.
        detector_flip_cols
            Controls detector ordering. See ``libertem_parallax.udf.base.BaseParallaxUDF._preprocess_geometry?``
            https://github.com/LiberTEM/LiberTEM-parallax/blob/main/src/libertem_parallax/udf/base.py#L82
        """

        pre = cls._preprocess_geometry(
            shape,
            scan_sampling,
            energy,
            semiangle_cutoff,
            reciprocal_sampling,
            angular_sampling,
            aberration_coefs,
            rotation_angle,
            upsampling_factor,
        )

        return cls(
            preprocessed_geometry=pre,
            suppress_Nyquist_noise=suppress_Nyquist_noise,
            detector_flip_cols=detector_flip_cols,
            **kwargs,
        )

    def process_partition(self, partition):
        frames = np.asarray(partition.data)
        if self.params.detector_flip_cols:
            frames = frames[..., ::-1]

        pre = self.preprocessed_geometry

        # multiply signal coordinates by upsampling factor
        coords = self.meta.coordinates * pre.upsampling_factor

        parallax_accumulate_cpu(
            frames,
            pre.bf_flat_inds,
            pre.shifts,
            coords,
            self.results.reconstruction,
        )

    def merge(self, dest, src):
        reconstruction = src.reconstruction
        upsampling_factor = self.preprocessed_geometry.upsampling_factor

        # Zero out largest spatial frequency to suppress Nyquist noise in integer shifts
        if self.params.suppress_Nyquist_noise and upsampling_factor > 1:
            reconstruction = suppress_nyquist_frequency(reconstruction)

        dest.reconstruction[:] += reconstruction
