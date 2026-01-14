import numba
import numpy as np
from libertem.udf import UDF

from libertem_parallax.utils import (
    electron_wavelength,
    polar_coordinates,
    quadratic_aberration_cartesian_gradients,
    spatial_frequencies,
    suppress_nyquist_frequency,
)


@numba.njit(fastmath=True, nogil=True, cache=True)
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


class ParallaxUDF(UDF):
    """
    User-Defined Function for streaming parallax reconstruction.

    Accumulates mean-subtracted bright-field intensities into a
    real-space parallax reconstruction using integer pixel shifts
    derived from probe aberrations.
    """

    def __init__(
        self, bf_flat_inds, shifts, upsampling_factor, suppress_Nyquist_noise, **kwargs
    ):
        super().__init__(
            bf_flat_inds=bf_flat_inds,
            shifts=shifts,
            upsampling_factor=upsampling_factor,
            suppress_Nyquist_noise=suppress_Nyquist_noise,
            **kwargs,
        )

    @classmethod
    def from_parameters(
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
        suppress_Nyquist_noise: bool = True,
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
        suppress_Nyquist_noise
            Whether to suppress Nyquist-frequency artifacts at merge time.
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

        return cls(
            bf_flat_inds=bf_flat_inds,
            shifts=shifts,
            upsampling_factor=upsampling_factor,
            suppress_Nyquist_noise=suppress_Nyquist_noise,
            **kwargs,
        )

    def get_result_buffers(self):
        return {
            "reconstruction": self.buffer(
                kind="single",
                dtype=np.float64,
                extra_shape=self.upsampled_scan_gpts,
            )
        }

    @property
    def gpts(self) -> tuple[int, int]:
        return self.meta.dataset_shape.sig  # ty:ignore[invalid-return-type]

    @property
    def scan_gpts(self) -> tuple[int, int]:
        return self.meta.dataset_shape.nav  # ty:ignore[invalid-return-type]

    @property
    def upsampled_scan_gpts(self) -> tuple[int, int]:
        upsampling_factor: int = self.params.upsampling_factor  # ty:ignore[invalid-assignment]
        return (
            self.scan_gpts[0] * upsampling_factor,
            self.scan_gpts[1] * upsampling_factor,
        )

    def process_tile(self, tile):
        frames = tile.data

        # multiply signal coordinates by upsampling factor
        upsampling_factor = self.params.upsampling_factor
        coords = self.meta.coordinates * upsampling_factor

        parallax_accumulate_cpu(
            frames,
            self.params.bf_flat_inds,
            self.params.shifts,
            coords,
            self.results.reconstruction,
        )

    def merge(self, dest, src):
        reconstruction = src.reconstruction
        upsampling_factor: int = self.params.upsampling_factor  # ty:ignore[invalid-assignment]

        # Zero out largest spatial frequency to suppress Nyquist noise in integer shifts
        if self.params.suppress_Nyquist_noise and upsampling_factor > 1:
            reconstruction = suppress_nyquist_frequency(reconstruction)

        dest.reconstruction[:] += reconstruction

    def postprocess(self):
        # No postprocessing required
        pass
