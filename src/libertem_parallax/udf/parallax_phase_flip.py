import numba
import numpy as np

from libertem_parallax.utils import (
    polar_coordinates,
    prepare_grouped_phase_flipping_kernel,
    quadratic_aberration_surface,
    spatial_frequencies,
)

from .base import BaseParallaxUDF


@numba.njit(fastmath=True, nogil=True, cache=True)
def parallax_phase_flip_accumulate_cpu(
    frames, bf_rows, bf_cols, coords, unique_offsets, K, out
):
    """
    Scatter-add phase-flip contributions into a real-space accumulator.

    This kernel performs:
    1. Mean subtraction over BF pixels for each frame.
    2. Weighting by the phase-flip kernel `K`.
    3. Scatter-add of contributions using `unique_offsets`.

    Parameters
    ----------
    frames : (T, sy, sx) float32/64
        Input frames (BF pixels)
    bf_rows, bf_cols : (M,) int32
        Row/col indices of BF pixels
    coords : (T, 2) int64
        Real-space navigation coordinates
    unique_offsets : (U,) int64
        Flattened offsets for the phase-flip kernel
    K : (U, M) float64
        Phase-flip weights for each unique offset and BF pixel
    out : (Ny, Nx) float64
        Real-space accumulator (in-place)
    """

    T = frames.shape[0]
    M = len(bf_rows)
    U = len(unique_offsets)
    Ny, Nx = out.shape

    for t in range(T):
        # Extract BF pixels and subtract per-frame mean
        I_bf = np.empty(M, dtype=np.float64)
        s = 0.0
        for m in range(M):
            s += frames[t, bf_rows[m], bf_cols[m]]
        mean = s / M
        for m in range(M):
            I_bf[m] = frames[t, bf_rows[m], bf_cols[m]] - mean

        # Compute contributions
        vals = np.empty(U, dtype=np.float64)
        for u in range(U):
            acc = 0.0
            for m in range(M):
                acc += K[u, m] * I_bf[m]
            vals[u] = acc

        # Scatter-add to accumulator
        yt, xt = coords[t]
        r_off = yt * Nx + xt
        for u in range(U):
            idx = (r_off + unique_offsets[u]) % (Ny * Nx)
            out.flat[idx] += vals[u]


class ParallaxPhaseFlipUDF(BaseParallaxUDF):
    """
    User-Defined Function for streaming phase-flipped parallax reconstruction.

    Accumulates mean-subtracted bright-field intensities into a
    real-space phase-flipped parallax reconstruction by indexing the phase-flipped
    kernel using integer pixel shifts derived from probe aberrations.

    Instances must be constructed via `from_parameters()`.
    Direct instantiation is not recommended to ensure
    consistent preprocessing and streaming-safe configuration.
    """

    def __init__(
        self, bf_flat_inds, unique_offsets, grouped_kernel, upsampling_factor, **kwargs
    ):
        super().__init__(
            bf_flat_inds=bf_flat_inds,
            unique_offsets=unique_offsets,
            grouped_kernel=grouped_kernel,
            upsampling_factor=upsampling_factor,
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
        Construct a ParallaxPhaseFlipUDF from acquisition parameters.

        This constructor computes:
        - reciprocal-space sampling
        - the bright-field mask defined by `semiangle_cutoff`
        - integer parallax shifts from aberration phase gradients
        - phase-flip kernel and unique offsets

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
            Polar aberration coefficients in Angstroms and radians using abTEM conventions:
            https://abtem.readthedocs.io/en/latest/user_guide/walkthrough/contrast_transfer_function.html#phase-aberrations
        rotation_angle
            Active, counter-clockwise rotation in radians of the detector frequency grid to match
            the spatial frequency grid. See ``libertem_parallax.utils.spatial_frequencies?``
            https://github.com/LiberTEM/LiberTEM-parallax/blob/main/src/libertem_parallax/utils.py#L59
        upsampling_factor
            Integer upsampling factor for the scan grid.
        suppress_Nyquist_noise
            Whether to suppress Nyquist-frequency artifacts in the kernel.
        """

        pre = cls.preprocess_geometry(
            gpts,
            scan_gpts,
            scan_sampling,
            energy,
            semiangle_cutoff,
            reciprocal_sampling,
            angular_sampling,
            aberration_coefs,
            rotation_angle,
            upsampling_factor,
        )

        if aberration_coefs is None:
            aberration_coefs = {}

        bf_flat_inds = pre.bf_flat_inds
        shifts = pre.shifts
        wavelength = pre.wavelength
        upsampled_gpts = pre.upsampled_gpts
        upsampled_sampling = pre.upsampled_sampling

        # Phase-flip kernel
        qxa, qya = spatial_frequencies(upsampled_gpts, upsampled_sampling)
        q, theta = polar_coordinates(qxa, qya)
        chi_q = quadratic_aberration_surface(
            q * wavelength,
            theta,
            wavelength,
            aberration_coefs=aberration_coefs,
        )
        sign_sin_chi_q = np.sign(np.sin(chi_q))

        if suppress_Nyquist_noise:
            Nx, Ny = sign_sin_chi_q.shape
            sign_sin_chi_q[Nx // 2, :] = 0.0
            sign_sin_chi_q[:, Ny // 2] = 0.0
        kernel = np.fft.ifft2(sign_sin_chi_q).real

        unique_offsets, grouped_kernel = prepare_grouped_phase_flipping_kernel(
            kernel, shifts, upsampled_gpts
        )

        return cls(
            bf_flat_inds=bf_flat_inds,
            unique_offsets=unique_offsets,
            grouped_kernel=grouped_kernel,
            upsampling_factor=upsampling_factor,
            **kwargs,
        )

    def process_partition(self, partition):
        frames = partition.data  # shape (T, sy, sx)

        # multiply signal coordinates by upsampling factor
        upsampling_factor = self.params.upsampling_factor
        coords = self.meta.coordinates * upsampling_factor

        bf_flat_inds = np.asarray(self.params.bf_flat_inds)
        bf_rows = bf_flat_inds // self.gpts[0]
        bf_cols = bf_flat_inds % self.gpts[0]

        parallax_phase_flip_accumulate_cpu(
            frames,
            bf_rows,
            bf_cols,
            coords,
            self.params.unique_offsets,
            self.params.grouped_kernel,
            self.results.reconstruction,
        )

    def merge(self, dest, src):
        dest.reconstruction[:] += src.reconstruction
