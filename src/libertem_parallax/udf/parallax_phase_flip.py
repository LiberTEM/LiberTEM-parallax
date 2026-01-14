import numba
import numpy as np
from libertem.udf import UDF

from libertem_parallax.utils import (
    electron_wavelength,
    polar_coordinates,
    prepare_grouped_phase_flipping_kernel,
    quadratic_aberration_cartesian_gradients,
    quadratic_aberration_surface,
    spatial_frequencies,
)


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


class ParallaxPhaseFlipUDF(UDF):
    """
    User-Defined Function for streaming phase-flipped parallax reconstruction.

    Accumulates mean-subtracted bright-field intensities into a
    real-space phase-flipped parallax reconstruction by indexing the phase-flipped
    kernel using integer pixel shifts derived from probe aberrations.
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

        qxa, qya = spatial_frequencies(
            upsampled_gpts,
            upsampled_sampling,
        )
        q, theta = polar_coordinates(qxa, qya)
        chi_q = quadratic_aberration_surface(
            q * wavelength,
            theta,
            wavelength,
            aberration_coefs=aberration_coefs,
        )
        sign_sin_chi_q = np.sign(np.sin(chi_q))

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
        frames = tile.data  # shape (T, sy, sx)
        coords = self.meta.coordinates  # shape (T, 2)

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

    def postprocess(self):
        # No extra post-processing needed
        pass
