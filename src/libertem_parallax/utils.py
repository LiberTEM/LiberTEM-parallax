# This file contains code adapted from the quantEM project
# (https://github.com/electronmicroscopy/quantem).
#
# Original code licensed under MIT license.
# Modifications have been made for use in libertem-parallax.

import numpy as np
from numpy.typing import NDArray


def electron_wavelength(energy: float) -> float:
    """
    Returns the relativistic electron wavelength in Angstroms.
    """
    m = 9.109383e-31
    e = 1.602177e-19
    c = 299792458
    h = 6.62607e-34

    lam = h / np.sqrt(2 * m * e * energy) / np.sqrt(1 + e * energy / 2 / m / c**2)
    return lam * 1e10


def spatial_frequencies(
    gpts: tuple[int, int],
    sampling: tuple[float, float],
    rotation_angle: float | None = None,
) -> tuple[NDArray, NDArray]:
    """
    Returns (optionally rotated) corner-centered spatial frequencies on a grid.
    """
    ny, nx = gpts
    sy, sx = sampling

    kx = np.fft.fftfreq(ny, sy)
    ky = np.fft.fftfreq(nx, sx)
    kxa, kya = np.meshgrid(kx, ky, indexing="ij")

    if rotation_angle is not None:
        c = np.cos(rotation_angle)
        s = np.sin(rotation_angle)
        kx_rot = c * kxa - s * kya
        ky_rot = s * kxa + c * kya
        kxa, kya = kx_rot, ky_rot

    return kxa, kya


def polar_coordinates(kx: NDArray, ky: NDArray) -> tuple[NDArray, NDArray]:
    """
    Converts cartesian to polar coordinates.
    """
    k = np.sqrt(kx**2 + ky**2)
    phi = np.arctan2(ky, kx)
    return k, phi


def quadratic_aberration_surface(
    alpha: NDArray, phi: NDArray, wavelength: float, aberration_coefs: dict[str, float]
) -> NDArray:
    """
    Evaluates the quadratic part of the aberration surface on a polar grid of angular frequencies.
    Uses the same polar coefficients conventions as abTEM:
    https://abtem.readthedocs.io/en/latest/user_guide/walkthrough/contrast_transfer_function.html#phase-aberrations
    """
    C10 = aberration_coefs.get("C10", 0.0)
    C12 = aberration_coefs.get("C12", 0.0)
    phi12 = aberration_coefs.get("phi12", 0.0)

    prefactor = np.pi / wavelength

    chi = prefactor * alpha**2 * (C10 + C12 * np.cos(2.0 * (phi - phi12)))

    return chi


def quadratic_aberration_cartesian_gradients(
    alpha: NDArray, phi: NDArray, aberration_coefs: dict[str, float]
) -> tuple[NDArray, NDArray]:
    """
    Evaluates the cartesian gradients of the quadratic part of the aberration surface on a polar grid of frequencies.
    """
    C10 = aberration_coefs.get("C10", 0.0)
    C12 = aberration_coefs.get("C12", 0.0)
    phi12 = aberration_coefs.get("phi12", 0.0)

    cos2 = np.cos(2.0 * (phi - phi12))
    sin2 = np.sin(2.0 * (phi - phi12))

    # dχ/dα and dχ/dφ
    scale = 2 * np.pi
    dchi_dk = scale * alpha * (C10 + C12 * cos2)
    dchi_dphi = -scale * alpha * (C12 * sin2)

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    dchi_dx = cos_phi * dchi_dk - sin_phi * dchi_dphi
    dchi_dy = sin_phi * dchi_dk + cos_phi * dchi_dphi

    return dchi_dx, dchi_dy


def suppress_nyquist_frequency(array: NDArray):
    """
    Zeros Nyquist frequencies of a real-space array.
    """
    F = np.fft.fft2(array)
    Nx, Ny = F.shape
    F[Nx // 2, :] = 0.0
    F[:, Ny // 2] = 0.0
    return np.fft.ifft2(F).real


def prepare_grouped_phase_flipping_kernel(H, s_m_up, upsampled_gpts):
    """
    Prepare the phase-flip kernel offsets and weights in NumPy.
    Parameters
    ----------
    H : np.ndarray, shape (h, w)
        Base kernel.
    s_m_up : np.ndarray, shape (M, 2)
        Up-sampled shifts for M BF pixels, [y, x].
    upsampled_gpts : tuple[int, int]
        (Ny, Nx) real-space grid size
    Returns
    -------
    unique_offsets : np.ndarray[int64], shape (U,)
        Flattened offsets for scatter-add.
    K : np.ndarray[float64], shape (U, M)
        Phase-flip weights for each unique offset and BF pixel.
    """
    Ny, Nx = upsampled_gpts
    h, w = H.shape
    M = s_m_up.shape[0]
    L0 = h * w

    # kernel grid
    dy = np.arange(h)
    dx = np.arange(w)
    dy_grid = np.repeat(dy, w)
    dx_grid = np.tile(dx, h)

    # repeat for M BF pixels
    dy_rep = np.tile(dy_grid, M)
    dx_rep = np.tile(dx_grid, M)

    # shifts repeated
    s_my = np.repeat(s_m_up[:, 0], L0)
    s_mx = np.repeat(s_m_up[:, 1], L0)

    # compute flattened offsets
    offsets = (dy_rep + s_my) * Nx + (dx_rep + s_mx)

    # find unique offsets and inverse indices
    unique_offsets, inv = np.unique(offsets, return_inverse=True)
    U = unique_offsets.size

    # build grouped kernel
    H_flat = H.ravel()
    H_all = np.tile(H_flat, M)
    m_idx = np.repeat(np.arange(M), L0)

    K = np.zeros((U, M), dtype=H.dtype)
    np.add.at(K, (inv, m_idx), H_all)  # accumulate values

    return unique_offsets.astype(np.int64), K
