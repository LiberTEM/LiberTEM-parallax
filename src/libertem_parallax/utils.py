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
    Uses the same polar coefficients conventions as quantem and abTEM:
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
