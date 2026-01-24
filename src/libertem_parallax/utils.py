# This file contains code adapted from the quantEM project
#   https://github.com/electronmicroscopy/quantem.
#
#
# Original license:
#     MIT License

#     Copyright (c) 2025 ophusgroup

#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:

#     The above copyright notice and this permission notice shall be included in all
#     copies or substantial portions of the Software.

#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#     SOFTWARE.
#
# Modifications have been made for use in libertem-parallax.

import numpy as np
from numpy.typing import NDArray


def electron_wavelength(energy: float) -> float:
    """
    Returns the relativistic electron wavelength in Angstroms.
    Adapted from:
        https://github.com/electronmicroscopy/quantem/blob/dd7f29a0724eefd71fac8550fc757cbe9c7a8a74/src/quantem/core/utils/utils.py#L97
    """
    m = 9.109383e-31  # mass in SI (Kg)
    e = 1.602177e-19  # elementary charge in SI (C)
    c = 299792458  # speed of light in SI (m/s)
    h = 6.62607e-34  # planch constant in SI (Kg m^2 / s)

    lam = h / np.sqrt(2 * m * e * energy) / np.sqrt(1 + e * energy / 2 / m / c**2)
    return lam * 1e10  # convert from m to Angstroms


def spatial_frequencies(
    gpts: tuple[int, int],
    sampling: tuple[float, float],
    rotation_angle: float | None = None,
) -> tuple[NDArray, NDArray]:
    """
    Returns (optionally rotated) corner-centered spatial frequencies on a grid.
    Adapted from:
        https://github.com/electronmicroscopy/quantem/blob/dd7f29a0724eefd71fac8550fc757cbe9c7a8a74/src/quantem/diffractive_imaging/complex_probe.py#L392

    Rotation convention
    -------------------

    The ``rotation_angle`` parameter applies an **active, counter-clockwise (CCW)**
    rotation to the detector-frequency coordinates ``(kx, ky)``:

        (kx', ky') = R(+θ) · (kx, ky)

    with

        R(θ) = [[ cos(θ), -sin(θ)],
                [ sin(θ),  cos(θ)]]

    This corresponds to rotating the **detector frequency vectors themselves** in a
    fixed laboratory coordinate system.

    Important note on sign conventions
    ----------------------------------

    This active frequency-grid rotation differs from common image-space rotation
    functions such as ``scipy.ndimage.rotate``, which perform a **passive rotation**
    of the image content within a fixed array grid.

    As a result:

    - A **counter-clockwise rotation of the detector image by +θ** (e.g. using ``ndimage.rotate``)
    - corresponds to a **clockwise rotation of the frequency coordinates by −θ**

    Therefore, when a detector image has been rotated CCW by an angle ``θ`` in
    image space, the corresponding spatial-frequency grid must be rotated using

        rotation_angle = -θ
    """
    ny, nx = gpts
    sy, sx = sampling

    kx = np.fft.fftfreq(ny, sy)
    ky = np.fft.fftfreq(nx, sx)
    kxa, kya = np.meshgrid(kx, ky, indexing="ij")

    if rotation_angle is not None:
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)

        kxa_new = cos_theta * kxa - sin_theta * kya
        kya_new = sin_theta * kxa + cos_theta * kya

        kxa, kya = kxa_new, kya_new

    return kxa, kya


def polar_coordinates(kx: NDArray, ky: NDArray) -> tuple[NDArray, NDArray]:
    """
    Converts cartesian to polar coordinates.
    Adapted from:
        https://github.com/electronmicroscopy/quantem/blob/dd7f29a0724eefd71fac8550fc757cbe9c7a8a74/src/quantem/diffractive_imaging/complex_probe.py#L411
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

    aberration_surface = (
        prefactor * alpha**2 * (C10 + C12 * np.cos(2.0 * (phi - phi12)))
    )

    return aberration_surface


def quadratic_aberration_cartesian_gradients(
    alpha: NDArray, phi: NDArray, aberration_coefs: dict[str, float]
) -> tuple[NDArray, NDArray]:
    """
    Evaluates the cartesian gradients of the quadratic part of the aberration surface
    on a polar grid of frequencies. Adapted from:
        https://github.com/electronmicroscopy/quantem/blob/dd7f29a0724eefd71fac8550fc757cbe9c7a8a74/src/quantem/diffractive_imaging/complex_probe.py#L218
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
    fourier_array = np.fft.fft2(array)
    Nx, Ny = fourier_array.shape

    if Nx % 2 == 0:
        fourier_array[Nx // 2, :] = 0.0
    if Ny % 2 == 0:
        fourier_array[:, Ny // 2] = 0.0

    return np.fft.ifft2(fourier_array).real


def prepare_grouped_phase_flipping_kernel(kernel, shifts_m_upsampled, upsampled_gpts):
    """
    Prepare the phase-flip kernel offsets and weights in NumPy.
    Parameters
    ----------
    kernel : np.ndarray, shape (h, w)
        Base kernel.
    shifts_m_upsampled : np.ndarray, shape (M, 2)
        Up-sampled shifts for M BF pixels, [y, x].
    upsampled_gpts : tuple[int, int]
        (Ny, Nx) real-space grid size
    Returns
    -------
    unique_offsets : np.ndarray[int64], shape (U,)
        Flattened offsets for scatter-add.
    grouped_kernel : np.ndarray[float64], shape (U, M)
        Phase-flip weights for each unique offset and BF pixel.
    """
    Ny, Nx = upsampled_gpts
    h, w = kernel.shape
    M = shifts_m_upsampled.shape[0]
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
    s_my = np.repeat(shifts_m_upsampled[:, 0], L0)
    s_mx = np.repeat(shifts_m_upsampled[:, 1], L0)

    # compute flattened offsets (wrapped properly)
    offsets = ((dy_rep + s_my) % Ny) * Nx + ((dx_rep + s_mx) % Nx)

    # find unique offsets and inverse indices
    unique_offsets, inv = np.unique(offsets, return_inverse=True)
    U = unique_offsets.size

    # build grouped kernel
    H_flat = kernel.ravel()
    H_all = np.tile(H_flat, M)
    m_idx = np.repeat(np.arange(M), L0)

    grouped_kernel = np.zeros((U, M), dtype=kernel.dtype)
    np.add.at(grouped_kernel, (inv, m_idx), H_all)  # accumulate values

    return unique_offsets.astype(np.int64), grouped_kernel
