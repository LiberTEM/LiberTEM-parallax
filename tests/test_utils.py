import numpy as np
import pytest

from libertem_parallax.utils import (
    electron_wavelength,
    polar_coordinates,
    prepare_grouped_phase_flipping_kernel,
    quadratic_aberration_cartesian_gradients,
    quadratic_aberration_surface,
    spatial_frequencies,
    suppress_nyquist_frequency,
)


class TestUtils:
    def test_electron_wavelength(self):
        lam = electron_wavelength(200e3)
        assert lam > 0.0
        # Monotonicity check: ~ 1/sqrt(E)
        lam2 = electron_wavelength(300e3)
        assert lam2 < lam

    @pytest.mark.parametrize(
        "gpts",
        [
            (5, 5),
            (5, 6),
            (6, 6),
            (6, 5),
        ],
    )
    def test_spatial_frequencies_no_rotation(self, gpts):
        sampling = (1.0, 2.0)
        kx, ky = spatial_frequencies(gpts, sampling)
        assert kx.shape == gpts
        assert ky.shape == gpts
        # Corner-centered, zero freq near center of FFT grid
        assert np.isclose(kx[0, 0], 0.0)
        assert np.isclose(ky[0, 0], 0.0)

    @pytest.mark.parametrize(
        "gpts",
        [
            (5, 5),
            (5, 6),
            (6, 6),
            (6, 5),
        ],
    )
    def test_spatial_frequencies_with_rotation(self, gpts):
        sampling = (1.0, 1.0)
        theta = np.pi / 2
        kx, ky = spatial_frequencies(gpts, sampling, rotation_angle=theta)

        # Rotation preserves magnitude of vectors
        k_mag, _ = polar_coordinates(kx, ky)
        kx0, ky0 = spatial_frequencies(gpts, sampling)
        k_mag0, _ = polar_coordinates(kx0, ky0)
        np.testing.assert_allclose(k_mag, k_mag0, atol=1e-12)

    def test_polar_coordinates_and_back(self):
        x = np.array([[1.0, 0.0], [0.0, -1.0]])
        y = np.array([[0.0, 1.0], [-1.0, 0.0]])
        k, phi = polar_coordinates(x, y)
        x2 = k * np.cos(phi)
        y2 = k * np.sin(phi)
        np.testing.assert_allclose(x, x2, atol=1e-12)
        np.testing.assert_allclose(y, y2, atol=1e-12)

    @pytest.mark.parametrize(
        "gpts",
        [
            (5, 5),
            (5, 6),
            (6, 6),
            (6, 5),
        ],
    )
    def test_quadratic_aberration_surface_and_gradients(self, gpts):
        sampling = (1.0, 1.0)
        theta = np.pi / 2

        kxa, kya = spatial_frequencies(gpts, sampling, rotation_angle=theta)
        k, phi = polar_coordinates(kxa, kya)

        aberration_coefs = {"C10": 1.0, "C12": 0.5, "phi12": 0.0}
        surface = quadratic_aberration_surface(k, phi, 1.0, aberration_coefs)
        dx, dy = quadratic_aberration_cartesian_gradients(k, phi, aberration_coefs)

        # Surface should be non-zero
        assert np.any(surface != 0)
        # Gradients should be finite
        assert np.all(np.isfinite(dx))
        assert np.all(np.isfinite(dy))

    @pytest.mark.parametrize(
        "gpts",
        [
            (5, 5),
            (5, 6),
            (6, 6),
            (6, 5),
        ],
    )
    def test_suppress_nyquist_frequency(self, gpts):
        arr = np.random.rand(*gpts)
        arr_suppressed = suppress_nyquist_frequency(arr)
        # Output is real
        assert np.isrealobj(arr_suppressed)

        # Rough check: Nyquist freq zeroed
        fft_arr = np.fft.fft2(arr_suppressed)
        Nx, Ny = fft_arr.shape
        if Nx % 2 == 0:
            assert np.allclose(fft_arr[Nx // 2, :], 0.0)
        if Ny % 2 == 0:
            assert np.allclose(fft_arr[:, Ny // 2], 0.0)

    @pytest.mark.parametrize(
        "upsampled_gpts",
        [
            (7, 8),
            (8, 8),
        ],
    )
    def test_kernel_weight_conservation(self, upsampled_gpts):
        kernel = np.ones((3, 3), dtype=np.float64)
        shifts = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
            ],
            dtype=np.int64,
        )

        offsets, grouped = prepare_grouped_phase_flipping_kernel(
            kernel,
            shifts,
            upsampled_gpts,
        )

        # Each BF pixel contributes sum(kernel)
        expected = kernel.sum()

        # Sum over spatial offsets → per BF pixel
        summed = grouped.sum(axis=0)

        np.testing.assert_allclose(
            summed,
            expected,
            rtol=0,
            atol=0,
        )
