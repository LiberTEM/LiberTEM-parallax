import numpy as np  # noqa: I001
import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises
from quantem.core.utils.utils import (
    electron_wavelength_angstrom as electron_wavelength_quantem,
)
from quantem.diffractive_imaging.complex_probe import (
    aberration_surface as aberration_surface_quantem,
    aberration_surface_cartesian_gradients as aberration_surface_cartesian_gradients_quantem,
    polar_coordinates as polar_coordinates_quantem,
    spatial_frequencies as spatial_frequencies_quantem,
)

from libertem_parallax.utils import (
    electron_wavelength as electron_wavelength_libertem,
    polar_coordinates as polar_coordinates_libertem,
    quadratic_aberration_cartesian_gradients as aberration_surface_cartesian_gradients_libertem,
    quadratic_aberration_surface as aberration_surface_libertem,
    spatial_frequencies as spatial_frequencies_libertem,
)

ENERGY = 300e3
SAMPLING = (0.1, 0.1)


class TestElectronWavelength:
    def test_matches_quantem(self):
        assert_equal(
            electron_wavelength_quantem(ENERGY),
            electron_wavelength_libertem(ENERGY),
        )


class TestSpatialFrequencies:
    @pytest.mark.parametrize(
        "gpts",
        [
            (64, 63),
            (65, 66),
        ],
    )
    def test_unrotated(self, gpts):
        quant = spatial_frequencies_quantem(gpts, SAMPLING)
        liber = spatial_frequencies_libertem(gpts, SAMPLING)
        assert_allclose(quant, liber)

    @pytest.mark.parametrize(
        "gpts, rotation_angle",
        [
            ((64, 63), np.pi / 3),
            ((65, 66), np.pi / 3),
        ],
    )
    def test_rotated(self, gpts, rotation_angle):
        quant = spatial_frequencies_quantem(gpts, SAMPLING, rotation_angle)
        liber = spatial_frequencies_libertem(gpts, SAMPLING, rotation_angle)
        assert_allclose(quant, liber, atol=1e-6)


class TestPolarCoordinates:
    @pytest.mark.parametrize(
        "gpts",
        [
            (64, 63),
            (65, 66),
        ],
    )
    def test_matches_quantem(self, gpts):
        kxa, kya = spatial_frequencies_quantem(gpts, SAMPLING)
        k_q, phi_q = polar_coordinates_quantem(kxa, kya)

        k_l, phi_l = polar_coordinates_libertem(
            kxa.numpy(),
            kya.numpy(),
        )

        assert_allclose(k_q.numpy(), k_l, atol=1e-6)
        assert_allclose(phi_q.numpy(), phi_l, atol=1e-6)


class TestAberrationSurface:
    @pytest.mark.parametrize(
        "gpts",
        [
            (64, 63),
            (65, 66),
        ],
    )
    @pytest.mark.parametrize(
        "aberration_coefs",
        [
            {"C10": 50.0},
            {"C10": 50.0, "C12": 25.0, "phi12": 0.1244},
        ],
    )
    def test_matches_quantem(self, gpts, aberration_coefs):
        wavelength = electron_wavelength_libertem(ENERGY)

        kxa, kya = spatial_frequencies_quantem(gpts, SAMPLING)
        k, phi = polar_coordinates_quantem(kxa, kya)
        alpha = k * wavelength

        quant = aberration_surface_quantem(
            alpha,
            phi,
            wavelength,
            aberration_coefs,
        )

        liber = aberration_surface_libertem(
            alpha.numpy(),
            phi.numpy(),
            wavelength,
            aberration_coefs,
        )

        assert_allclose(quant.numpy(), liber, atol=1e-5, rtol=1e-6)


class TestAberrationGradients:
    def test_higher_order_aberrations_diverge(self):
        """
        Libertem-parallax intentionally ignores higher-order aberrations.
        This test asserts that behavior differs silently from quantem.
        """
        gpts = (63, 64)
        aberration_coefs = {
            "C10": 50.0,
            "C12": 25.0,
            "phi12": 0.1244,
            "C21": 1e3,
        }

        wavelength = electron_wavelength_libertem(ENERGY)

        kxa, kya = spatial_frequencies_quantem(gpts, SAMPLING)
        k, phi = polar_coordinates_quantem(kxa, kya)
        alpha = k * wavelength

        with assert_raises(AssertionError):
            assert_allclose(
                aberration_surface_cartesian_gradients_quantem(
                    alpha,
                    phi,
                    aberration_coefs,
                ),
                aberration_surface_cartesian_gradients_libertem(
                    alpha.numpy(),
                    phi.numpy(),
                    aberration_coefs,
                ),
                atol=1e-5,
            )
