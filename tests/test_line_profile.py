"""Tests for line_profile.py — rotational broadening and quiescent profiles."""

import numpy as np
import pytest
import jax.numpy as jnp

from spotgp.line_profile import (
    rotational_broadening_kernel,
    local_line_profile,
    convolve_profiles,
    quiescent_line_profile,
)


class TestRotationalBroadeningKernel:
    def test_normalization(self):
        """∫ G(v) dv = 1."""
        v = np.linspace(-20, 20, 2001)
        G = rotational_broadening_kernel(v, vsini=10.0, epsilon=0.6)
        integral = np.trapezoid(np.array(G), v)
        np.testing.assert_allclose(integral, 1.0, atol=1e-3)

    def test_zero_outside_vsini(self):
        """G(v) = 0 for |v| > vsini."""
        v = np.array([-15.0, -10.1, 10.1, 15.0])
        G = rotational_broadening_kernel(v, vsini=10.0, epsilon=0.6)
        np.testing.assert_allclose(np.array(G), 0.0, atol=1e-15)

    def test_symmetry(self):
        """G(v) = G(-v)."""
        v = np.linspace(-10, 10, 201)
        G = np.array(rotational_broadening_kernel(v, vsini=10.0, epsilon=0.6))
        np.testing.assert_allclose(G, G[::-1], atol=1e-12)

    def test_no_limb_darkening(self):
        """ε=0: G(v) ∝ √(1 - (v/vsini)²) (semicircle)."""
        v = np.linspace(-9.9, 9.9, 501)
        G = np.array(rotational_broadening_kernel(v, vsini=10.0, epsilon=0.0))
        expected = 2.0 * np.sqrt(1 - (v / 10.0) ** 2) / (np.pi * 10.0)
        np.testing.assert_allclose(G, expected, rtol=1e-6)

    def test_positive_inside(self):
        """G(v) > 0 for |v| < vsini."""
        v = np.linspace(-9.5, 9.5, 100)
        G = rotational_broadening_kernel(v, vsini=10.0, epsilon=0.6)
        assert np.all(np.array(G) > 0)

    def test_different_epsilon_shapes(self):
        """Different ε values should produce different profile shapes."""
        v = np.linspace(-10, 10, 201)
        G_low = np.array(rotational_broadening_kernel(v, vsini=10.0, epsilon=0.0))
        G_high = np.array(rotational_broadening_kernel(v, vsini=10.0, epsilon=0.9))
        assert not np.allclose(G_low, G_high)
        integral_low = np.trapezoid(G_low, v)
        integral_high = np.trapezoid(G_high, v)
        np.testing.assert_allclose(integral_low, integral_high, atol=1e-3)


class TestLocalLineProfile:
    def test_gaussian_shape(self):
        v = np.linspace(-10, 10, 201)
        H = np.array(local_line_profile(v, sigma_H=2.0, depth=1.0))
        expected = np.exp(-v ** 2 / 8.0)
        np.testing.assert_allclose(H, expected, atol=1e-12)

    def test_depth_scaling(self):
        v = np.array([0.0])
        H = local_line_profile(v, sigma_H=2.0, depth=0.5)
        assert float(H[0]) == pytest.approx(0.5)

    def test_peak_at_center(self):
        v = np.linspace(-10, 10, 201)
        H = np.array(local_line_profile(v, sigma_H=2.0))
        assert np.argmax(H) == 100


class TestConvolveProfiles:
    def test_convolve_gaussians(self):
        """Convolution of two Gaussians gives a Gaussian with σ² = σ₁² + σ₂²."""
        v = np.linspace(-30, 30, 601)
        s1, s2 = 2.0, 3.0
        f = np.exp(-v ** 2 / (2 * s1 ** 2)) / (s1 * np.sqrt(2 * np.pi))
        g = np.exp(-v ** 2 / (2 * s2 ** 2)) / (s2 * np.sqrt(2 * np.pi))
        conv = np.array(convolve_profiles(v, f, g))
        s_expected = np.sqrt(s1 ** 2 + s2 ** 2)
        expected = np.exp(-v ** 2 / (2 * s_expected ** 2)) / (s_expected * np.sqrt(2 * np.pi))
        np.testing.assert_allclose(conv, expected, atol=1e-4)


class TestQuiescentLineProfile:
    def test_continuum_far_from_center(self):
        """Profile should approach 1.0 far from line center."""
        v = np.linspace(-50, 50, 2001)
        I0 = np.array(quiescent_line_profile(v, vsini=5.0, sigma_H=2.0))
        assert I0[0] == pytest.approx(1.0, abs=0.01)
        assert I0[-1] == pytest.approx(1.0, abs=0.01)

    def test_absorption_at_center(self):
        """Profile minimum should be < 1 (absorption)."""
        v = np.linspace(-30, 30, 601)
        I0 = np.array(quiescent_line_profile(v, vsini=5.0, sigma_H=2.0))
        assert np.min(I0) < 0.95

    def test_symmetry(self):
        """I_0(v) = I_0(-v)."""
        v = np.linspace(-20, 20, 401)
        I0 = np.array(quiescent_line_profile(v, vsini=5.0, sigma_H=2.0))
        np.testing.assert_allclose(I0, I0[::-1], atol=1e-6)

    def test_broader_with_larger_vsini(self):
        """Larger vsini should produce a broader line."""
        v = np.linspace(-30, 30, 601)
        I0_slow = np.array(quiescent_line_profile(v, vsini=3.0, sigma_H=2.0))
        I0_fast = np.array(quiescent_line_profile(v, vsini=10.0, sigma_H=2.0))
        depth_slow = 1.0 - np.min(I0_slow)
        depth_fast = 1.0 - np.min(I0_fast)
        assert depth_fast < depth_slow

    def test_output_in_01_range(self):
        """Profile should stay in [0, 1] for reasonable parameters."""
        v = np.linspace(-30, 30, 601)
        I0 = np.array(quiescent_line_profile(v, vsini=5.0, sigma_H=2.0, depth=0.8))
        assert np.all(I0 >= -0.01)
        assert np.all(I0 <= 1.01)
