"""Tests for ft_zeros.py — Fourier transform zeros method."""

import numpy as np
import pytest

from spotgp.ft_zeros import (
    broadening_function_ft,
    find_ft_zeros,
    q_ratio,
    q_ratio_from_kernel,
    rigid_rotation_q_ratio,
)
from spotgp.line_profile import rotational_broadening_kernel


class TestBroadeningFunctionFT:
    def test_output_shapes(self):
        v = np.linspace(-30, 30, 1024)
        G = np.array(rotational_broadening_kernel(v, vsini=10.0))
        sigma, G_hat = broadening_function_ft(v, G)
        assert sigma.shape == G_hat.shape
        assert len(sigma) > 0

    def test_peak_at_zero_frequency(self):
        """G_hat should be normalized to 1 at σ=0."""
        v = np.linspace(-30, 30, 1024)
        G = np.array(rotational_broadening_kernel(v, vsini=10.0))
        sigma, G_hat = broadening_function_ft(v, G)
        np.testing.assert_allclose(G_hat[0], 1.0, atol=0.01)

    def test_ft_decays_from_peak(self):
        """FT should be less than 1 away from σ=0."""
        v = np.linspace(-30, 30, 2048)
        G = np.array(rotational_broadening_kernel(v, vsini=10.0, epsilon=0.6))
        sigma, G_hat = broadening_function_ft(v, G)
        assert G_hat[0] > G_hat[5]
        assert G_hat[0] > G_hat[10]


class TestFindFTZeros:
    def test_finds_zeros_for_broadening_kernel(self):
        v = np.linspace(-30, 30, 4096)
        G = np.array(rotational_broadening_kernel(v, vsini=10.0, epsilon=0.6))
        sigma, G_hat = broadening_function_ft(v, G)
        q = find_ft_zeros(sigma, G_hat, n_zeros=3)
        assert not np.isnan(q[0])
        assert not np.isnan(q[1])
        assert q[1] > q[0] > 0

    def test_zeros_scale_with_vsini(self):
        """q_n ∝ 1/vsini — doubling vsini halves the zero positions."""
        v = np.linspace(-60, 60, 4096)
        G1 = np.array(rotational_broadening_kernel(v, vsini=10.0))
        G2 = np.array(rotational_broadening_kernel(v, vsini=20.0))
        _, Gh1 = broadening_function_ft(v, G1)
        _, Gh2 = broadening_function_ft(v, G2)
        sigma, _ = broadening_function_ft(v, G1)

        q1 = find_ft_zeros(sigma, Gh1, n_zeros=2)
        q2 = find_ft_zeros(sigma, Gh2, n_zeros=2)
        np.testing.assert_allclose(q1[0] / q2[0], 2.0, rtol=0.15)

    def test_returns_nan_when_no_zeros(self):
        """Should return NaN for unfound zeros."""
        sigma = np.linspace(0, 1, 100)
        G_hat = np.ones(100)  # constant — no zeros
        q = find_ft_zeros(sigma, G_hat, n_zeros=3)
        assert np.all(np.isnan(q))


class TestQRatio:
    def test_rigid_rotation_range(self):
        """q_2/q_1 for rigid rotation with ε=0.6 should be ~1.72–1.83."""
        ratio = rigid_rotation_q_ratio(vsini=15.0, epsilon=0.6)
        assert 1.5 < ratio < 2.0

    def test_rigid_rotation_independent_of_vsini(self):
        """q_2/q_1 should not depend on vsini for rigid rotation."""
        r1 = rigid_rotation_q_ratio(vsini=10.0, epsilon=0.6)
        r2 = rigid_rotation_q_ratio(vsini=20.0, epsilon=0.6)
        np.testing.assert_allclose(r1, r2, rtol=0.05)

    def test_nan_when_insufficient_zeros(self):
        sigma = np.linspace(0, 1, 100)
        G_hat = np.ones(100)
        ratio = q_ratio(sigma, G_hat)
        assert np.isnan(ratio)


class TestQRatioFromKernel:
    def test_returns_ratio_and_ft(self, default_hparam):
        from spotgp.spectral_temporal_kernel import SpectralTemporalKernel

        dv_grid = np.linspace(-20, 20, 81)
        sk = SpectralTemporalKernel(
            default_hparam, dv_grid, sigma_H=2.0, vsini=10.0,
            n_harmonics=3, n_lat=16, n_theta=64, n_v_internal=256,
        )
        ratio, sigma, G_hat = q_ratio_from_kernel(sk)
        assert sigma.shape == G_hat.shape
        assert isinstance(ratio, float)

    def test_ratio_is_finite(self, default_hparam):
        from spotgp.spectral_temporal_kernel import SpectralTemporalKernel

        dv_grid = np.linspace(-25, 25, 101)
        sk = SpectralTemporalKernel(
            default_hparam, dv_grid, sigma_H=2.0, vsini=10.0,
            n_harmonics=3, n_lat=32, n_theta=128, n_v_internal=512,
        )
        ratio, _, _ = q_ratio_from_kernel(sk)
        assert np.isfinite(ratio)
        assert ratio > 0
