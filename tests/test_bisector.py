"""Tests for bisector.py — bisector velocity span and BVS covariance."""

import numpy as np
import pytest

from spotgp.bisector import (
    compute_bisector,
    bisector_velocity_span,
    bvs_sensitivity_kernel,
    bvs_covariance,
)
from spotgp.line_profile import quiescent_line_profile


@pytest.fixture
def symmetric_profile():
    """Symmetric absorption line centered at v=0."""
    v = np.linspace(-20, 20, 401)
    I0 = quiescent_line_profile(v, vsini=5.0, sigma_H=2.0)
    return v, np.array(I0)


class TestComputeBisector:
    def test_output_shape(self, symmetric_profile):
        v, profile = symmetric_profile
        flux, bis = compute_bisector(v, profile, n_levels=15)
        assert flux.shape == (15,)
        assert bis.shape == (15,)

    def test_symmetric_profile_bisector_near_zero(self, symmetric_profile):
        """Bisector of a symmetric profile should be at v ≈ 0."""
        v, profile = symmetric_profile
        _, bis = compute_bisector(v, profile, n_levels=20)
        np.testing.assert_allclose(bis, 0.0, atol=0.05)

    def test_flux_levels_ordered(self, symmetric_profile):
        v, profile = symmetric_profile
        flux, _ = compute_bisector(v, profile, n_levels=20)
        assert np.all(np.diff(flux) > 0)

    def test_shifted_profile(self):
        """Bisector of a shifted profile should track the shift."""
        v = np.linspace(-20, 20, 801)
        shift = 1.5
        unshifted = np.array(quiescent_line_profile(v, vsini=5.0, sigma_H=2.0))
        shifted = np.interp(v, v + shift, unshifted)
        _, bis = compute_bisector(v, shifted, n_levels=20)
        np.testing.assert_allclose(bis, shift, atol=0.2)


class TestBisectorVelocitySpan:
    def test_symmetric_bvs_near_zero(self, symmetric_profile):
        """BVS of a symmetric profile should be ≈ 0."""
        v, profile = symmetric_profile
        bvs = bisector_velocity_span(v, profile)
        assert abs(bvs) < 0.05

    def test_returns_float(self, symmetric_profile):
        v, profile = symmetric_profile
        bvs = bisector_velocity_span(v, profile)
        assert isinstance(bvs, float)


class TestBVSSensitivityKernel:
    def test_shape(self, symmetric_profile):
        v, profile = symmetric_profile
        W = bvs_sensitivity_kernel(v, profile)
        assert W.shape == v.shape

    def test_antisymmetric_for_symmetric_profile(self, symmetric_profile):
        """W(v) ≈ -W(-v) for a symmetric profile (BVS is odd under reflection)."""
        v, profile = symmetric_profile
        W = bvs_sensitivity_kernel(v, profile)
        np.testing.assert_allclose(W, -W[::-1], atol=0.5 * np.max(np.abs(W)))

    def test_nonzero_in_line(self, symmetric_profile):
        """W should be non-negligible inside the line wings."""
        v, profile = symmetric_profile
        W = bvs_sensitivity_kernel(v, profile)
        assert np.max(np.abs(W)) > 0


class TestBVSCovariance:
    def test_bvs_covariance_shape(self, default_hparam, symmetric_profile):
        """BVS covariance should have shape (len(lag),)."""
        from spotgp.spectral_temporal_kernel import SpectralTemporalKernel

        v, profile = symmetric_profile
        dv_grid = np.linspace(-15, 15, 31)
        sk = SpectralTemporalKernel(
            default_hparam, dv_grid, sigma_H=2.0, vsini=5.0,
            n_harmonics=2, n_lat=16, n_theta=64, n_v_internal=256,
        )
        lags = np.array([0.0, 2.0, 5.0])
        C = bvs_covariance(lags, sk, v, profile)
        assert C.shape == (3,)

    def test_bvs_covariance_nonnegative_at_zero_lag(self, default_hparam, symmetric_profile):
        """Cov[BVS, BVS] at τ=0 should be non-negative."""
        from spotgp.spectral_temporal_kernel import SpectralTemporalKernel

        v, profile = symmetric_profile
        dv_grid = np.linspace(-15, 15, 31)
        sk = SpectralTemporalKernel(
            default_hparam, dv_grid, sigma_H=2.0, vsini=5.0,
            n_harmonics=2, n_lat=16, n_theta=64, n_v_internal=256,
        )
        C = bvs_covariance(np.array([0.0]), sk, v, profile)
        assert C[0] >= -1e-15
