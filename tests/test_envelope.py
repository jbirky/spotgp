"""Tests for src.envelope — envelope functions and autocorrelation."""

import numpy as np
import pytest
import jax.numpy as jnp

from spotgp.envelope import (
    TrapezoidSymmetricEnvelope,
    TrapezoidAsymmetricEnvelope,
    SkewedGaussianEnvelope,
    ExponentialEnvelope,
    compute_R_Gamma_numerical,
    _R_Gamma_symmetric,
    _R_Gamma_asymmetric,
    _Gamma_hat,
)


# =====================================================================
# Closed-form helpers
# =====================================================================

class TestRGammaSymmetric:
    def test_zero_lag(self):
        ell, tau = 5.0, 1.0
        R0 = _R_Gamma_symmetric(jnp.array([0.0]), ell, tau)
        np.testing.assert_allclose(float(R0[0]), ell + 2 * tau / 5, rtol=1e-12)

    def test_symmetry(self):
        lags = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        R = np.array(_R_Gamma_symmetric(lags, 5.0, 1.0))
        np.testing.assert_allclose(R[0], R[4], rtol=1e-10)
        np.testing.assert_allclose(R[1], R[3], rtol=1e-10)

    def test_beyond_support(self):
        ell, tau = 5.0, 1.0
        R = _R_Gamma_symmetric(jnp.array([ell + 2 * tau + 1.0]), ell, tau)
        np.testing.assert_allclose(float(R[0]), 0.0, atol=1e-12)

    def test_non_negative(self):
        lags = jnp.linspace(0, 10, 200)
        R = np.array(_R_Gamma_symmetric(lags, 5.0, 1.0))
        assert np.all(R >= -1e-15)

    def test_monotone_decreasing(self):
        lags = jnp.linspace(0, 6.9, 200)
        R = np.array(_R_Gamma_symmetric(lags, 5.0, 1.0))
        assert np.all(np.diff(R) <= 1e-10)


class TestRGammaAsymmetric:
    def test_zero_lag_positive(self):
        R0 = _R_Gamma_asymmetric(jnp.array([0.0]), 5.0, 0.5, 1.5)
        assert float(R0[0]) > 0

    def test_symmetry(self):
        lags = jnp.array([-2.0, 2.0])
        R = np.array(_R_Gamma_asymmetric(lags, 5.0, 0.5, 1.5))
        np.testing.assert_allclose(R[0], R[1], rtol=1e-10)

    def test_beyond_support(self):
        ell, te, td = 5.0, 0.5, 1.5
        R = _R_Gamma_asymmetric(jnp.array([ell + te + td + 1.0]), ell, te, td)
        np.testing.assert_allclose(float(R[0]), 0.0, atol=1e-12)


class TestGammaHat:
    def test_zero_frequency(self):
        ell, tau = 5.0, 1.0
        Gh = _Gamma_hat(jnp.array([0.0]), ell, tau)
        np.testing.assert_allclose(float(Gh[0]), ell + 2 * tau / 3, rtol=1e-10)

    def test_vectorized(self):
        omega = jnp.linspace(-5, 5, 100)
        Gh = _Gamma_hat(omega, 5.0, 1.0)
        assert Gh.shape == (100,)

    def test_even_symmetry(self):
        omega = jnp.linspace(0.1, 10, 50)
        Gh_pos = np.array(_Gamma_hat(omega, 5.0, 1.0))
        Gh_neg = np.array(_Gamma_hat(-omega, 5.0, 1.0))
        np.testing.assert_allclose(Gh_pos, Gh_neg, rtol=1e-10)


# =====================================================================
# TrapezoidSymmetricEnvelope
# =====================================================================

class TestTrapezoidSymmetricEnvelope:
    def test_init(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        assert env.tau_spot == 1.0
        assert env.lspot == 5.0

    def test_gamma_peak_is_one(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        # At t=0 (center of plateau), Gamma should be 1
        assert float(env.Gamma(jnp.array(0.0))) == pytest.approx(1.0)

    def test_gamma_zero_outside_support(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        t_far = env.lspot / 2 + env.tau_spot + 1.0
        assert float(env.Gamma(jnp.array(t_far))) == pytest.approx(0.0, abs=1e-12)

    def test_R_Gamma_matches_closed_form(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        lags = jnp.linspace(0, 8, 50)
        R_env = np.array(env.R_Gamma(lags))
        R_cf = np.array(_R_Gamma_symmetric(lags, 5.0, 1.0))
        np.testing.assert_allclose(R_env, R_cf, rtol=1e-6)

    def test_kernel_support(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        # Support is lspot + 2*tau_spot (where R_Gamma drops to zero)
        assert env.kernel_support() == 5.0 + 2 * 1.0

    def test_param_dict(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        pd = env.param_dict
        assert "lspot" in pd
        assert "tau_spot" in pd


# =====================================================================
# TrapezoidAsymmetricEnvelope
# =====================================================================

class TestTrapezoidAsymmetricEnvelope:
    def test_init(self):
        env = TrapezoidAsymmetricEnvelope(lspot=5.0, tau_em=0.5, tau_dec=1.5)
        assert env.tau_spot == pytest.approx(1.0)
        assert env.lspot == 5.0

    def test_gamma_peak_is_one(self):
        env = TrapezoidAsymmetricEnvelope(lspot=5.0, tau_em=0.5, tau_dec=1.5)
        assert float(env.Gamma(jnp.array(0.0))) == pytest.approx(1.0)

    def test_R_Gamma_matches_closed_form(self):
        env = TrapezoidAsymmetricEnvelope(lspot=5.0, tau_em=0.5, tau_dec=1.5)
        lags = jnp.linspace(0, 8, 50)
        R_env = np.array(env.R_Gamma(lags))
        R_cf = np.array(_R_Gamma_asymmetric(lags, 5.0, 0.5, 1.5))
        np.testing.assert_allclose(R_env, R_cf, rtol=1e-6)

    def test_reduces_to_symmetric_when_equal(self):
        env_asym = TrapezoidAsymmetricEnvelope(lspot=5.0, tau_em=1.0, tau_dec=1.0)
        env_sym = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        lags = jnp.linspace(0, 8, 50)
        R_asym = np.array(env_asym.R_Gamma(lags))
        R_sym = np.array(env_sym.R_Gamma(lags))
        np.testing.assert_allclose(R_asym, R_sym, rtol=1e-6)


# =====================================================================
# SkewedGaussianEnvelope
# =====================================================================

class TestSkewedGaussianEnvelope:
    def test_init(self):
        env = SkewedGaussianEnvelope(sigma_sn=2.0, n_sn=-3.0)
        assert env.tau_spot == pytest.approx(2.0)

    def test_gamma_peak_is_one(self):
        env = SkewedGaussianEnvelope(sigma_sn=2.0, n_sn=0.0)
        # For n_sn=0 this is a Gaussian centered at 0
        assert float(env.Gamma(jnp.array(0.0))) == pytest.approx(1.0)

    def test_R_Gamma_zero_lag_positive(self):
        env = SkewedGaussianEnvelope(sigma_sn=2.0, n_sn=-3.0)
        R0 = float(env.R_Gamma(jnp.array([0.0]))[0])
        assert R0 > 0


# =====================================================================
# ExponentialEnvelope
# =====================================================================

class TestExponentialEnvelope:
    def test_init(self):
        env = ExponentialEnvelope(tau_spot=2.0)
        assert env.tau_spot == 2.0

    def test_gamma_peak_is_one(self):
        env = ExponentialEnvelope(tau_spot=2.0)
        assert float(env.Gamma(jnp.array(0.0))) == pytest.approx(1.0)

    def test_gamma_decays(self):
        env = ExponentialEnvelope(tau_spot=2.0)
        g0 = float(env.Gamma(jnp.array(0.0)))
        g1 = float(env.Gamma(jnp.array(5.0)))
        assert g1 < g0

    def test_R_Gamma_zero_lag_positive(self):
        env = ExponentialEnvelope(tau_spot=2.0)
        R0 = float(env.R_Gamma(jnp.array([0.0]))[0])
        assert R0 > 0


# =====================================================================
# compute_R_Gamma_numerical
# =====================================================================

class TestComputeRGammaNumerical:
    def test_returns_valid_output(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        lag_grid, R_vals = compute_R_Gamma_numerical(env.Gamma, tau_ref=1.0)
        assert len(lag_grid) == len(R_vals)
        assert len(lag_grid) > 0
        # R(0) should be the maximum (positive)
        assert R_vals[0] > 0
        assert R_vals[0] >= np.max(R_vals) - 1e-10
