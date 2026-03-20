"""Tests for envelope advanced methods: Gamma_hat_sq, check_functions, SkewedGaussian, Exponential."""

import numpy as np
import pytest
import jax.numpy as jnp

from spotgp.envelope import (
    TrapezoidSymmetricEnvelope,
    TrapezoidAsymmetricEnvelope,
    SkewedGaussianEnvelope,
    ExponentialEnvelope,
)


class TestTrapezoidSymmetricGammaHat:
    def test_gamma_hat_shape(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        omega = jnp.linspace(0, 5, 50)
        Gh = env.Gamma_hat(omega)
        assert Gh.shape == (50,)

    def test_gamma_hat_zero_freq(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        Gh0 = float(env.Gamma_hat(jnp.array([0.0]))[0])
        assert Gh0 > 0


class TestTrapezoidAsymmetricGammaHat:
    def test_gamma_hat_shape(self):
        env = TrapezoidAsymmetricEnvelope(lspot=5.0, tau_em=0.5, tau_dec=1.5)
        omega = jnp.linspace(0, 5, 50)
        Gh = env.Gamma_hat(omega)
        assert Gh.shape == (50,)


class TestSkewedGaussianEnvelopeAdvanced:
    def test_gamma_hat_sq(self):
        env = SkewedGaussianEnvelope(sigma_sn=2.0, n_sn=-3.0)
        omega = jnp.linspace(0, 5, 50)
        Gh_sq = env.Gamma_hat_sq(omega)
        assert Gh_sq.shape == (50,)
        assert np.all(np.array(Gh_sq) >= 0)

    def test_R_Gamma_shape(self):
        env = SkewedGaussianEnvelope(sigma_sn=2.0, n_sn=-3.0)
        lags = jnp.linspace(0, 20, 50)
        R = env.R_Gamma(lags)
        assert R.shape == (50,)

    def test_R_Gamma_decays(self):
        env = SkewedGaussianEnvelope(sigma_sn=2.0, n_sn=-3.0)
        R0 = float(env.R_Gamma(jnp.array([0.0]))[0])
        R10 = float(env.R_Gamma(jnp.array([10.0]))[0])
        assert R0 > R10

    def test_gaussian_case(self):
        """n_sn=0 gives a Gaussian envelope."""
        env = SkewedGaussianEnvelope(sigma_sn=2.0, n_sn=0.0)
        assert env.tau_spot == pytest.approx(2.0)
        g = float(env.Gamma(jnp.array(0.0)))
        assert g == pytest.approx(1.0)

    def test_kernel_support(self):
        env = SkewedGaussianEnvelope(sigma_sn=2.0, n_sn=-3.0)
        ks = env.kernel_support()
        assert ks > 0


class TestExponentialEnvelopeAdvanced:
    def test_Gamma_hat_sq(self):
        env = ExponentialEnvelope(tau_spot=2.0)
        omega = jnp.linspace(0, 5, 50)
        Gh_sq = env.Gamma_hat_sq(omega)
        assert Gh_sq.shape == (50,)
        assert np.all(np.array(Gh_sq) >= 0)

    def test_R_Gamma_shape(self):
        env = ExponentialEnvelope(tau_spot=2.0)
        lags = jnp.linspace(0, 20, 50)
        R = env.R_Gamma(lags)
        assert R.shape == (50,)

    def test_R_Gamma_zero_lag_max(self):
        env = ExponentialEnvelope(tau_spot=2.0)
        lags = jnp.linspace(0, 10, 100)
        R = np.array(env.R_Gamma(lags))
        assert R[0] >= np.max(R) - 1e-10

    def test_param_dict(self):
        env = ExponentialEnvelope(tau_spot=2.0)
        pd = env.param_dict
        assert "tau_spot" in pd
        assert pd["tau_spot"] == 2.0

    def test_kernel_support(self):
        env = ExponentialEnvelope(tau_spot=2.0)
        ks = env.kernel_support()
        assert ks > 0


class TestCheckFunctions:
    @pytest.fixture(autouse=True)
    def _disable_tex(self):
        """Disable LaTeX rendering so tests work without a TeX installation."""
        import matplotlib
        matplotlib.rcParams["text.usetex"] = False
        yield
        matplotlib.rcParams["text.usetex"] = False

    def test_trapezoid_symmetric(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        result = env.check_functions(show=False)
        assert "R_Gamma" in result
        assert "Gamma_hat" in result
        # Max error should be reasonable for the closed-form
        assert result["R_Gamma"]["rmse"] < 0.1
        assert result["Gamma_hat"]["rmse"] < 0.1

    def test_trapezoid_asymmetric(self):
        env = TrapezoidAsymmetricEnvelope(lspot=5.0, tau_em=0.5, tau_dec=1.5)
        result = env.check_functions(show=False)
        assert "R_Gamma" in result
        assert "rmse" in result["R_Gamma"]

    def test_exponential(self):
        env = ExponentialEnvelope(tau_spot=2.0)
        result = env.check_functions(show=False)
        assert isinstance(result, dict)
