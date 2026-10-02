"""Tests for src.envelope — envelope functions and autocorrelation."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from spotgp.envelope import (
    TrapezoidSymmetricEnvelope,
    TrapezoidAsymmetricEnvelope,
    SkewedGaussianEnvelope,
    ExponentialEnvelope,
    ModulatedGammaEnvelope,
    compute_R_Gamma_numerical,
    _R_Gamma_symmetric,
    _R_Gamma_symmetric_numerical,
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
# TrapezoidSymmetricEnvelope(numerical=True)
# =====================================================================

def _brute_force_R_Gamma(ell, tau, lags, dt=2e-4):
    """Direct Riemann-sum autocorrelation of the squared trapezoid."""
    h = ell / 2 + tau
    t = np.arange(-h - 1.0, h + max(lags) + 1.0, dt)

    def gamma(x):
        x = np.abs(x)
        return np.where(x <= ell / 2, 1.0, np.where(x < h, (h - x) / tau, 0.0))

    g = gamma(t) ** 2
    return np.array([np.sum(g * gamma(t + L) ** 2) * dt for L in lags])


class TestTrapezoidSymmetricNumerical:
    def test_default_is_analytic(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        assert env.numerical is False
        assert env.io_attrs == {}

    def test_io_attrs_round_trip_through_floats(self):
        # save_gp writes io_attrs as floats and load_gp passes them back
        # to the constructor.
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0,
                                         numerical=True, n_grid=2048)
        env2 = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0, **env.io_attrs)
        assert env2.numerical is True
        assert env2.n_grid == 2048

    @pytest.mark.parametrize("ell,tau", [(5.0, 1.0), (100.0, 1.0),
                                         (50.0, 5.0), (20.0, 20.0)])
    def test_matches_closed_form_where_valid(self, ell, tau):
        env_a = TrapezoidSymmetricEnvelope(lspot=ell, tau_spot=tau)
        env_n = TrapezoidSymmetricEnvelope(lspot=ell, tau_spot=tau, numerical=True)
        lags = jnp.linspace(0, ell + 2 * tau + 1.0, 500)
        R_a = np.array(env_a.R_Gamma(lags))
        R_n = np.array(env_n.R_Gamma(lags))
        np.testing.assert_allclose(R_n, R_a, atol=1e-5 * R_a[0])

    def test_grid_convergence(self):
        lags = jnp.linspace(0, 12.0, 300)
        R_cf = np.array(_R_Gamma_symmetric(lags, 10.0, 1.0))
        err = [np.max(np.abs(np.array(_R_Gamma_symmetric_numerical(
                   lags, 10.0, 1.0, n_grid=n)) - R_cf)) / R_cf[0]
               for n in (512, 4096)]
        assert err[0] < 1e-4
        assert err[1] < err[0]

    @pytest.mark.parametrize("ell,tau", [(5.0, 50.0), (1.0, 100.0), (10.0, 12.0)])
    def test_valid_where_closed_form_is_not(self, ell, tau):
        with pytest.raises(ValueError, match="closed form is invalid"):
            TrapezoidSymmetricEnvelope(lspot=ell, tau_spot=tau).R_Gamma(
                jnp.array([0.0, 1.0]))
        env = TrapezoidSymmetricEnvelope(lspot=ell, tau_spot=tau, numerical=True)
        lags = np.linspace(0, ell + 2 * tau + 1.0, 120)
        R = np.array(env.R_Gamma(jnp.asarray(lags)))
        np.testing.assert_allclose(R, _brute_force_R_Gamma(ell, tau, lags),
                                   atol=2e-6 * R[0])
        np.testing.assert_allclose(R[0], ell + 2 * tau / 5, rtol=1e-5)
        assert np.all(R >= -1e-12 * R[0])
        assert R[-1] == 0.0          # beyond the support ell + 2 tau

    def test_toeplitz_positive_semidefinite_ell_lt_tau(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=50.0, numerical=True)
        x = np.arange(0.0, 120.0, 0.5)
        lag = np.abs(x[:, None] - x[None, :])
        K = np.array(env.R_Gamma(jnp.asarray(lag.ravel()))).reshape(lag.shape)
        eig = np.linalg.eigvalsh(K)
        assert eig.min() > -1e-9 * eig.max()

    def test_r_gamma_jax_traceable_and_differentiable(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=50.0, numerical=True)
        lags = jnp.linspace(0.3, 104.0, 40)
        theta = jnp.array([5.0, 50.0])
        R_jit = np.array(jax.jit(env.r_gamma_jax)(theta, lags))
        np.testing.assert_allclose(R_jit, np.array(env.R_Gamma(lags)),
                                   rtol=1e-10, atol=1e-10)
        J = np.array(jax.jacfwd(lambda th: env.r_gamma_jax(th, lags))(theta))
        assert np.all(np.isfinite(J))
        eps = 1e-3
        for i in range(2):
            d = np.zeros(2)
            d[i] = eps
            fd = (np.array(env.r_gamma_jax(theta + d, lags))
                  - np.array(env.r_gamma_jax(theta - d, lags))) / (2 * eps)
            np.testing.assert_allclose(J[:, i], fd, atol=1e-3 * np.max(np.abs(fd)))

    def test_marginalized_distribution_uses_numerical_path(self):
        from spotgp.distributions import UniformDistribution
        # Quadrature nodes for lspot ~ U(2, 6) all lie below tau_spot = 8,
        # so the analytic average is out of domain but the FFT path is not.
        with pytest.raises(ValueError, match="closed form is invalid"):
            TrapezoidSymmetricEnvelope(lspot=UniformDistribution(2.0, 6.0),
                                       tau_spot=8.0).R_Gamma(jnp.array([0.0]))
        env = TrapezoidSymmetricEnvelope(lspot=UniformDistribution(2.0, 6.0),
                                         tau_spot=8.0, numerical=True)
        lags = jnp.linspace(0.0, 25.0, 60)
        R = np.array(env.R_Gamma(lags))
        assert np.all(np.isfinite(R)) and np.all(R >= -1e-12)
        # E[R(0)] = E[lspot] + 2 tau / 5 for a uniform lspot distribution
        np.testing.assert_allclose(R[0], 4.0 + 2 * 8.0 / 5, rtol=1e-5)
        assert R[-1] == 0.0          # beyond max support 6 + 16 = 22

    def test_gamma_hat_and_support_unchanged(self):
        # The closed-form Fourier transform holds for any lspot, tau_spot,
        # so the numerical option only replaces R_Gamma.
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=50.0, numerical=True)
        assert float(env.Gamma_hat(jnp.array(0.0))) == pytest.approx(5.0 + 2 * 50.0 / 3)
        assert env.kernel_support() == 5.0 + 2 * 50.0
        assert float(env.Gamma(jnp.array(0.0))) == 1.0


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
# ModulatedGammaEnvelope
# =====================================================================

class TestModulatedGammaEnvelope:
    def test_init(self):
        env = ModulatedGammaEnvelope(alpha=2.0, tau=10.0, a=0.3, omega=0.5)
        assert env.tau_spot == 10.0
        assert env.alpha == 2.0
        assert env.a == 0.3
        assert env.omega == 0.5
        assert env.lspot == 0.0

    def test_gamma_peak_is_one(self):
        env = ModulatedGammaEnvelope(alpha=2.0, tau=10.0, a=0.3, omega=0.5)
        t = jnp.linspace(-80, 80, 2000)
        g = np.array(env.Gamma(t))
        np.testing.assert_allclose(np.max(g), 1.0, rtol=1e-3)

    def test_gamma_zero_at_origin(self):
        env = ModulatedGammaEnvelope(alpha=2.0, tau=10.0, a=0.0, omega=0.0)
        assert float(env.Gamma(jnp.array(0.0))) == pytest.approx(0.0, abs=1e-12)

    def test_gamma_non_negative(self):
        env = ModulatedGammaEnvelope(alpha=2.0, tau=10.0, a=0.8, omega=0.5)
        t = jnp.linspace(-80, 80, 2000)
        g = np.array(env.Gamma(t))
        assert np.all(g >= -1e-10)

    def test_gamma_decays_at_large_t(self):
        env = ModulatedGammaEnvelope(alpha=2.0, tau=10.0, a=0.3, omega=0.5)
        g_far = float(env.Gamma(jnp.array(200.0)))
        assert g_far < 1e-5

    def test_R_Gamma_zero_lag_positive(self):
        env = ModulatedGammaEnvelope(alpha=2.0, tau=10.0, a=0.3, omega=0.5)
        R0 = float(env.R_Gamma(jnp.array([0.0]))[0])
        assert R0 > 0

    def test_R_Gamma_decays(self):
        env = ModulatedGammaEnvelope(alpha=2.0, tau=10.0, a=0.3, omega=0.5)
        R0 = float(env.R_Gamma(jnp.array([0.0]))[0])
        R_far = float(env.R_Gamma(jnp.array([100.0]))[0])
        assert R0 > R_far

    def test_param_dict(self):
        env = ModulatedGammaEnvelope(alpha=2.0, tau=10.0, a=0.3, omega=0.5)
        pd = env.param_dict
        assert "alpha_env" in pd
        assert "tau_spot" in pd
        assert "a_mod" in pd
        assert "omega_mod" in pd

    def test_kernel_support(self):
        env = ModulatedGammaEnvelope(alpha=2.0, tau=10.0, a=0.3, omega=0.5)
        ks = env.kernel_support()
        assert ks > 0

    def test_positivity_constraint_raises(self):
        with pytest.raises(ValueError, match="positivity"):
            ModulatedGammaEnvelope(alpha=2.0, tau=10.0, a=1.0, omega=0.5)

    def test_reduces_to_unmodulated(self):
        """With a=0, should match the unmodulated gamma shape (one-sided)."""
        env = ModulatedGammaEnvelope(alpha=2.0, tau=10.0, a=0.0, omega=0.5)
        t = jnp.linspace(-50, 50, 500)
        g = np.array(env.Gamma(t))
        # unmodulated: t^alpha * exp(-t/tau) for t>0, zero otherwise
        t_np = np.array(t)
        raw = np.where(t_np > 0,
                       (t_np / 10.0) ** 2 * np.exp(-t_np / 10.0),
                       0.0)
        raw /= raw.max()
        np.testing.assert_allclose(g, raw, atol=1e-4)


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
