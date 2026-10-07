"""Tests for ExponentialPlateauEnvelope, the three-parameter exponential profile."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.integrate import quad

from spotgp.envelope import (
    ExponentialPlateauEnvelope,
    ExponentialAsymmetricEnvelope,
    _exp_plateau_gamma,
    _exp_plateau_Gamma_hat_re,
    _exp_plateau_Gamma_hat_im,
    _exp_plateau_R_Gamma,
)

# (tau_em, tau_plat, tau_dec): asymmetric both ways, equal timescales, no plateau,
# a plateau shorter than the timescales, and a long plateau
CASES = [(2.0, 10.0, 5.0), (5.0, 10.0, 2.0), (3.0, 10.0, 3.0), (2.0, 0.0, 5.0),
         (4.0, 2.0, 4.0), (1.0, 30.0, 8.0)]


def gamma_paper(t, te, tp, td):
    """Gamma = alpha_exp^2 / alpha_max^2 with the plateau on [0, tp], as typeset."""
    t = np.asarray(t, dtype=float)
    return np.where(t < 0, np.exp(2 * t / te),
                    np.where(t <= tp, 1.0, np.exp(-2 * (t - tp) / td)))


def gamma_hat_paper(w, te, tp, td):
    """eq:Gamma_hat_exp_R and eq:Gamma_hat_exp_I as typeset, for w != 0."""
    ce, cd = 4 + w**2 * te**2, 4 + w**2 * td**2
    re = (2 * te / ce + np.sin(w * tp) / w
          + 2 * td * np.cos(w * tp) / cd - w * td**2 * np.sin(w * tp) / cd)
    im = (-w * te**2 / ce + (1 - np.cos(w * tp)) / w
          + w * td**2 * np.cos(w * tp) / cd + 2 * td * np.sin(w * tp) / cd)
    return re, im


def autocorrelation_numerical(te, tp, td, lags, dt=2e-3):
    """R(lag) = int Gamma(s) Gamma(s + lag) ds on a fine grid (FFT correlation)."""
    t = np.arange(-12 * te, tp + 12 * td, dt)
    g = gamma_paper(t, te, tp, td)
    n = 2 * len(t)
    R = np.fft.irfft(np.abs(np.fft.rfft(g, n)) ** 2, n)[:len(t)] * dt
    return np.interp(lags, np.arange(len(t)) * dt, R)


class TestGamma:
    @pytest.mark.parametrize("te,tp,td", CASES)
    def test_helper_matches_paper(self, te, tp, td):
        t = np.linspace(-5 * te, tp + 5 * td, 2001)
        np.testing.assert_allclose(np.asarray(_exp_plateau_gamma(t, te, tp, td)),
                                   gamma_paper(t, te, tp, td), rtol=1e-14, atol=0)

    def test_plateau_is_centred_with_unit_peak(self):
        env = ExponentialPlateauEnvelope(tau_em=2.0, tau_plat=10.0, tau_dec=5.0)
        t = jnp.array([-5.0, 0.0, 5.0])
        np.testing.assert_allclose(np.asarray(env.Gamma(t)), 1.0)
        assert float(env.Gamma(jnp.array(-6.0))) < 1.0
        assert float(env.Gamma(jnp.array(6.0))) < 1.0

    def test_e_folding_of_the_radius(self):
        # alpha e-folds over tau_em / tau_dec, so Gamma = alpha^2 drops to e^-2
        env = ExponentialPlateauEnvelope(tau_em=2.0, tau_plat=10.0, tau_dec=5.0)
        assert float(env.Gamma(jnp.array(-5.0 - 2.0))) == pytest.approx(np.exp(-2))
        assert float(env.Gamma(jnp.array(5.0 + 5.0))) == pytest.approx(np.exp(-2))

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            ExponentialPlateauEnvelope(tau_em=0.0, tau_plat=1.0, tau_dec=1.0)
        with pytest.raises(ValueError):
            ExponentialPlateauEnvelope(tau_em=1.0, tau_plat=-1.0, tau_dec=1.0)


class TestRGamma:
    @pytest.mark.parametrize("te,tp,td", CASES)
    def test_matches_numerical_autocorrelation(self, te, tp, td):
        env = ExponentialPlateauEnvelope(te, tp, td)
        lags = np.linspace(0, tp + 6 * max(te, td), 301)
        R = np.asarray(env.R_Gamma(jnp.asarray(lags)))
        R_num = autocorrelation_numerical(te, tp, td, lags)
        # grid error of the rectangle-rule correlation is ~dt * Gamma ~ 1e-3 absolute
        np.testing.assert_allclose(R, R_num, rtol=0, atol=2e-5 * R[0] + 2e-3)

    @pytest.mark.parametrize("te,tp,td", CASES)
    def test_zero_lag(self, te, tp, td):
        env = ExponentialPlateauEnvelope(te, tp, td)
        assert float(env.R_Gamma(jnp.array(0.0))) == pytest.approx(tp + (te + td) / 4, rel=1e-14)

    def test_zero_lag_is_integral_of_gamma_squared(self):
        te, tp, td = 2.0, 10.0, 5.0
        pieces = [(-np.inf, 0.0), (0.0, tp), (tp, np.inf)]
        integral = sum(quad(lambda s: gamma_paper(s, te, tp, td) ** 2, a, b)[0] for a, b in pieces)
        assert float(_exp_plateau_R_Gamma(0.0, te, tp, td)) == pytest.approx(integral, rel=1e-10)

    def test_symmetric_in_rise_and_decay(self):
        lags = jnp.linspace(0, 40, 101)
        R1 = np.asarray(_exp_plateau_R_Gamma(lags, 2.0, 10.0, 5.0))
        R2 = np.asarray(_exp_plateau_R_Gamma(lags, 5.0, 10.0, 2.0))
        np.testing.assert_allclose(R1, R2, rtol=1e-13)

    @pytest.mark.parametrize("te,td", [(2.0, 5.0), (3.0, 3.0), (6.0, 1.5)])
    def test_no_plateau_is_asymmetric_exponential(self, te, td):
        # ExponentialAsymmetricEnvelope takes e-folding times of Gamma, half of these
        lags = jnp.linspace(0, 30, 121)
        R = np.asarray(ExponentialPlateauEnvelope(te, 0.0, td).R_Gamma(lags))
        R_ref = np.asarray(ExponentialAsymmetricEnvelope(te / 2, td / 2).R_Gamma(lags))
        np.testing.assert_allclose(R, R_ref, rtol=1e-10)

    def test_continuous_through_equal_timescales(self):
        lags = jnp.linspace(0, 40, 81)
        R_eq = np.asarray(_exp_plateau_R_Gamma(lags, 3.0, 10.0, 3.0))
        for eps in (1e-9, 1e-6, 1e-3):
            R_near = np.asarray(_exp_plateau_R_Gamma(lags, 3.0, 10.0, 3.0 + eps))
            np.testing.assert_allclose(R_near, R_eq, rtol=0, atol=10 * eps * R_eq[0])

    def test_continuous_at_end_of_plateau(self):
        te, tp, td = 2.0, 10.0, 5.0
        below = float(_exp_plateau_R_Gamma(tp * (1 - 1e-12), te, tp, td))
        above = float(_exp_plateau_R_Gamma(tp * (1 + 1e-12), te, tp, td))
        assert below == pytest.approx(above, rel=1e-10)

    def test_finite_at_long_lags(self):
        R = np.asarray(_exp_plateau_R_Gamma(jnp.array([1e3, 1e5]), 0.2, 5.0, 30.0))
        assert np.all(np.isfinite(R)) and np.all(R >= 0)


class TestGammaHat:
    @pytest.mark.parametrize("te,tp,td", CASES)
    def test_parts_match_paper(self, te, tp, td):
        w = np.linspace(0.01, 5.0, 200)
        re, im = gamma_hat_paper(w, te, tp, td)
        np.testing.assert_allclose(np.asarray(_exp_plateau_Gamma_hat_re(w, te, tp, td)), re, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.asarray(_exp_plateau_Gamma_hat_im(w, te, tp, td)), im, rtol=1e-12, atol=1e-12)

    def test_zero_frequency(self):
        te, tp, td = 2.0, 10.0, 5.0
        assert float(_exp_plateau_Gamma_hat_re(0.0, te, tp, td)) == pytest.approx(te / 2 + tp + td / 2)
        assert float(_exp_plateau_Gamma_hat_im(0.0, te, tp, td)) == 0.0

    @pytest.mark.parametrize("w", [0.05, 0.7, 3.0])
    def test_parts_match_quadrature(self, w):
        te, tp, td = 2.0, 10.0, 5.0
        pieces = [(-np.inf, 0.0), (0.0, tp), (tp, np.inf)]
        re = sum(quad(lambda s: gamma_paper(s, te, tp, td) * np.cos(w * s), a, b, limit=400)[0] for a, b in pieces)
        im = sum(quad(lambda s: gamma_paper(s, te, tp, td) * np.sin(w * s), a, b, limit=400)[0] for a, b in pieces)
        assert float(_exp_plateau_Gamma_hat_re(w, te, tp, td)) == pytest.approx(re, rel=1e-8, abs=1e-10)
        assert float(_exp_plateau_Gamma_hat_im(w, te, tp, td)) == pytest.approx(im, rel=1e-8, abs=1e-10)

    @pytest.mark.parametrize("te,tp,td", CASES)
    def test_power_matches_fft(self, te, tp, td):
        dt = 2e-3
        t = np.arange(-15 * te, tp + 15 * td, dt)
        g = gamma_paper(t, te, tp, td)
        w = np.linspace(0.0, 3.0, 61)
        ft = np.array([np.sum(g * np.exp(-1j * wi * t)) * dt for wi in w])
        env = ExponentialPlateauEnvelope(te, tp, td)
        np.testing.assert_allclose(np.asarray(env.Gamma_hat_sq(jnp.asarray(w))), np.abs(ft) ** 2,
                                   rtol=0, atol=2e-3 * (tp + (te + td) / 2) ** 2)
        np.testing.assert_allclose(np.asarray(env.Gamma_hat(jnp.asarray(w))) ** 2,
                                   np.asarray(env.Gamma_hat_sq(jnp.asarray(w))), rtol=1e-12)


class TestJax:
    def test_r_gamma_jax_matches_method(self):
        env = ExponentialPlateauEnvelope(2.0, 10.0, 5.0)
        lags = jnp.linspace(0, 40, 101)
        theta_env = jnp.array(list(env.param_dict.values()))
        np.testing.assert_allclose(np.asarray(env.r_gamma_jax(theta_env, lags)),
                                   np.asarray(env.R_Gamma(lags)), rtol=1e-14)

    @pytest.mark.parametrize("theta", [(2.0, 10.0, 5.0), (3.0, 10.0, 3.0), (2.0, 0.0, 5.0)])
    def test_gradients_finite_and_match_finite_differences(self, theta):
        env = ExponentialPlateauEnvelope(*theta)
        lags = jnp.array([0.0, 1.0, theta[1], theta[1] + 2.0, 30.0])

        def loss(th):
            return jnp.sum(env.r_gamma_jax(th, lags))

        th = jnp.array(theta)
        g = np.asarray(jax.grad(loss)(th))
        assert np.all(np.isfinite(g))
        h = 1e-6
        for k in range(3):
            e = jnp.zeros(3).at[k].set(h)
            fd = (float(loss(th + e)) - float(loss(th - e))) / (2 * h)
            assert g[k] == pytest.approx(fd, rel=1e-5, abs=1e-7)

    def test_gamma_hat_gradients_finite_at_zero_frequency(self):
        g = jax.grad(lambda th: _exp_plateau_Gamma_hat_re(0.0, *th) + _exp_plateau_Gamma_hat_im(0.0, *th))
        assert np.all(np.isfinite(np.asarray(g(jnp.array([2.0, 10.0, 5.0])))))


class TestIntegration:
    def test_param_dict_and_support(self):
        env = ExponentialPlateauEnvelope(2.0, 10.0, 5.0)
        assert env.param_dict == {"tau_em": 2.0, "tau_plat": 10.0, "tau_dec": 5.0}
        assert env.lspot == 10.0
        support = env.kernel_support()
        assert float(env.R_Gamma(jnp.array(support))) < 1e-3 * float(env.R_Gamma(jnp.array(0.0)))

    def test_analytic_kernel_peaks_follow_r_gamma(self):
        # for kappa = 0 the kernel at integer periods is R_Gamma(m P) / R_Gamma(0) of K(0)
        from spotgp.analytic_kernel import AnalyticKernel
        from spotgp.spot_model import SpotEvolutionModel, VisibilityFunction
        env = ExponentialPlateauEnvelope(2.0, 10.0, 5.0)
        vis = VisibilityFunction(peq=5.0, kappa=0.0, inc=np.pi / 3)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)
        ak = AnalyticKernel(model, n_harmonics=2)
        lags = jnp.array([0.0, 5.0, 10.0, 15.0])
        K = np.asarray(ak.kernel(lags))
        R = np.asarray(env.R_Gamma(lags))
        np.testing.assert_allclose(K / K[0], R / R[0], rtol=1e-6)
        assert ak.envelope_type == "exponential_plateau"

    def test_gp_solver_fits_all_three_parameters(self, synthetic_data):
        from spotgp.gp_solver import GPSolver
        from spotgp.spot_model import SpotEvolutionModel, VisibilityFunction
        x, y, yerr = synthetic_data
        env = ExponentialPlateauEnvelope(1.0, 5.0, 2.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)
        gp = GPSolver(x, y, yerr, model, matrix_solver="cholesky_full")
        assert {"tau_em", "tau_plat", "tau_dec"} <= set(gp.param_keys)
        assert np.isfinite(float(gp.log_likelihood()))
        gp.build_jax()
        grad = np.asarray(jax.grad(gp.log_likelihood_fn)(jnp.asarray(gp.theta0)))
        assert np.all(np.isfinite(grad))

    def test_save_load_round_trip(self, synthetic_data, tmp_path):
        from spotgp import load_gp, save_gp
        from spotgp.gp_solver import GPSolver
        from spotgp.spot_model import SpotEvolutionModel, VisibilityFunction
        x, y, yerr = synthetic_data
        env = ExponentialPlateauEnvelope(1.0, 5.0, 2.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
        gp = GPSolver(x, y, yerr, SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01),
                      matrix_solver="cholesky_full")
        path = str(tmp_path / "exp_plateau.h5")
        save_gp(path, gp)
        gp2 = load_gp(path)
        assert isinstance(gp2.spot_model.envelope, ExponentialPlateauEnvelope)
        assert gp2.spot_model.envelope.param_dict == env.param_dict
        assert float(gp2.log_likelihood()) == pytest.approx(float(gp.log_likelihood()), rel=1e-12)
