"""Tests for the non-spot analytic terms (Phase 03): SHO, Matern-3/2,
jitter — and their composition with spot terms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spotgp import GPSolver
from spotgp.terms import (
    JitterTerm, KernelSum, Matern32Term, SHOTerm, SpotTerm,
)

HPARAM = dict(peq=10.0, kappa=0.2, inc=np.pi / 4, lspot=2.0,
              tau_spot=0.5, sigma_k=0.01)


def _data(N=80):
    rng = np.random.default_rng(11)
    x = np.linspace(0.0, 25.0, N)
    y = (1.0 + 0.01 * np.sin(2 * np.pi * x / 10.0)
         + 0.002 * rng.standard_normal(N))
    return x, y, 0.002 * np.ones(N)


class TestSHOTerm:
    @pytest.mark.parametrize("Q", [0.1, 0.4999, 0.5, 0.5001, 0.7071, 5.0])
    def test_variance_is_S0_w0_Q(self, Q):
        term = SHOTerm(S0=1e-3, Q=Q, w0=2.0)
        k0 = float(term.k_of_lag(jnp.asarray(term.theta0),
                                 jnp.zeros(1))[0])
        np.testing.assert_allclose(k0, 1e-3 * 2.0 * Q, rtol=1e-3)

    def test_continuous_across_critical_damping(self):
        # A factor-2 seam at Q = 1/2 (the plan's uncorrected critical
        # form) would show up as |dk| ~ k(0)/2; genuine parameter
        # sensitivity at dQ = 1e-3 stays below ~0.5% of k(0).
        tau = jnp.linspace(0.0, 5.0, 64)
        vals = {}
        for Q in (0.499, 0.5, 0.501):
            term = SHOTerm(S0=1e-3, Q=Q, w0=2.0)
            vals[Q] = np.asarray(
                term.k_of_lag(jnp.asarray(term.theta0), tau))
        k0 = 1e-3 * 2.0 * 0.5
        np.testing.assert_allclose(vals[0.499], vals[0.5],
                                   rtol=0, atol=5e-3 * k0)
        np.testing.assert_allclose(vals[0.501], vals[0.5],
                                   rtol=0, atol=5e-3 * k0)

    @pytest.mark.parametrize("Q", [0.05, 0.5, 2.0])
    def test_gradients_finite_all_regimes(self, Q):
        tau = jnp.linspace(0.0, 10.0, 32)
        term = SHOTerm()

        def total(theta):
            return jnp.sum(term.k_of_lag(theta, tau))

        g = jax.grad(total)(jnp.array([1e-3, Q, 2.0]))
        assert np.all(np.isfinite(np.asarray(g)))

    def test_overdamped_no_overflow_at_long_lags(self):
        # cosh form would overflow at w0*t ~ 1e3; the two-exponential
        # form must stay finite (and tiny).
        term = SHOTerm(S0=1.0, Q=0.05, w0=50.0)
        k = np.asarray(term.k_of_lag(jnp.asarray(term.theta0),
                                     jnp.array([0.0, 30.0, 100.0])))
        assert np.all(np.isfinite(k))
        assert k[1] < k[0]

    def test_psd_peak_and_dc(self):
        S0, Q, w0 = 2e-3, 4.0, 3.0
        term = SHOTerm(S0=S0, Q=Q, w0=w0)
        freq, power = term.psd(jnp.array([0.0, w0]))
        np.testing.assert_allclose(power[0], np.sqrt(2 / np.pi) * S0,
                                   rtol=1e-12)
        np.testing.assert_allclose(power[1],
                                   np.sqrt(2 / np.pi) * S0 * Q ** 2,
                                   rtol=1e-12)

    def test_bandwidth_scales_with_Q(self):
        term = SHOTerm()
        keys = ("S0", "Q", "w0")
        lo = SHOTerm.DEFAULT_BOUNDS
        b_low = term.bandwidth_support(
            keys, np.array([lo["S0"], (0.1, 1.0), (1.0, 10.0)]))
        b_high = term.bandwidth_support(
            keys, np.array([lo["S0"], (0.1, 50.0), (1.0, 10.0)]))
        assert b_high > b_low  # high-Q ringing widens the band


class TestMatern32Term:
    def test_kernel_values(self):
        term = Matern32Term(sigma=0.5, rho=2.0)
        tau = jnp.array([0.0, 1.0, 10.0])
        k = np.asarray(term.k_of_lag(jnp.asarray(term.theta0), tau))
        np.testing.assert_allclose(k[0], 0.25, rtol=1e-12)
        arg = np.sqrt(3.0) / 2.0
        np.testing.assert_allclose(k[1], 0.25 * (1 + arg) * np.exp(-arg),
                                   rtol=1e-12)
        assert k[2] < k[1] < k[0]

    def test_psd_integrates_to_variance(self):
        # Wiener-Khinchin sanity: (1/2pi) * int S(w) dw == k(0).
        term = Matern32Term(sigma=0.3, rho=1.5)
        omega = jnp.linspace(-400.0, 400.0, 400001)
        _, power = term.psd(omega)
        var = np.trapezoid(power, np.asarray(omega)) / (2 * np.pi)
        np.testing.assert_allclose(var, 0.09, rtol=1e-3)


class TestJitterTerm:
    def test_delta_at_zero_lag(self):
        term = JitterTerm(sigma_j=0.02)
        k = np.asarray(term.k_of_lag(jnp.asarray(term.theta0),
                                     jnp.array([0.0, 1e-9, 1.0])))
        np.testing.assert_allclose(k[0], 4e-4, rtol=1e-12)
        assert k[1] == 0.0 and k[2] == 0.0

    def test_zero_bandwidth(self):
        assert JitterTerm().bandwidth_support((), np.zeros((0, 2))) == 0.0


class TestSpotPlusSHOComposition:
    def test_finite_logL_and_gradient(self):
        x, y, yerr = _data()
        ks = KernelSum(SpotTerm(dict(HPARAM)),
                       SHOTerm(S0=1e-5, Q=2.0, w0=2 * np.pi / 3.0,
                               prefix="gran"))
        gp = GPSolver(x, y, yerr, ks, matrix_solver="cholesky_full")
        assert gp.param_keys[-3:] == ("gran.S0", "gran.Q", "gran.w0")
        assert np.isfinite(float(gp.log_likelihood_fn(gp.theta0)))
        val, grad = gp.value_and_grad_log_posterior(gp.theta0)
        assert np.all(np.isfinite(np.asarray(grad)))

    def test_summed_psd_is_sum_of_component_psds(self):
        ks = KernelSum(SHOTerm(S0=1e-4, Q=3.0, w0=2.0, prefix="a"),
                       Matern32Term(sigma=0.02, rho=2.0, prefix="b"))
        omega = jnp.linspace(0.01, 20.0, 128)
        freq, total = ks.psd(omega)
        _, pa = ks.terms[0].psd(omega)
        _, pb = ks.terms[1].psd(omega)
        np.testing.assert_allclose(total, pa + pb, rtol=1e-12)

    def test_high_Q_sho_widens_bandwidth(self):
        x, y, yerr = _data()
        bounds_narrow = {"gran.Q": (0.5, 1.0), "gran.w0": (2.0, 10.0),
                         "gran.S0": (1e-8, 1e-2)}
        bounds_wide = {"gran.Q": (0.5, 200.0), "gran.w0": (2.0, 10.0),
                       "gran.S0": (1e-8, 1e-2)}

        def bw(bounds):
            ks = KernelSum(SpotTerm(dict(HPARAM)),
                           SHOTerm(prefix="gran", w0=4.0))
            gp = GPSolver(x, y, yerr, ks,
                          matrix_solver="cholesky_banded", bounds=bounds)
            return gp.bandwidth

        # High-Q ringing must force a wider band (here clamped at N-1,
        # i.e. the full-matrix fallback).
        assert bw(bounds_wide) >= bw(bounds_narrow)
        assert bw(bounds_wide) == len(x) - 1

    def test_spot_plus_jitter_matches_sigma_n_diagonal(self):
        # JitterTerm and the sigma_n noise diagonal must produce the
        # same likelihood at equal amplitude (both add sigma^2 to the
        # diagonal).  sigma_n is evaluated at an explicit theta because
        # its hparam value is not carried into theta0 (it initializes
        # at the lower bound).
        x, y, yerr = _data(N=60)
        sj = 0.003
        ks = KernelSum(SpotTerm(dict(HPARAM)), JitterTerm(sigma_j=sj))
        gp_j = GPSolver(x, y, yerr, ks, matrix_solver="cholesky_full")
        logL_j = float(gp_j.log_likelihood_fn(gp_j.theta0))

        gp_n = GPSolver(x, y, yerr, dict(HPARAM),
                        matrix_solver="cholesky_full", fit_sigma_n=True)
        theta_n = jnp.asarray(
            np.append(np.asarray(gp_j.theta0)[:6], sj))
        logL_n = float(gp_n.log_likelihood_fn(theta_n))
        np.testing.assert_allclose(logL_j, logL_n, rtol=1e-12)

    def test_save_load_roundtrip_with_analytic_terms(self, tmp_path):
        from spotgp import load_gp, save_gp

        x, y, yerr = _data(N=60)
        ks = KernelSum(SpotTerm(dict(HPARAM)),
                       SHOTerm(S0=2e-5, Q=1.5, w0=4.0, prefix="gran"),
                       Matern32Term(sigma=0.01, rho=3.0, prefix="m"))
        gp = GPSolver(x, y, yerr, ks, matrix_solver="cholesky_full")
        path = str(tmp_path / "mixed.h5")
        save_gp(path, gp)
        gp2 = load_gp(path)
        assert gp2.param_keys == gp.param_keys
        np.testing.assert_allclose(np.asarray(gp2.theta0),
                                   np.asarray(gp.theta0))
        np.testing.assert_allclose(
            float(gp2.log_likelihood_fn(gp2.theta0)),
            float(gp.log_likelihood_fn(gp.theta0)), rtol=1e-12)
