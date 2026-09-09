"""Tests for composite-kernel decomposition: ``Term.k_of_lag_no_dc``,
``KernelSum.components``, and ``GPSolver.predict_at``.

This machinery replaces what used to be paper-analysis-script-local code
(``plot_trial.py``'s ``_spot_term_no_dc`` / ``compute_component_kernels`` /
``predict_at_theta``), so any spotgp user building composite kernels gets
per-term decomposition and DC-dropping for free.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from spotgp import AnalyticKernel, GPSolver, SpotEvolutionModel
from spotgp.terms import (
    JitterTerm, KernelSum, SharedVisibilitySpotSum, SHOTerm, SpotTerm,
)

GEOM = dict(peq=10.0, kappa=0.2, inc=np.pi / 4)
HPARAM_A = dict(GEOM, lspot=2.0, tau_spot=0.5, sigma_k=0.01)
HPARAM_B = dict(GEOM, lspot=8.0, tau_spot=2.0, sigma_k=0.005)


def _model(hparam):
    return SpotEvolutionModel.from_hparam(dict(hparam))


def _data(N=80, seed=7):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 25.0, N)
    y = (1.0 + 0.01 * np.sin(2 * np.pi * x / 10.0)
         + 0.002 * rng.standard_normal(N))
    return x, y, 0.002 * np.ones(N)


class TestSpotTermNoDC:
    def test_matches_kernel_components_sum(self):
        """Dropping n=0 via k_of_lag_no_dc must equal the sum of the
        n>=1 rows of AnalyticKernel.kernel_components (the independently
        implemented per-harmonic decomposition)."""
        model = _model(HPARAM_A)
        term = SpotTerm(model)
        lag = jnp.linspace(0.0, 20.0, 100)

        no_dc = np.asarray(term.k_of_lag_no_dc(jnp.asarray(term.theta0), lag))

        ak = AnalyticKernel(model)
        K_n = ak.kernel_components(np.asarray(lag))
        expected = sum(K_n[i] for i, n in enumerate(ak.harmonics) if n != 0)
        np.testing.assert_allclose(no_dc, expected, rtol=1e-10, atol=1e-20)

    def test_no_dc_plus_dc_equals_full_kernel(self):
        model = _model(HPARAM_A)
        term = SpotTerm(model)
        theta = jnp.asarray(term.theta0)
        lag = jnp.linspace(0.0, 20.0, 100)

        full = np.asarray(term.k_of_lag(theta, lag))
        no_dc = np.asarray(term.k_of_lag_no_dc(theta, lag))
        ak = AnalyticKernel(model)
        dc_only = ak.kernel_components(np.asarray(lag))[
            list(ak.harmonics).index(0)]

        np.testing.assert_allclose(full, no_dc + dc_only,
                                   rtol=1e-10, atol=1e-20)

    def test_harmonics_without_dc_is_a_noop(self):
        """If the term was built without n=0 in the first place,
        k_of_lag_no_dc must just equal k_of_lag (nothing to drop)."""
        model = _model(HPARAM_A)
        term = SpotTerm(model, n_harmonics=[1, 2])
        theta = jnp.asarray(term.theta0)
        lag = jnp.linspace(0.0, 20.0, 50)

        np.testing.assert_array_equal(
            np.asarray(term.k_of_lag_no_dc(theta, lag)),
            np.asarray(term.k_of_lag(theta, lag)))


class TestSharedVisibilitySpotSumNoDC:
    def test_matches_naive_spot_term_no_dc_sum(self):
        """Same acceptance criterion as k_of_lag itself (Phase 04): the
        factorized no-DC kernel equals the per-population no-DC sum."""
        models = [_model(HPARAM_A), _model(HPARAM_B)]
        shared = SharedVisibilitySpotSum(models)
        theta = jnp.asarray(shared.theta0)
        lag = jnp.linspace(0.0, 20.0, 100)

        no_dc_shared = np.asarray(shared.k_of_lag_no_dc(theta, lag))
        no_dc_naive = sum(
            np.asarray(SpotTerm(m).k_of_lag_no_dc(jnp.asarray(m.theta0), lag))
            for m in models)
        np.testing.assert_allclose(no_dc_shared, no_dc_naive,
                                   rtol=1e-12, atol=1e-20)


class TestNonSpotTermsHaveNothingToDrop:
    @pytest.mark.parametrize("term", [
        SHOTerm(S0=1e-3, Q=0.7071, w0=2.0),
        JitterTerm(sigma_j=1e-3),
    ])
    def test_no_dc_equals_k_of_lag(self, term):
        theta = jnp.asarray(term.theta0)
        lag = jnp.linspace(0.0, 10.0, 32)
        np.testing.assert_array_equal(
            np.asarray(term.k_of_lag_no_dc(theta, lag)),
            np.asarray(term.k_of_lag(theta, lag)))


class TestKernelSumComponents:
    def test_single_term_returns_total_singleton(self):
        term = SpotTerm(_model(HPARAM_A))
        ks = KernelSum(term)
        theta = jnp.asarray(ks.theta0)
        lag = jnp.linspace(0.0, 20.0, 50)

        parts = ks.components(theta, lag)
        assert [label for label, _ in parts] == ["Total"]
        np.testing.assert_array_equal(
            parts[0][1], np.asarray(ks.k_of_lag(theta, lag)))

    def test_multi_term_parts_sum_to_total(self):
        spot = SpotTerm(_model(HPARAM_A))
        sho = SHOTerm(S0=1e-3, Q=0.7071, w0=2.0)
        ks = KernelSum(spot, sho)
        theta = jnp.asarray(ks.theta0)
        lag = jnp.linspace(0.0, 20.0, 50)

        parts = ks.components(theta, lag)
        labels = [label for label, _ in parts]
        assert labels[-1] == "Total"
        assert set(labels[:-1]) == {spot.prefix, sho.prefix}

        summed = sum(K for _, K in parts[:-1])
        np.testing.assert_allclose(summed, parts[-1][1],
                                   rtol=1e-12, atol=1e-20)
        np.testing.assert_allclose(
            parts[-1][1], np.asarray(ks.k_of_lag(theta, lag)),
            rtol=1e-12, atol=1e-20)

    def test_drop_dc_only_affects_spot_term(self):
        spot = SpotTerm(_model(HPARAM_A))
        sho = SHOTerm(S0=1e-3, Q=0.7071, w0=2.0)
        ks = KernelSum(spot, sho)
        theta = jnp.asarray(ks.theta0)
        lag = jnp.linspace(0.0, 20.0, 50)

        i, n_spot = ks._slices[0]
        j, n_sho = ks._slices[1]
        theta_spot = theta[i:i + n_spot]
        theta_sho = theta[j:j + n_sho]

        parts = dict(ks.components(theta, lag, drop_dc=True))
        np.testing.assert_allclose(
            parts[spot.prefix],
            np.asarray(spot.k_of_lag_no_dc(theta_spot, lag)),
            rtol=1e-12, atol=1e-20)
        np.testing.assert_allclose(
            parts[sho.prefix],
            np.asarray(sho.k_of_lag(theta_sho, lag)),
            rtol=1e-12, atol=1e-20)


class TestTermSample:
    def test_reproducible_with_same_seed(self):
        term = SpotTerm(_model(HPARAM_A))
        x = np.linspace(0.0, 30.0, 50)
        y1 = term.sample(term.theta0, x, seed=0)
        y2 = term.sample(term.theta0, x, seed=0)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        term = SpotTerm(_model(HPARAM_A))
        x = np.linspace(0.0, 30.0, 50)
        y1 = term.sample(term.theta0, x, seed=0)
        y2 = term.sample(term.theta0, x, seed=1)
        assert not np.allclose(y1, y2)

    def test_multi_sample_shape_and_distinct_rows(self):
        term = SpotTerm(_model(HPARAM_A))
        x = np.linspace(0.0, 30.0, 50)
        ys = np.asarray(term.sample(term.theta0, x, n_samples=5, seed=0))
        assert ys.shape == (5, 50)
        assert not np.allclose(ys[0], ys[1])
        # Same call, same seed: reproducible as a whole.
        ys2 = np.asarray(term.sample(term.theta0, x, n_samples=5, seed=0))
        np.testing.assert_array_equal(ys, ys2)

    def test_variance_matches_kernel_at_zero_lag(self):
        """Monte Carlo check: Var[y(0)] over many draws ~ K(0)."""
        term = SpotTerm(_model(HPARAM_A))
        theta = jnp.asarray(term.theta0)
        n = 20_000
        ys = term.sample(theta, np.zeros(1), n_samples=n, seed=0)
        empirical_var = np.var(np.asarray(ys)[:, 0])
        k0 = float(term.k_of_lag(theta, jnp.zeros(1))[0])
        np.testing.assert_allclose(empirical_var, k0, rtol=0.05)

    def test_kernel_sum_sample_matches_sum_of_component_variances(self):
        """KernelSum inherits Term.sample (no override needed): summing
        independent per-term draws must reproduce the total kernel's
        variance, the same additive-GP property KernelSum.k_of_lag
        relies on for the mean."""
        spot = SpotTerm(_model(HPARAM_A))
        sho = SHOTerm(S0=1e-3, Q=0.7071, w0=2.0)
        ks = KernelSum(spot, sho)
        theta = jnp.asarray(ks.theta0)
        n = 20_000

        y_total = ks.sample(theta, np.zeros(1), n_samples=n, seed=0)
        empirical_var = np.var(np.asarray(y_total)[:, 0])
        k0_total = float(ks.k_of_lag(theta, jnp.zeros(1))[0])
        np.testing.assert_allclose(empirical_var, k0_total, rtol=0.05)

        i, n_spot = ks._slices[0]
        j, n_sho = ks._slices[1]
        y_spot = spot.sample(theta[i:i + n_spot], np.zeros(1),
                             n_samples=n, seed=1)
        y_sho = sho.sample(theta[j:j + n_sho], np.zeros(1),
                           n_samples=n, seed=2)
        empirical_var_sum = (np.var(np.asarray(y_spot)[:, 0])
                             + np.var(np.asarray(y_sho)[:, 0]))
        np.testing.assert_allclose(empirical_var_sum, k0_total, rtol=0.05)


class TestPredictAt:
    def test_matches_predict_at_current_theta_single_term(self):
        x, y, yerr = _data()
        gp = GPSolver(x, y, yerr, _model(HPARAM_A),
                     matrix_solver="cholesky_full")
        xpred = np.linspace(x[0], x[-1], 40)

        mu_ref, var_ref = gp.predict(xpred)
        mu_at, var_at = gp.predict_at(gp.theta0, xpred)

        np.testing.assert_allclose(mu_at, mu_ref, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(var_at, var_ref, rtol=1e-6, atol=1e-12)

    def test_works_for_composite_kernel_where_update_hparam_cannot(self):
        x, y, yerr = _data()
        ks = KernelSum(SpotTerm(_model(HPARAM_A)),
                       SHOTerm(S0=1e-3, Q=0.7071, w0=2.0))
        gp = GPSolver(x, y, yerr, ks, matrix_solver="cholesky_full")

        with pytest.raises(NotImplementedError):
            gp.update_hparam({"peq": 11.0})

        xpred = np.linspace(x[0], x[-1], 40)
        mu, var = gp.predict_at(gp.theta0, xpred)
        assert np.all(np.isfinite(mu))
        assert np.all(np.isfinite(var))
        assert np.all(np.asarray(var) >= 0)
