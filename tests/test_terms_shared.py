"""Tests for SharedVisibilitySpotSum (Phase 04): the shared-geometry
composite fast path, K(tau) = V(tau) * sum_i sigma_k_i^2 R_Gamma_i(tau)."""

import jax.numpy as jnp
import numpy as np
import pytest

from spotgp import GPSolver, SpotEvolutionModel
from spotgp.terms import KernelSum, SharedVisibilitySpotSum, SpotTerm

GEOM = dict(peq=10.0, kappa=0.2, inc=np.pi / 4)
HPARAM_A = dict(GEOM, lspot=2.0, tau_spot=0.5, sigma_k=0.01)
HPARAM_B = dict(GEOM, lspot=8.0, tau_spot=2.0, sigma_k=0.005)
HPARAM_C = dict(GEOM, lspot=4.0, tau_spot=1.0, sigma_k=0.02)


def _models(*hparams):
    return [SpotEvolutionModel.from_hparam(dict(h)) for h in hparams]


def _data(N=80):
    rng = np.random.default_rng(3)
    x = np.linspace(0.0, 25.0, N)
    y = (1.0 + 0.01 * np.sin(2 * np.pi * x / 10.0)
         + 0.002 * rng.standard_normal(N))
    return x, y, 0.002 * np.ones(N)


class TestSharedVisibilitySpotSum:
    def test_param_layout(self):
        term = SharedVisibilitySpotSum(_models(HPARAM_A, HPARAM_B))
        assert term.base_keys == (
            "peq", "kappa", "inc",
            "pop0.lspot", "pop0.tau_spot", "pop0.sigma_k",
            "pop1.lspot", "pop1.tau_spot", "pop1.sigma_k")
        np.testing.assert_allclose(
            term.theta0,
            [10.0, 0.2, np.pi / 4, 2.0, 0.5, 0.01, 8.0, 2.0, 0.005])

    @pytest.mark.parametrize("n_comp", [2, 3])
    def test_matches_naive_spot_term_sum(self, n_comp):
        """The factorized kernel equals the per-term sum to machine
        precision when geometry values coincide (Phase 04 acceptance)."""
        hparams = [HPARAM_A, HPARAM_B, HPARAM_C][:n_comp]
        shared = SharedVisibilitySpotSum(_models(*hparams))
        lag = jnp.linspace(0.0, 20.0, 200)

        k_shared = np.asarray(
            shared.k_of_lag(jnp.asarray(shared.theta0), lag))
        k_naive = sum(
            np.asarray(SpotTerm(m).k_of_lag(jnp.asarray(m.theta0), lag))
            for m in _models(*hparams))
        np.testing.assert_allclose(k_shared, k_naive,
                                   rtol=1e-13, atol=1e-20)

    def test_geometry_mismatch_raises(self):
        bad = dict(HPARAM_B, peq=7.0)
        with pytest.raises(ValueError, match="geometry"):
            SharedVisibilitySpotSum(_models(HPARAM_A, bad))

    def test_single_component_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            SharedVisibilitySpotSum(_models(HPARAM_A))

    def test_duplicate_labels_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            SharedVisibilitySpotSum(_models(HPARAM_A, HPARAM_B),
                                    labels=["p", "p"])

    def test_custom_labels(self):
        term = SharedVisibilitySpotSum(
            _models(HPARAM_A, HPARAM_B), labels=["short", "long"])
        assert "short.lspot" in term.base_keys
        assert "long.sigma_k" in term.base_keys


class TestSharedVisibilityInSolver:
    def test_finite_logL_and_gradient(self):
        x, y, yerr = _data()
        term = SharedVisibilitySpotSum(_models(HPARAM_A, HPARAM_B))
        gp = GPSolver(x, y, yerr, term, matrix_solver="cholesky_full")
        assert gp.n_params == 9
        assert np.isfinite(float(gp.log_likelihood_fn(gp.theta0)))
        _, grad = gp.value_and_grad_log_posterior(gp.theta0)
        assert np.all(np.isfinite(np.asarray(grad)))

    def test_solver_logL_matches_naive_sum_solver(self):
        """Same data, same physical parameters: the shared-geometry
        solver and the naive two-SpotTerm solver agree closely (they
        differ only in summation order inside the kernel)."""
        x, y, yerr = _data()
        gp_shared = GPSolver(
            x, y, yerr, SharedVisibilitySpotSum(_models(HPARAM_A, HPARAM_B)),
            matrix_solver="cholesky_full")
        gp_naive = GPSolver(
            x, y, yerr, KernelSum(SpotTerm(dict(HPARAM_A)),
                                  SpotTerm(dict(HPARAM_B))),
            matrix_solver="cholesky_full")
        np.testing.assert_allclose(
            float(gp_shared.log_likelihood_fn(gp_shared.theta0)),
            float(gp_naive.log_likelihood_fn(gp_naive.theta0)),
            rtol=1e-10)

    def test_bandwidth_is_max_over_components(self):
        x, y, yerr = _data()
        term = SharedVisibilitySpotSum(_models(HPARAM_A, HPARAM_B))
        bounds = {
            "pop0.lspot": (0.1, 1.0), "pop0.tau_spot": (0.05, 0.2),
            "pop1.lspot": (0.1, 6.0), "pop1.tau_spot": (0.05, 1.5),
        }
        gp = GPSolver(x, y, yerr, term,
                      matrix_solver="cholesky_banded", bounds=bounds)
        # support = lspot_hi + 2*tau_hi of the WIDER component (pop1)
        dt = x[1] - x[0]
        want = min(int(np.ceil((6.0 + 2 * 1.5) / dt)), len(x) - 1)
        assert gp.bandwidth == want

    def test_log_space_amplitudes(self):
        x, y, yerr = _data()
        term = SharedVisibilitySpotSum(_models(HPARAM_A, HPARAM_B))
        bounds = {"pop0.log_sigma_k": (-6.0, 0.0),
                  "pop1.log_sigma_k": (-6.0, 0.0)}
        gp = GPSolver(x, y, yerr, term,
                      matrix_solver="cholesky_full", bounds=bounds)
        i0 = gp.param_keys.index("pop0.log_sigma_k")
        np.testing.assert_allclose(float(gp.theta0[i0]), -2.0)
        assert np.isfinite(float(gp.log_likelihood_fn(gp.theta0)))

    def test_save_load_roundtrip(self, tmp_path):
        from spotgp import load_gp, save_gp

        x, y, yerr = _data(N=60)
        term = SharedVisibilitySpotSum(
            _models(HPARAM_A, HPARAM_B), labels=["short", "long"])
        gp = GPSolver(x, y, yerr, term, matrix_solver="cholesky_full")
        path = str(tmp_path / "shared.h5")
        save_gp(path, gp)
        gp2 = load_gp(path)
        assert gp2.param_keys == gp.param_keys
        np.testing.assert_allclose(np.asarray(gp2.theta0),
                                   np.asarray(gp.theta0))
        t2 = gp2.kernel_sum.terms[0]
        assert isinstance(t2, SharedVisibilitySpotSum)
        assert t2.labels == ["short", "long"]
        np.testing.assert_allclose(
            float(gp2.log_likelihood_fn(gp2.theta0)),
            float(gp.log_likelihood_fn(gp.theta0)), rtol=1e-12)
