"""Tests for src.gp_solver — GPSolver."""

import numpy as np
import pytest
import jax.numpy as jnp

from src.gp_solver import GPSolver
from src.envelope import TrapezoidSymmetricEnvelope
from src.spot_model import SpotEvolutionModel, VisibilityFunction


class TestGPSolverInit:
    def test_from_hparam(self, default_hparam, synthetic_data):
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, default_hparam)
        assert gp.N == len(x)
        assert gp.n_params == 6

    def test_from_spot_model(self, synthetic_data):
        x, y, yerr = synthetic_data
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)
        gp = GPSolver(x, y, yerr, model)
        assert gp.N == len(x)
        assert gp.n_params == 6

    def test_fit_sigma_n(self, default_hparam, synthetic_data):
        x, y, yerr = synthetic_data
        default_hparam["sigma_n"] = 0.001
        gp = GPSolver(x, y, yerr, default_hparam, fit_sigma_n=True)
        assert gp.n_params == 7
        assert "sigma_n" in gp.param_keys

    def test_param_keys(self, default_hparam, synthetic_data):
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, default_hparam)
        assert gp.param_keys == ("peq", "kappa", "inc", "lspot", "tau_spot", "sigma_k")


class TestGPSolverLikelihood:
    def test_log_likelihood_finite(self, default_hparam, synthetic_data):
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, default_hparam)
        ll = gp.log_likelihood()
        assert np.isfinite(float(ll))

    def test_log_likelihood_changes_with_params(self, default_hparam, synthetic_data):
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, default_hparam)
        ll1 = float(gp.log_likelihood())
        new_hp = dict(default_hparam)
        new_hp["sigma_k"] = 0.02
        gp.update_hparam(new_hp)
        ll2 = float(gp.log_likelihood())
        assert ll1 != ll2

    def test_full_vs_banded_solver(self, default_hparam, synthetic_data):
        """Full and banded Cholesky should give similar log-likelihoods."""
        x, y, yerr = synthetic_data
        gp_banded = GPSolver(x, y, yerr, default_hparam, matrix_solver="cholesky_banded")
        gp_full = GPSolver(x, y, yerr, default_hparam, matrix_solver="cholesky_full")
        ll_banded = float(gp_banded.log_likelihood())
        ll_full = float(gp_full.log_likelihood())
        np.testing.assert_allclose(ll_banded, ll_full, rtol=1e-4)


class TestGPSolverPrediction:
    def test_predict_shape(self, default_hparam, synthetic_data):
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, default_hparam)
        xpred = np.linspace(0, 20, 15)
        mu, var = gp.predict(xpred)
        assert mu.shape == (15,)
        assert var.shape == (15,)

    def test_predict_with_cov(self, default_hparam, synthetic_data):
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, default_hparam)
        xpred = np.linspace(0, 20, 10)
        mu, cov = gp.predict(xpred, return_cov=True)
        assert mu.shape == (10,)
        assert cov.shape == (10, 10)

    def test_predict_variance_non_negative(self, default_hparam, synthetic_data):
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, default_hparam)
        _, var = gp.predict(np.linspace(0, 20, 20))
        assert np.all(np.array(var) >= -1e-10)

    def test_sample_prior(self, default_hparam, synthetic_data):
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, default_hparam)
        samples = gp.sample_prior(
            np.linspace(0, 20, 10), n_samples=3,
            rng=np.random.default_rng(0),
        )
        assert samples.shape == (3, 10)


class TestGPSolverUpdate:
    def test_update_hparam(self, default_hparam, synthetic_data):
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, default_hparam)
        ll1 = float(gp.log_likelihood())
        new_hp = dict(default_hparam)
        new_hp["sigma_k"] = 0.05
        gp.update_hparam(new_hp)
        ll2 = float(gp.log_likelihood())
        assert ll1 != ll2
        assert gp.hparam["sigma_k"] == 0.05
