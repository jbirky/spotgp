"""Tests for GPSolver advanced methods: build_jax, log_posterior, gradients, mass matrices."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from spotgp.gp_solver import GPSolver
from spotgp.envelope import TrapezoidSymmetricEnvelope
from spotgp.spot_model import SpotEvolutionModel, VisibilityFunction


@pytest.fixture
def gp_full(default_hparam, synthetic_data):
    """GPSolver with full Cholesky solver."""
    x, y, yerr = synthetic_data
    return GPSolver(x, y, yerr, default_hparam, matrix_solver="cholesky_full")


@pytest.fixture
def gp_banded(default_hparam, synthetic_data):
    """GPSolver with banded Cholesky solver."""
    x, y, yerr = synthetic_data
    return GPSolver(x, y, yerr, default_hparam, matrix_solver="cholesky_banded")


class TestBuildJax:
    def test_returns_self(self, gp_full):
        result = gp_full.build_jax()
        assert result is gp_full

    def test_jit_functions_exist(self, gp_full):
        gp_full.build_jax()
        assert callable(gp_full.log_posterior)
        assert callable(gp_full.neg_log_posterior)
        assert callable(gp_full.grad_log_posterior)
        assert callable(gp_full.grad_neg_log_posterior)


class TestLogPosterior:
    def test_log_posterior_finite(self, gp_full):
        lp = float(gp_full.log_posterior(gp_full.theta0))
        assert np.isfinite(lp)

    def test_neg_log_posterior_is_negative(self, gp_full):
        lp = float(gp_full.log_posterior(gp_full.theta0))
        nlp = float(gp_full.neg_log_posterior(gp_full.theta0))
        np.testing.assert_allclose(nlp, -lp, rtol=1e-10)

    def test_gradient_shape(self, gp_full):
        grad = gp_full.grad_log_posterior(gp_full.theta0)
        assert grad.shape == (gp_full.n_params,)

    def test_gradient_finite(self, gp_full):
        grad = np.array(gp_full.grad_log_posterior(gp_full.theta0))
        assert np.all(np.isfinite(grad))

    def test_neg_gradient_is_negative(self, gp_full):
        grad = np.array(gp_full.grad_log_posterior(gp_full.theta0))
        ngrad = np.array(gp_full.grad_neg_log_posterior(gp_full.theta0))
        np.testing.assert_allclose(ngrad, -grad, rtol=1e-10)

    def test_banded_log_posterior_finite(self, gp_banded):
        lp = float(gp_banded.log_posterior(gp_banded.theta0))
        assert np.isfinite(lp)

    def test_banded_gradient_finite(self, gp_banded):
        grad = np.array(gp_banded.grad_log_posterior(gp_banded.theta0))
        assert np.all(np.isfinite(grad))


class TestMassMatrixHessian:
    def test_shape(self, gp_full):
        M = gp_full.mass_matrix_hessian_map(gp_full.theta0)
        assert M.shape == (gp_full.n_params, gp_full.n_params)

    def test_symmetric(self, gp_full):
        M = np.array(gp_full.mass_matrix_hessian_map(gp_full.theta0))
        np.testing.assert_allclose(M, M.T, atol=1e-10)

    def test_positive_diagonal(self, gp_full):
        M = np.array(gp_full.mass_matrix_hessian_map(gp_full.theta0))
        assert np.all(np.diag(M) > 0)

    def test_stored_on_solver(self, gp_full):
        gp_full.mass_matrix_hessian_map(gp_full.theta0)
        assert gp_full.inverse_mass_matrix is not None
        assert gp_full._hessian is not None


class TestMassMatrixFisher:
    def test_full_solver_shape(self, gp_full):
        M = gp_full.mass_matrix_fisher(gp_full.theta0)
        assert M.shape == (gp_full.n_params, gp_full.n_params)

    def test_full_solver_symmetric(self, gp_full):
        M = np.array(gp_full.mass_matrix_fisher(gp_full.theta0))
        np.testing.assert_allclose(M, M.T, atol=1e-10)

    def test_banded_solver_shape(self, gp_banded):
        M = gp_banded.mass_matrix_fisher(gp_banded.theta0)
        assert M.shape == (gp_banded.n_params, gp_banded.n_params)


class TestMassMatrixLaplace:
    def test_shape(self, gp_full):
        M = gp_full.mass_matrix_laplace(gp_full.theta0)
        assert M.shape == (gp_full.n_params, gp_full.n_params)

    def test_symmetric(self, gp_full):
        M = np.array(gp_full.mass_matrix_laplace(gp_full.theta0))
        np.testing.assert_allclose(M, M.T, atol=1e-10)

    def test_stored_on_solver(self, gp_full):
        gp_full.mass_matrix_laplace(gp_full.theta0)
        assert gp_full.inverse_mass_matrix is not None
        assert gp_full._laplace_hessian is not None
        assert gp_full._laplace_mean is not None


class TestLaplaceSamples:
    def test_shape(self, gp_full):
        gp_full.mass_matrix_laplace(gp_full.theta0)
        gp_full.map_estimate = gp_full.theta0
        samples = gp_full.laplace_samples(n_samples=50)
        assert samples.shape == (50, gp_full.n_params)


class TestGetMassMatrix:
    def test_identity(self, gp_full):
        M = gp_full._get_mass_matrix(None, gp_full.theta0)
        np.testing.assert_allclose(np.array(M), np.eye(gp_full.n_params))

    def test_hessian_map(self, gp_full):
        M = gp_full._get_mass_matrix("hessian_map", gp_full.theta0)
        assert M.shape == (gp_full.n_params, gp_full.n_params)

    def test_fisher(self, gp_full):
        M = gp_full._get_mass_matrix("fisher", gp_full.theta0)
        assert M.shape == (gp_full.n_params, gp_full.n_params)

    def test_laplace(self, gp_full):
        M = gp_full._get_mass_matrix("laplace", gp_full.theta0)
        assert M.shape == (gp_full.n_params, gp_full.n_params)

    def test_unknown_raises(self, gp_full):
        with pytest.raises(ValueError, match="Unknown"):
            gp_full._get_mass_matrix("bogus", gp_full.theta0)


class TestPredictFullSolver:
    def test_predict_shape_full(self, gp_full):
        xpred = np.linspace(0, 20, 15)
        mu, var = gp_full.predict(xpred)
        assert mu.shape == (15,)
        assert var.shape == (15,)

    def test_predict_with_cov_full(self, gp_full):
        xpred = np.linspace(0, 20, 10)
        mu, cov = gp_full.predict(xpred, return_cov=True)
        assert cov.shape == (10, 10)
