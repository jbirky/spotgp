"""Tests for src.gp_solver — GPSolver."""

import warnings

import numpy as np
import pytest
import jax.numpy as jnp

from spotgp.gp_solver import GPSolver
from spotgp.envelope import TrapezoidSymmetricEnvelope
from spotgp.spot_model import SpotEvolutionModel, VisibilityFunction
from spotgp.terms import JitterTerm, KernelSum, SHOTerm, SpotTerm, Term
from spotgp.validation import CholeskyWarning


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

    def test_fit_sigma_n_removed(self, default_hparam, synthetic_data):
        # White noise now enters only through a JitterTerm; the old flag
        # must fail loudly rather than be forwarded to the kernel.
        x, y, yerr = synthetic_data
        with pytest.raises(TypeError, match="JitterTerm"):
            GPSolver(x, y, yerr, default_hparam, fit_sigma_n=True)

    def test_jitter_is_the_white_noise_parameter(self, default_hparam,
                                                 synthetic_data):
        x, y, yerr = synthetic_data
        kernel = KernelSum(SpotTerm(default_hparam, prefix="spot"),
                           JitterTerm(sigma_j=1e-3, prefix="jit"))
        gp = GPSolver(x, y, yerr, kernel)
        assert gp.param_keys[-1] == "jit.sigma_j"
        assert gp.n_params == 7

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


class TestComputeKernelSamples:
    @pytest.fixture
    def composite(self, default_hparam, synthetic_data):
        x, y, yerr = synthetic_data
        kernel = KernelSum(SpotTerm(default_hparam, prefix="spot"),
                           SHOTerm(S0=1e-6, Q=2.0, w0=3.0, prefix="sho"),
                           JitterTerm(sigma_j=1e-3, prefix="jit"))
        gp = GPSolver(x, y, yerr, kernel,
                      bounds={"spot.log_sigma_k": (-6.0, 0.0)})
        rng = np.random.default_rng(1)
        theta0 = np.asarray(gp.theta0)
        samples = theta0 * (1 + 0.02 * rng.standard_normal((7, gp.n_params)))
        return gp, samples

    def test_kernel_matches_per_sample_eval(self, composite):
        # Samples are in sampling space (log_sigma_k), like sampler output.
        gp, samples = composite
        tlags = np.linspace(0, 15, 50)
        grid, K = gp.compute_kernel_samples(samples, tlags=tlags, batch_size=3)
        ref = np.stack([np.asarray(gp.kernel_sum.k_of_lag(
            gp._to_physical(jnp.asarray(s)), jnp.asarray(tlags)))
            for s in samples])
        np.testing.assert_array_equal(grid, tlags)
        np.testing.assert_allclose(K, ref, rtol=1e-12, atol=0)

    def test_drop_dc_matches_components(self, composite):
        gp, samples = composite
        tlags = np.linspace(0, 15, 50)
        _, K = gp.compute_kernel_samples(samples[:1], tlags=tlags,
                                         drop_dc=True)
        _, ref = gp.kernel_sum.components(
            gp._to_physical(jnp.asarray(samples[0])), jnp.asarray(tlags),
            drop_dc=True)[-1]
        np.testing.assert_allclose(K[0], ref, rtol=1e-12, atol=0)

    def test_psd_shape_and_parseval(self, composite):
        gp, samples = composite
        tlags = np.arange(0, 2000.0, 0.01)
        freq, P = gp.compute_kernel_samples(samples[:6].reshape(2, 3, -1),
                                            kind="psd", tlags=tlags)
        assert freq.shape == (len(tlags) - 1,) and freq[0] > 0
        assert P.shape == (2, 3, len(freq))
        # One-sided PSD integrates to K(0), up to the dropped f=0 bin.
        k0 = float(gp.kernel_sum.k_of_lag(
            gp._to_physical(jnp.asarray(samples[0])), jnp.zeros(1))[0])
        df = freq[1] - freq[0]
        np.testing.assert_allclose(np.sum(P[0, 0]) * df, k0, rtol=5e-3)

    def test_psd_matches_sho_analytic_shape(self, synthetic_data):
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, KernelSum(SHOTerm(S0=1e-6, Q=2.0, w0=3.0)))
        freq, P = gp.compute_kernel_samples(
            np.asarray(gp.theta0), kind="psd", tlags=np.arange(0, 400, 0.01))
        _, P_an = gp.kernel_sum.terms[0].psd(2 * np.pi * freq)
        sel = (freq > 0.05) & (freq < 5)
        # One-sided per-frequency PSD = 2 sqrt(2 pi) x celerite S(omega).
        np.testing.assert_allclose(P[sel] / P_an[sel], 2 * np.sqrt(2 * np.pi),
                                   rtol=1e-4)

    def test_bad_inputs(self, composite):
        gp, samples = composite
        with pytest.raises(ValueError, match="kind"):
            gp.compute_kernel_samples(samples, kind="acf")
        with pytest.raises(ValueError, match="uniform grid"):
            gp.compute_kernel_samples(samples, kind="psd",
                                      tlags=np.array([0.0, 1.0, 3.0, 4.0]))
        with pytest.raises(ValueError, match="n_params"):
            gp.compute_kernel_samples(samples[:, :5])


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


class _NotPositiveDefinite(Term):
    """Invalid kernel: k(0) = a^2 and k(tau != 0) = -a^2.

    The covariance has an eigenvalue a^2 (2 - N) < 0, so its Cholesky
    factorization fails deterministically whatever yerr is.
    """

    _prefix_tag = "npd"

    @property
    def base_keys(self):
        return ("a",)

    @property
    def theta0(self):
        return np.array([1.0])

    @property
    def default_bounds(self):
        return {"a": (0.1, 10.0)}

    def k_of_lag(self, theta_slice, lag_flat):
        lag = jnp.asarray(lag_flat)
        return jnp.where(lag == 0.0, 1.0, -1.0) * theta_slice[0] ** 2

    def bandwidth_support(self, param_keys, bounds_arr):
        return 1e6


def _failing_solver(**kw):
    x = np.linspace(0.0, 10.0, 30)
    return GPSolver(x, np.sin(x), 0.01 * np.ones_like(x),
                    KernelSum(_NotPositiveDefinite()), **kw)


class TestCholeskyFailureWarning:
    @pytest.mark.parametrize("solver", ["cholesky_full", "cholesky_banded"])
    def test_construction_warns(self, solver):
        with pytest.warns(CholeskyWarning, match="JitterTerm"):
            _failing_solver(matrix_solver=solver)

    def test_build_jax_warns(self):
        with pytest.warns(CholeskyWarning):
            gp = _failing_solver(matrix_solver="cholesky_full")
        with pytest.warns(CholeskyWarning, match="build_jax"):
            gp.build_jax()
        assert not np.isfinite(float(gp.log_likelihood_fn(gp.theta0)))

    def test_predict_at_warns(self):
        with pytest.warns(CholeskyWarning):
            gp = _failing_solver(matrix_solver="cholesky_full")
        with pytest.warns(CholeskyWarning, match="predict_at"):
            gp.predict_at(gp.theta0, np.linspace(0.0, 10.0, 5))

    def test_dynesty_warns_once(self, tmp_path):
        from spotgp.samplers import DynestySampler

        with pytest.warns(CholeskyWarning):
            gp = _failing_solver(matrix_solver="cholesky_full")
        sampler = DynestySampler(gp, save_dir=str(tmp_path))
        theta = np.asarray(gp.theta0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert not np.isfinite(sampler._log_likelihood(theta))
            sampler._log_likelihood(theta)
        assert sum(issubclass(w.category, CholeskyWarning)
                   for w in caught) == 1

    def test_valid_kernel_does_not_warn(self, default_hparam, synthetic_data):
        x, y, yerr = synthetic_data
        kernel = KernelSum(SpotTerm(default_hparam, prefix="spot"),
                           JitterTerm(sigma_j=1e-3, prefix="jit"))
        with warnings.catch_warnings():
            warnings.simplefilter("error", CholeskyWarning)
            gp = GPSolver(x, y, yerr, kernel,
                          matrix_solver="cholesky_full").build_jax()
            gp.predict_at(gp.theta0, np.linspace(0.0, 10.0, 5))


class TestNumericalTrapezoidEnvelope:
    """TrapezoidSymmetricEnvelope(numerical=True) through the solver."""

    @staticmethod
    def _model(lspot=3.0, tau_spot=8.0):
        # lspot < tau_spot: outside the closed form's validity range.
        env = TrapezoidSymmetricEnvelope(lspot=lspot, tau_spot=tau_spot,
                                         numerical=True)
        vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
        return SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)

    def test_likelihood_and_fisher_finite(self, synthetic_data):
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, self._model(), matrix_solver="cholesky_full")
        assert np.isfinite(float(gp.log_likelihood()))
        gp.build_jax()
        cov = np.array(gp.mass_matrix_fisher(gp.theta0))
        assert np.all(np.isfinite(cov))
        assert np.all(np.diag(cov) > 0)

    def test_save_load_preserves_numerical(self, synthetic_data, tmp_path):
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, self._model())
        path = str(tmp_path / "numerical_envelope.h5")
        gp.save(path)
        gp2 = GPSolver.load(path)
        env2 = gp2.spot_model.envelope
        assert env2.numerical is True
        assert env2.n_grid == 4096
        lags = jnp.linspace(0.0, 20.0, 50)
        np.testing.assert_allclose(np.array(env2.R_Gamma(lags)),
                                   np.array(gp.spot_model.envelope.R_Gamma(lags)))
        assert np.isfinite(float(gp2.log_likelihood()))

    def test_update_model_preserves_numerical(self, synthetic_data):
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, self._model())
        gp._update_model_from_theta({"lspot": 2.0, "tau_spot": 9.0})
        env = gp.spot_model.envelope
        assert env.numerical is True
        assert env.lspot == 2.0 and env.tau_spot == 9.0
