"""Tests for the Toeplitz fast path and shared value_and_grad compilation."""

import numpy as np
import jax.numpy as jnp
import pytest

from spotgp.gp_solver import (
    GPSolver,
    _detect_uniform_dt,
    _gp_log_likelihood,
    _gp_log_likelihood_banded,
    _build_banded_kernel_jax,
)

LAT_RANGE = (-np.pi / 2, np.pi / 2)


@pytest.fixture
def uniform_data():
    """Uniformly sampled synthetic dataset (dt = 0.5 exactly)."""
    rng = np.random.default_rng(0)
    N = 40
    x = np.arange(N) * 0.5
    y = 1.0 + 0.005 * rng.standard_normal(N)
    yerr = np.full(N, 0.001)
    return x, y, yerr


def _theta_arr(hparam):
    return jnp.array([hparam[k] for k in
                      ("peq", "kappa", "inc", "lspot", "tau_spot", "sigma_k")],
                     dtype=jnp.float64)


class TestDetectUniformDt:
    def test_uniform_grid(self):
        x = np.arange(50) * 0.25
        assert _detect_uniform_dt(x) == pytest.approx(0.25)

    def test_linspace_grid(self):
        x = np.linspace(0, 20, 30)
        assert _detect_uniform_dt(x) == pytest.approx(20.0 / 29.0)

    def test_gapped_grid(self):
        x = np.concatenate([np.arange(20) * 0.5,
                            30.0 + np.arange(20) * 0.5])
        assert _detect_uniform_dt(x) is None

    def test_irregular_grid(self):
        rng = np.random.default_rng(1)
        x = np.sort(rng.uniform(0, 20, 40))
        assert _detect_uniform_dt(x) is None

    def test_too_short(self):
        assert _detect_uniform_dt(np.array([1.0])) is None


class TestToeplitzMatchesGeneralPath:
    """The fast path must reproduce the general path to fp precision."""

    def test_banded_kernel_build(self, default_hparam, uniform_data):
        x = jnp.asarray(uniform_data[0])
        theta = _theta_arr(default_hparam)
        b = 12
        cb_ref = _build_banded_kernel_jax(theta, x, b, 3, 16, LAT_RANGE,
                                          uniform_dt=None)
        cb_fast = _build_banded_kernel_jax(theta, x, b, 3, 16, LAT_RANGE,
                                           uniform_dt=0.5)
        np.testing.assert_allclose(np.asarray(cb_fast), np.asarray(cb_ref),
                                   rtol=1e-9, atol=1e-20)

    def test_full_log_likelihood(self, default_hparam, uniform_data):
        x, y, yerr = (jnp.asarray(a) for a in uniform_data)
        theta = _theta_arr(default_hparam)
        mean = float(jnp.mean(y))
        ll_ref = _gp_log_likelihood(theta, x, y, yerr, mean,
                                    3, 16, LAT_RANGE, False,
                                    uniform_dt=None)
        ll_fast = _gp_log_likelihood(theta, x, y, yerr, mean,
                                     3, 16, LAT_RANGE, False,
                                     uniform_dt=0.5)
        assert np.isclose(float(ll_fast), float(ll_ref), rtol=1e-8)

    def test_banded_log_likelihood(self, default_hparam, uniform_data):
        x, y, yerr = (jnp.asarray(a) for a in uniform_data)
        theta = _theta_arr(default_hparam)
        mean = float(jnp.mean(y))
        b = 12
        ll_ref = _gp_log_likelihood_banded(theta, x, y, yerr, mean,
                                           3, 16, LAT_RANGE, False, b,
                                           uniform_dt=None)
        ll_fast = _gp_log_likelihood_banded(theta, x, y, yerr, mean,
                                            3, 16, LAT_RANGE, False, b,
                                            uniform_dt=0.5)
        assert np.isclose(float(ll_fast), float(ll_ref), rtol=1e-8)


class TestGPSolverToeplitz:
    def test_uniform_grid_detected(self, default_hparam, uniform_data):
        x, y, yerr = uniform_data
        gp = GPSolver(x, y, yerr, default_hparam, n_lat=16)
        assert gp.uniform_dt == pytest.approx(0.5)

    def test_irregular_grid_not_detected(self, default_hparam, uniform_data):
        x, y, yerr = uniform_data
        rng = np.random.default_rng(2)
        x = x + rng.uniform(0.0, 0.05, len(x))
        gp = GPSolver(x, y, yerr, default_hparam, n_lat=16)
        assert gp.uniform_dt is None

    @pytest.mark.parametrize("solver", ["cholesky_banded", "cholesky_full"])
    def test_posterior_consistent_with_general_path(
            self, default_hparam, uniform_data, solver):
        x, y, yerr = uniform_data
        gp = GPSolver(x, y, yerr, default_hparam, n_lat=16,
                      matrix_solver=solver)
        assert gp.uniform_dt is not None

        lp_fast = float(gp.log_posterior(gp.theta0))
        g_fast = np.asarray(gp.grad_log_posterior(gp.theta0))
        ll_fast = gp.log_likelihood()

        # Rebuild the same solver with the fast path disabled
        gp.uniform_dt = None
        gp._build_covariance()
        gp._build_logposterior()
        lp_ref = float(gp.log_posterior(gp.theta0))
        g_ref = np.asarray(gp.grad_log_posterior(gp.theta0))
        ll_ref = gp.log_likelihood()

        assert np.isclose(lp_fast, lp_ref, rtol=1e-8)
        assert np.isclose(ll_fast, ll_ref, rtol=1e-8)
        np.testing.assert_allclose(g_fast, g_ref, rtol=1e-5, atol=1e-8)


class TestValueAndGrad:
    def test_wrappers_consistent(self, default_hparam, uniform_data):
        x, y, yerr = uniform_data
        gp = GPSolver(x, y, yerr, default_hparam, n_lat=16)
        gp.build_jax(recompute=False)
        t0 = gp.theta0

        val, grad = gp.value_and_grad_log_posterior(t0)
        assert np.isfinite(float(val))
        assert float(val) == pytest.approx(float(gp.log_posterior(t0)))
        assert float(gp.neg_log_posterior(t0)) == pytest.approx(-float(val))
        np.testing.assert_allclose(
            np.asarray(gp.grad_log_posterior(t0)), np.asarray(grad))
        np.testing.assert_allclose(
            np.asarray(gp.grad_neg_log_posterior(t0)), -np.asarray(grad))

    def test_grad_matches_finite_difference(self, default_hparam,
                                            uniform_data):
        x, y, yerr = uniform_data
        gp = GPSolver(x, y, yerr, default_hparam, n_lat=16)
        t0 = gp.theta0
        _, grad = gp.value_and_grad_log_posterior(t0)

        eps = 1e-5
        for i in (0, 4):  # peq and tau_spot
            tp = t0.at[i].add(eps)
            tm = t0.at[i].add(-eps)
            fd = (float(gp.log_posterior(tp))
                  - float(gp.log_posterior(tm))) / (2 * eps)
            assert np.isclose(float(grad[i]), fd, rtol=5e-4, atol=1e-6)

    def test_fit_map_runs(self, default_hparam, uniform_data):
        x, y, yerr = uniform_data
        gp = GPSolver(x, y, yerr, default_hparam, n_lat=16)
        theta_dict, result = gp.fit_map(keys=["peq", "sigma_k"], maxiter=5)
        assert set(theta_dict) == set(gp.param_keys)
        assert np.isfinite(float(result.fun))


class TestBandwidthWarning:
    """_check_bandwidth_efficiency: warn when the band defeats sparsity."""

    def test_warns_when_band_approaches_n(self, default_hparam, uniform_data):
        x, y, yerr = uniform_data
        with pytest.warns(UserWarning, match="banded Cholesky"):
            GPSolver(x, y, yerr, default_hparam, n_lat=16,
                     bandwidth=len(x) - 1)

    def test_no_warning_for_narrow_band(self, default_hparam, uniform_data):
        x, y, yerr = uniform_data
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as rec:
            _warnings.simplefilter("always")
            GPSolver(x, y, yerr, default_hparam, n_lat=16, bandwidth=5)
        assert not [w for w in rec
                    if "banded Cholesky" in str(w.message)]


class TestVmappedParallelFits:
    """method='L-BFGS-B' multi-start fits run as one vmapped jaxopt program."""

    def test_fit_map_parallel_lbfgsb(self, default_hparam, uniform_data):
        pytest.importorskip("jaxopt")
        x, y, yerr = uniform_data
        gp = GPSolver(x, y, yerr, default_hparam, n_lat=16)
        theta, res = gp.fit_map_parallel(
            nopt=3, keys=["peq", "sigma_k"], method="L-BFGS-B",
            maxiter=20, batch=True, rng=np.random.default_rng(1))

        assert "vmapped" in res.message
        assert np.isfinite(res.fun)
        assert set(theta) == set(gp.param_keys)

        # The reported objective must be the negative log-posterior of
        # the returned parameters (same posterior as the samplers use).
        theta_arr = jnp.array([theta[k] for k in gp.param_keys],
                              dtype=jnp.float64)
        assert float(res.fun) == pytest.approx(
            -float(gp.log_posterior(theta_arr)), rel=1e-10)

    def test_fit_map_parallel_return_all_sorted(self, default_hparam,
                                                uniform_data):
        pytest.importorskip("jaxopt")
        x, y, yerr = uniform_data
        gp = GPSolver(x, y, yerr, default_hparam, n_lat=16)
        thetas, ress = gp.fit_map_parallel(
            nopt=3, keys=["peq", "sigma_k"], method="L-BFGS-B",
            maxiter=20, batch=True, return_all=True,
            rng=np.random.default_rng(3))
        funs = [float(r.fun) for r in ress]
        assert funs == sorted(funs)
        assert len(thetas) == 3

    def test_fit_acf_parallel_lbfgsb(self, default_hparam, uniform_data):
        pytest.importorskip("jaxopt")
        x, y, yerr = uniform_data
        gp = GPSolver(x, y, yerr, default_hparam, n_lat=16)
        theta, res = gp.fit_acf_parallel(
            nopt=3, keys=["peq", "sigma_k"], method="L-BFGS-B",
            maxiter=20, batch=True, rng=np.random.default_rng(2))

        assert "vmapped" in res.message
        assert np.isfinite(res.fun)
        # Free parameters must respect their bounds
        kernel_keys = list(gp.spot_model.param_keys)
        bounds = np.asarray(gp.bounds[:len(kernel_keys)])
        for j, k in enumerate(kernel_keys):
            assert bounds[j, 0] - 1e-12 <= theta[k] <= bounds[j, 1] + 1e-12


class TestGappyCadence:
    """Uniform cadence with gaps: kernel evaluated once per distinct lag."""

    @pytest.fixture
    def gappy_data(self):
        """Two uniform segments (dt = 0.5) separated by a gap."""
        rng = np.random.default_rng(4)
        x = np.concatenate([np.arange(20) * 0.5,
                            30.0 + np.arange(20) * 0.5])
        y = 1.0 + 0.005 * rng.standard_normal(len(x))
        yerr = np.full(len(x), 0.001)
        return x, y, yerr

    def test_detect_cadence_offsets(self, gappy_data):
        from spotgp.gp_solver import _detect_cadence_offsets
        x = gappy_data[0]
        dt, off = _detect_cadence_offsets(x)
        assert dt == pytest.approx(0.5)
        np.testing.assert_array_equal(off[:3], [0, 1, 2])
        assert off[20] == 60  # 30.0 / 0.5

    def test_detect_rejects_irregular(self):
        from spotgp.gp_solver import _detect_cadence_offsets
        rng = np.random.default_rng(5)
        x = np.sort(rng.uniform(0, 20, 40))
        dt, off = _detect_cadence_offsets(x)
        assert dt is None and off is None

    def test_solver_detects_gappy_grid(self, default_hparam, gappy_data):
        x, y, yerr = gappy_data
        gp = GPSolver(x, y, yerr, default_hparam, n_lat=16)
        assert gp.uniform_dt is None
        assert gp._cadence_offsets is not None
        assert gp._band_lag_table() is not None

    @pytest.mark.parametrize("solver", ["cholesky_banded", "cholesky_full"])
    def test_posterior_consistent_with_general_path(
            self, default_hparam, gappy_data, solver):
        x, y, yerr = gappy_data
        gp = GPSolver(x, y, yerr, default_hparam, n_lat=16,
                      matrix_solver=solver)
        assert gp._cadence_offsets is not None

        lp_fast = float(gp.log_posterior(gp.theta0))
        g_fast = np.asarray(gp.grad_log_posterior(gp.theta0))
        ll_fast = gp.log_likelihood()

        # Rebuild with the fast path disabled
        gp._cadence_offsets = None
        gp._band_lag_table_cache = None
        gp._full_lag_table_cache = None
        gp._build_covariance()
        gp._build_logposterior()
        lp_ref = float(gp.log_posterior(gp.theta0))
        g_ref = np.asarray(gp.grad_log_posterior(gp.theta0))
        ll_ref = gp.log_likelihood()

        assert np.isclose(lp_fast, lp_ref, rtol=1e-8)
        assert np.isclose(ll_fast, ll_ref, rtol=1e-8)
        np.testing.assert_allclose(g_fast, g_ref, rtol=1e-5, atol=1e-8)
