"""Tests for the Term / KernelSum seam (Phase 00: no behavior change).

Two layers of protection:

1. ``test_pinned_*`` — likelihood/posterior/gradient values captured on
   main *before* the seam refactor (commit ffbc9bb), pinned here.  A
   tight relative tolerance (30 ulps) allows for cross-platform XLA
   codegen differences while still catching any real change in kernel
   math or normalization.

2. ``test_seam_matches_legacy_closure_*`` — in-process bit-for-bit
   equality between the solver's ``k_of_lag`` seam and the legacy
   ``_kernel_eval`` closure-kwarg path, on the same machine in the same
   run.  This is the exact "pure refactor" guarantee.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from spotgp import AnalyticKernel, GPSolver, SpotEvolutionModel
from spotgp.analytic_kernel import _kernel_eval
from spotgp.gp_solver import _gp_log_likelihood, _gp_log_likelihood_banded
from spotgp.terms import DEFAULT_TERM_BOUNDS, KernelSum, SpotTerm

HPARAM = dict(peq=10.0, kappa=0.2, inc=np.pi / 4, lspot=5.0,
              tau_spot=1.0, sigma_k=0.01)

# ~30 ulps at logL ≈ 540: tolerates platform-level codegen variation,
# catches any genuine change in kernel math.
RTOL = 1e-12

# Originally captured on main @ ffbc9bb (pre-refactor), jax 0.10.0 x64 CPU.
#
# Re-captured when the kernel began taking its harmonic orders from
# ``visibility.harmonics``, which moved the default set from 0..3 to 0..2.
# The values below were validated against the *pre-change* implementation
# run with an explicit ``n_harmonics=2``: the two agree to 1.2e-14 relative
# (worst case: grad0), so only the default order set moved — the kernel math
# itself is unchanged.
PINS = {
    "uniform/cholesky_banded": {
        "logL_fn": 537.2845653996955,
        "logL_init": 537.2845653996956,
        "logpost": 526.6446715274975,
        "grad0": 0.5239965812265719,
    },
    "uniform/cholesky_full": {
        "logL_fn": 537.2845653996955,
        "logL_init": 537.2845653996956,
        "logpost": 526.6446715274975,
        "grad0": 0.5239965812265653,
    },
    "irregular/cholesky_banded": {
        "logL_fn": 532.4534737577502,
        "logL_init": 532.4534737577502,
        "logpost": 521.8135798855523,
        "grad0": 0.48252885278164437,
    },
    "irregular/cholesky_full": {
        "logL_fn": 532.4534737577502,
        "logL_init": 532.4534737577502,
        "logpost": 521.8135798855523,
        "grad0": 0.4825288527816439,
    },
    "gappy/cholesky_banded": {
        "logL_fn": 538.2710188149504,
        "logL_init": 538.2710188149503,
        "logpost": 527.6311249427524,
        "grad0": 0.47607743658135165,
    },
    "gappy/cholesky_full": {
        "logL_fn": 538.2710188149504,
        "logL_init": 538.2710188149503,
        "logpost": 527.6311249427524,
        "grad0": 0.4760774365813532,
    },
    "uniform/banded/logspace": {
        "logL_fn": 537.2845624907399,
        "logL_init": 537.2845653996956,
        "logpost": 522.8636116389628,
        "grad0": 0.5239965820302568,
    },
    "uniform/banded/gl": {
        "logL_fn": 537.3033092397471,
        "logL_init": 537.3033092397471,
        "logpost": 526.6634153675492,
        "grad0": 0.5298326734343418,
    },
    "uniform/banded/b40": {
        "logL_fn": 548.0256514915891,
        "logL_init": 548.0256514915891,
        "logpost": 537.3857576193911,
        "grad0": 0.5471190414466703,
    },
}

EXPECTED_KEYS = ("peq", "kappa", "inc", "lspot", "tau_spot", "sigma_k")
EXPECTED_THETA0 = [10.0, 0.2, np.pi / 4, 5.0, 1.0, 0.01]


def _capture_grids():
    """Reproduce the exact rng consumption order of the capture script."""
    rng = np.random.default_rng(42)
    N = 120
    x_uni = np.linspace(0.0, 30.0, N)
    x_irr = np.sort(rng.uniform(0.0, 30.0, N))
    x_gap = np.linspace(0.0, 40.0, 160)
    x_gap = np.delete(x_gap, slice(60, 100))

    def y_of(x):
        return (1.0 + 0.01 * np.sin(2 * np.pi * x / 10.0)
                + 0.002 * rng.standard_normal(len(x)))

    # Draw order matters: uniform, then irregular, then gappy.
    data = {}
    for name, x in [("uniform", x_uni), ("irregular", x_irr),
                    ("gappy", x_gap)]:
        data[name] = (x, y_of(x))
    return data


def _b40_data():
    """The b40 pin was captured with its own rng (no uniform draw first)."""
    rng = np.random.default_rng(42)
    N = 120
    x = np.linspace(0.0, 30.0, N)
    y = (1.0 + 0.01 * np.sin(2 * np.pi * x / 10.0)
         + 0.002 * rng.standard_normal(N))
    return x, y


def _check_case(gp, pins):
    np.testing.assert_allclose(
        float(gp.log_likelihood_fn(gp.theta0)), pins["logL_fn"], rtol=RTOL)
    np.testing.assert_allclose(
        float(gp.log_likelihood()), pins["logL_init"], rtol=RTOL)
    np.testing.assert_allclose(
        float(gp.log_posterior(gp.theta0)), pins["logpost"], rtol=RTOL)
    np.testing.assert_allclose(
        float(gp.value_and_grad_log_posterior(gp.theta0)[1][0]),
        pins["grad0"], rtol=RTOL)


class TestPinnedRegression:
    """Solver output is unchanged by the Term seam (pinned on main)."""

    @pytest.mark.parametrize("grid", ["uniform", "irregular", "gappy"])
    @pytest.mark.parametrize("solver", ["cholesky_banded", "cholesky_full"])
    def test_pinned_likelihoods(self, grid, solver):
        x, y = _capture_grids()[grid]
        gp = GPSolver(x, y, 0.002 * np.ones_like(x), dict(HPARAM),
                      matrix_solver=solver)
        assert tuple(gp.param_keys) == EXPECTED_KEYS
        np.testing.assert_allclose(np.asarray(gp.theta0), EXPECTED_THETA0)
        _check_case(gp, PINS[f"{grid}/{solver}"])

    def test_pinned_logspace_sigma_n(self):
        x, y = _capture_grids()["uniform"]
        bounds = {"log_sigma_k": (-6.0, 0.0), "log_sigma_n": (-6.0, -1.0)}
        gp = GPSolver(x, y, 0.002 * np.ones_like(x), dict(HPARAM),
                      matrix_solver="cholesky_banded",
                      fit_sigma_n=True, bounds=bounds)
        assert tuple(gp.param_keys) == EXPECTED_KEYS[:5] + (
            "log_sigma_k", "log_sigma_n")
        np.testing.assert_allclose(
            np.asarray(gp.theta0), EXPECTED_THETA0[:5] + [-2.0, -6.0])
        _check_case(gp, PINS["uniform/banded/logspace"])

    def test_pinned_gauss_legendre(self):
        x, y = _capture_grids()["uniform"]
        model = SpotEvolutionModel.from_hparam(dict(HPARAM))
        gp = GPSolver(x, y, 0.002 * np.ones_like(x), model,
                      matrix_solver="cholesky_banded",
                      quadrature="gauss-legendre")
        _check_case(gp, PINS["uniform/banded/gl"])

    def test_pinned_narrow_band(self):
        x, y = _b40_data()
        gp = GPSolver(x, y, 0.002 * np.ones(len(x)), dict(HPARAM),
                      matrix_solver="cholesky_banded", bandwidth=40)
        _check_case(gp, PINS["uniform/banded/b40"])


class TestSeamMatchesLegacyClosure:
    """k_of_lag seam == legacy _kernel_eval closure path, bit for bit."""

    def _legacy_logL(self, gp):
        """Evaluate the likelihood exactly as pre-seam GPSolver did.

        The raw function is wrapped in jax.jit to match the solver's
        compiled path — eager evaluation fuses differently and drifts by
        ~1 ulp, which the bit-for-bit assertion must not conflate with a
        real seam difference.
        """
        import jax

        from spotgp.spot_model import EdgeOnVisibilityFunction

        model = gp.spot_model
        r_gamma_fn = model.get_r_gamma_func()
        lat_wt_fn = model.get_lat_weight_func()
        cn_sq_fn = model.get_cn_sq_func(gp.n_harmonics)
        if isinstance(model.visibility, EdgeOnVisibilityFunction):
            eo_cn = jnp.array(model.visibility.cn_squared(
                0.0, gp.n_harmonics))
        else:
            eo_cn = None
        n_kernel = len(model.param_keys)
        common = dict(
            n_kernel=n_kernel, r_gamma_func=r_gamma_fn,
            quad_nodes=gp._quad_nodes, quad_weights=gp._quad_weights,
            edgeon_cn_sq=eo_cn, lat_weight_func=lat_wt_fn,
            cn_sq_func=cn_sq_fn, uniform_dt=gp.uniform_dt)
        if gp.matrix_solver == "cholesky_banded":
            band_tab = gp._band_lag_table()

            def raw(theta_arr):
                return _gp_log_likelihood_banded(
                    gp._to_physical(theta_arr), gp.x, gp.y, gp.yerr,
                    gp.mean_val, gp.n_harmonics, gp.n_lat, gp.lat_range,
                    gp.fit_sigma_n, gp.bandwidth,
                    band_lag_table=band_tab, **common)
        else:
            full_tab = gp._full_lag_table()

            def raw(theta_arr):
                return _gp_log_likelihood(
                    gp._to_physical(theta_arr), gp.x, gp.y, gp.yerr,
                    gp.mean_val, gp.n_harmonics, gp.n_lat, gp.lat_range,
                    gp.fit_sigma_n, lag_table=full_tab, **common)

        return float(jax.jit(raw)(gp.theta0))

    @pytest.mark.parametrize("grid", ["uniform", "irregular", "gappy"])
    @pytest.mark.parametrize("solver", ["cholesky_banded", "cholesky_full"])
    def test_seam_matches_legacy_closure(self, grid, solver):
        x, y = _capture_grids()[grid]
        gp = GPSolver(x, y, 0.002 * np.ones_like(x), dict(HPARAM),
                      matrix_solver=solver)
        seam = float(gp.log_likelihood_fn(gp.theta0))
        legacy = self._legacy_logL(gp)
        assert seam == legacy  # bit-for-bit: identical traced program


class TestSingleTermTransparency:
    """KernelSum(SpotTerm(model)) is a transparent wrapper."""

    def test_param_layout_matches_model(self):
        model = SpotEvolutionModel.from_hparam(dict(HPARAM))
        ks = KernelSum(SpotTerm(model))
        assert ks.param_keys == tuple(model.param_keys)
        np.testing.assert_array_equal(ks.theta0, model.theta0)
        assert ks.stationary

    def test_k_of_lag_matches_kernel_eval(self):
        model = SpotEvolutionModel.from_hparam(dict(HPARAM))
        term = SpotTerm(model)
        lag = jnp.linspace(0.0, 20.0, 64)
        theta = jnp.asarray(model.theta0)
        got = term.k_of_lag(theta, lag)
        want = _kernel_eval(
            theta, lag, term.n_harmonics, term.n_lat, term.lat_range,
            quad_nodes=term._quad_nodes, quad_weights=term._quad_weights,
            r_gamma_func=model.get_r_gamma_func(),
            edgeon_cn_sq=None,
            lat_weight_func=model.get_lat_weight_func(),
            cn_sq_func=model.get_cn_sq_func(term.n_harmonics))
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    def test_spot_term_accepts_analytic_kernel(self):
        ak = AnalyticKernel(dict(HPARAM))
        term = SpotTerm(analytic_kernel=ak)
        assert term.analytic_kernel is ak
        assert term.spot_model is ak.spot_model

    def test_spot_term_accepts_hparam_dict(self):
        term = SpotTerm(dict(HPARAM))
        assert term.base_keys == EXPECTED_KEYS

    def test_default_bounds_single_source(self):
        assert GPSolver.DEFAULT_BOUNDS == DEFAULT_TERM_BOUNDS

    def test_solver_exposes_kernel_sum(self):
        x, y = _b40_data()
        gp = GPSolver(x, y, 0.002 * np.ones(len(x)), dict(HPARAM),
                      matrix_solver="cholesky_full")
        assert isinstance(gp.kernel_sum, KernelSum)
        assert len(gp.kernel_sum.terms) == 1
        assert isinstance(gp.kernel_sum.terms[0], SpotTerm)
        # Single unprefixed term keeps bare keys
        assert gp.kernel_sum.terms[0].prefix is None

    def test_update_hparam_rebuilds_kernel_sum(self):
        x, y = _b40_data()
        gp = GPSolver(x, y, 0.002 * np.ones(len(x)), dict(HPARAM),
                      matrix_solver="cholesky_full")
        old_sum = gp.kernel_sum
        logL_before = float(gp.log_likelihood())
        new_hparam = dict(HPARAM, sigma_k=0.02)
        gp.update_hparam(new_hparam)
        assert gp.kernel_sum is not old_sum
        assert gp.kernel_sum.terms[0].spot_model is gp.spot_model
        assert float(gp.log_likelihood()) != logL_before


class TestKernelSumValidation:
    def test_requires_terms(self):
        with pytest.raises(ValueError, match="at least one term"):
            KernelSum()

    def test_rejects_non_terms(self):
        with pytest.raises(TypeError, match="Term instances"):
            KernelSum(AnalyticKernel(dict(HPARAM)))


# ─────────────────────────────────────────────────────────────────────
# Phase 01: additive composition of stationary terms
# ─────────────────────────────────────────────────────────────────────

HPARAM_SHORT = dict(peq=10.0, kappa=0.2, inc=np.pi / 4, lspot=2.0,
                    tau_spot=0.5, sigma_k=0.01)
HPARAM_LONG = dict(peq=10.0, kappa=0.2, inc=np.pi / 4, lspot=8.0,
                   tau_spot=2.0, sigma_k=0.005)


def _small_data(N=80):
    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 25.0, N)
    y = (1.0 + 0.01 * np.sin(2 * np.pi * x / 10.0)
         + 0.002 * rng.standard_normal(N))
    return x, y, 0.002 * np.ones(N)


def _two_spot_sum():
    return KernelSum(SpotTerm(dict(HPARAM_SHORT)),
                     SpotTerm(dict(HPARAM_LONG)))


class TestAdditiveComposition:
    """Phase 01: KernelSum genuinely sums more than one term."""

    def test_auto_prefixing(self):
        ks = _two_spot_sum()
        assert ks.param_keys == tuple(
            [f"spot0.{k}" for k in EXPECTED_KEYS]
            + [f"spot1.{k}" for k in EXPECTED_KEYS])

    def test_explicit_prefix(self):
        ks = KernelSum(SpotTerm(dict(HPARAM_SHORT), prefix="short"),
                       SpotTerm(dict(HPARAM_LONG), prefix="long"))
        assert ks.param_keys[0] == "short.peq"
        assert ks.param_keys[6] == "long.peq"

    def test_duplicate_prefix_raises(self):
        with pytest.raises(ValueError, match="Duplicate parameter keys"):
            KernelSum(SpotTerm(dict(HPARAM_SHORT), prefix="a"),
                      SpotTerm(dict(HPARAM_LONG), prefix="a"))

    def test_theta0_concat(self):
        ks = _two_spot_sum()
        np.testing.assert_array_equal(
            ks.theta0,
            np.concatenate([ks.terms[0].theta0, ks.terms[1].theta0]))

    def test_k_of_lag_is_sum_of_terms(self):
        ks = _two_spot_sum()
        lag = jnp.linspace(0.0, 15.0, 50)
        theta = jnp.asarray(ks.theta0)
        total = np.asarray(ks.k_of_lag(theta, lag))
        parts = sum(
            np.asarray(t.k_of_lag(jnp.asarray(t.theta0), lag))
            for t in ks.terms)
        np.testing.assert_array_equal(total, parts)

    def test_bandwidth_is_max_over_terms(self):
        ks = _two_spot_sum()
        keys = ks.param_keys
        bounds = np.array([DEFAULT_TERM_BOUNDS[k.split(".")[1]]
                           for k in keys])
        # Tighten spot0's envelope so spot1 dominates the support
        bounds[keys.index("spot0.lspot"), 1] = 1.0
        bounds[keys.index("spot0.tau_spot"), 1] = 0.2
        total = ks.bandwidth_support(keys, bounds)
        t1_keys, t1_rows = ks.terms[1]._own_bounds_rows(keys, bounds)
        want = ks.terms[1].bandwidth_support(t1_keys, t1_rows)
        assert total == want

    def test_gram_is_sum_of_grams(self):
        x, y, yerr = _small_data()
        gp_ab = GPSolver(x, y, yerr, _two_spot_sum(),
                         matrix_solver="cholesky_full")
        gp_a = GPSolver(x, y, yerr, dict(HPARAM_SHORT),
                        matrix_solver="cholesky_full")
        gp_b = GPSolver(x, y, yerr, dict(HPARAM_LONG),
                        matrix_solver="cholesky_full")
        # self.K is the pure kernel Gram (noise lives in K_noise), so
        # the composite Gram must equal the sum of the per-term Grams.
        np.testing.assert_allclose(np.asarray(gp_ab.K),
                                   np.asarray(gp_a.K) + np.asarray(gp_b.K),
                                   rtol=1e-15, atol=1e-20)

    @pytest.mark.parametrize("solver", ["cholesky_banded", "cholesky_full"])
    def test_finite_logL_and_gradient(self, solver):
        x, y, yerr = _small_data()
        gp = GPSolver(x, y, yerr, _two_spot_sum(), matrix_solver=solver,
                      bandwidth=30 if solver == "cholesky_banded" else None)
        assert gp.n_params == 12
        logL = float(gp.log_likelihood_fn(gp.theta0))
        assert np.isfinite(logL)
        val, grad = gp.value_and_grad_log_posterior(gp.theta0)
        assert np.isfinite(float(val))
        assert np.all(np.isfinite(np.asarray(grad)))

    def test_toeplitz_gram_matches_direct_eval(self):
        x, y, yerr = _small_data()
        ks = _two_spot_sum()
        gp = GPSolver(x, y, yerr, ks, matrix_solver="cholesky_full")
        lag = jnp.abs(gp.x[:, None] - gp.x[None, :])
        K_direct = np.asarray(
            ks.k_of_lag(jnp.asarray(ks.theta0), lag.ravel())
        ).reshape(gp.N, gp.N)
        np.testing.assert_allclose(np.asarray(gp.K), K_direct,
                                   rtol=1e-12, atol=1e-16)

    def test_predict_finite(self):
        x, y, yerr = _small_data()
        gp = GPSolver(x, y, yerr, _two_spot_sum(),
                      matrix_solver="cholesky_full")
        xp = np.linspace(0.0, 25.0, 40)
        mu, var = gp.predict(xp)
        assert np.all(np.isfinite(np.asarray(mu)))
        assert np.all(np.isfinite(np.asarray(var)))

    def test_single_term_input_matches_wrapped_dict(self):
        x, y, yerr = _small_data()
        gp_term = GPSolver(x, y, yerr, SpotTerm(dict(HPARAM_SHORT)),
                           matrix_solver="cholesky_full")
        gp_dict = GPSolver(x, y, yerr, dict(HPARAM_SHORT),
                           matrix_solver="cholesky_full")
        assert gp_term.param_keys == gp_dict.param_keys
        np.testing.assert_allclose(
            float(gp_term.log_likelihood_fn(gp_term.theta0)),
            float(gp_dict.log_likelihood_fn(gp_dict.theta0)),
            rtol=1e-14)

    def test_nonstationary_term_rejected(self):
        class FakeNonstationary(SpotTerm):
            stationary = False

        x, y, yerr = _small_data()
        with pytest.raises(NotImplementedError, match="stationary"):
            GPSolver(x, y, yerr,
                     KernelSum(FakeNonstationary(dict(HPARAM_SHORT))),
                     matrix_solver="cholesky_full")

    def test_update_hparam_composite_raises(self):
        x, y, yerr = _small_data()
        gp = GPSolver(x, y, yerr, _two_spot_sum(),
                      matrix_solver="cholesky_full")
        with pytest.raises(NotImplementedError, match="composite"):
            gp.update_hparam(dict(HPARAM_SHORT))

    def test_default_bounds_align_with_prefixed_keys(self):
        x, y, yerr = _small_data()
        gp = GPSolver(x, y, yerr, _two_spot_sum(),
                      matrix_solver="cholesky_full")
        i = gp.param_keys.index("spot1.lspot")
        np.testing.assert_array_equal(np.asarray(gp.bounds[i]),
                                      DEFAULT_TERM_BOUNDS["lspot"])


# ─────────────────────────────────────────────────────────────────────
# Phase 02: parameter namespacing & plumbing
# ─────────────────────────────────────────────────────────────────────

class TestParameterPlumbing:
    """Prefixed log-space remap, PGM, save/load, and sampling layout."""

    def test_prefixed_log_space_remap(self):
        x, y, yerr = _small_data()
        bounds = {"spot0.log_sigma_k": (-6.0, 0.0),
                  "spot1.log_sigma_k": (-6.0, 0.0)}
        gp = GPSolver(x, y, yerr, _two_spot_sum(),
                      matrix_solver="cholesky_full", bounds=bounds)
        assert "spot0.log_sigma_k" in gp.param_keys
        assert "spot1.log_sigma_k" in gp.param_keys
        i0 = gp.param_keys.index("spot0.log_sigma_k")
        i1 = gp.param_keys.index("spot1.log_sigma_k")
        np.testing.assert_allclose(float(gp.theta0[i0]), np.log10(0.01))
        np.testing.assert_allclose(float(gp.theta0[i1]), np.log10(0.005))
        # _to_physical undoes the log10
        phys = np.asarray(gp._to_physical(gp.theta0))
        np.testing.assert_allclose(phys[i0], 0.01)
        np.testing.assert_allclose(phys[i1], 0.005)
        assert np.isfinite(float(gp.log_likelihood_fn(gp.theta0)))

    def test_pgm_reads_composite_keys(self):
        from spotgp import PGModelVis

        x, y, yerr = _small_data()
        gp = GPSolver(x, y, yerr, _two_spot_sum(),
                      matrix_solver="cholesky_full")
        vis = PGModelVis(gp)
        # Duplicated per-term keys collapse onto single structural nodes
        assert vis.rotation_params.count("peq") == 1
        assert "lspot" in vis.envelope_params
        assert "sigma_k" in vis.amplitude_params

    def test_validation_sees_prefixed_peq(self):
        # A composite kernel with an absurd peq upper bound must still
        # trigger the baseline warning (prefix-stripped lookup).
        x, y, yerr = _small_data()
        bounds = {"spot0.peq": (0.5, 500.0)}
        with pytest.warns(UserWarning, match="baseline"):
            GPSolver(x, y, yerr, _two_spot_sum(),
                     matrix_solver="cholesky_full", bounds=bounds)

    def test_save_load_roundtrip_composite(self, tmp_path):
        from spotgp import load_gp, save_gp

        x, y, yerr = _small_data()
        ks = KernelSum(SpotTerm(dict(HPARAM_SHORT), prefix="short"),
                       SpotTerm(dict(HPARAM_LONG), prefix="long"))
        gp = GPSolver(x, y, yerr, ks, matrix_solver="cholesky_full")
        path = str(tmp_path / "composite.h5")
        save_gp(path, gp)
        gp2 = load_gp(path)
        assert gp2.param_keys == gp.param_keys
        np.testing.assert_allclose(np.asarray(gp2.theta0),
                                   np.asarray(gp.theta0))
        np.testing.assert_array_equal(np.asarray(gp2.bounds),
                                      np.asarray(gp.bounds))
        assert [t.prefix for t in gp2.kernel_sum.terms] == ["short", "long"]
        np.testing.assert_allclose(
            float(gp2.log_likelihood_fn(gp2.theta0)),
            float(gp.log_likelihood_fn(gp.theta0)), rtol=1e-12)

    def test_save_load_roundtrip_single_spot_unchanged(self, tmp_path):
        from spotgp import load_gp, save_gp

        x, y, yerr = _small_data()
        gp = GPSolver(x, y, yerr, dict(HPARAM_SHORT),
                      matrix_solver="cholesky_full")
        path = str(tmp_path / "single.h5")
        save_gp(path, gp)
        gp2 = load_gp(path)
        assert gp2.param_keys == gp.param_keys
        np.testing.assert_allclose(np.asarray(gp2.theta0),
                                   np.asarray(gp.theta0))

    def test_sampler_summary_uses_prefixed_keys(self, tmp_path):
        from spotgp.mcmc import MCMCSampler

        x, y, yerr = _small_data()
        gp = GPSolver(x, y, yerr, _two_spot_sum(),
                      matrix_solver="cholesky_full")
        sampler = MCMCSampler(gp)
        assert sampler.n_params == 12
        assert sampler.param_keys == gp.param_keys
        rng = np.random.default_rng(0)
        sampler.samples = (np.asarray(gp.theta0)[None, :]
                           * (1.0 + 0.01 * rng.standard_normal((50, 12))))
        result = sampler.summary()
        assert set(result.keys()) >= set(gp.param_keys)

    def test_blackjax_short_chain_two_terms(self, tmp_path):
        pytest.importorskip("blackjax")
        from spotgp.mcmc import BlackJAXSampler

        x, y, yerr = _small_data(N=50)
        gp = GPSolver(x, y, yerr, _two_spot_sum(),
                      matrix_solver="cholesky_full")
        gp.build_jax()
        sampler = BlackJAXSampler(gp, save_dir=str(tmp_path))
        sampler.run_warmup(n_warmup=10)
        sampler.run_sampling(n_samples=10)
        samples = np.asarray(sampler.samples)
        assert samples.shape[-1] == 12
        assert np.all(np.isfinite(samples))
