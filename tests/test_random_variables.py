"""Tests for the per-spot random-variable declaration layer
(SpotRandomVariables) and its marginalized kernel term
(PopulationSpotTerm):

    K(tau) = V(tau) * sum_m w_m sigma_k(theta_m)^2 R_Gamma(tau; theta_m)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spotgp import GPSolver, SpotEvolutionModel
from spotgp.distributions import GaussianDistribution
from spotgp.random_variables import (
    Derived, Hyper, LogNormalLatent, SpotRandomVariables, UniformLatent,
    gnevyshev_waldmeier,
)
from spotgp.terms import KernelSum, PopulationSpotTerm, SHOTerm, SpotTerm

GEOM = dict(peq=10.0, kappa=0.2, inc=np.pi / 4)
HPARAM = dict(GEOM, lspot=2.0, tau_spot=0.5, sigma_k=0.01)


def _model(hparam=None):
    return SpotEvolutionModel.from_hparam(dict(hparam or HPARAM))


def _data(N=80):
    rng = np.random.default_rng(3)
    x = np.linspace(0.0, 25.0, N)
    y = (1.0 + 0.01 * np.sin(2 * np.pi * x / 10.0)
         + 0.002 * rng.standard_normal(N))
    return x, y, 0.002 * np.ones(N)


def _delta_rv(lspot=2.0, tau_spot=0.5, sigma_k=0.01):
    """All-hyper declaration: no latents, M = 1 node."""
    return SpotRandomVariables({
        "lspot": Hyper("lspot", lspot, (0.1, 20.0)),
        "tau_spot": Hyper("tau_spot", tau_spot, (0.05, 10.0)),
        "sigma_k": Hyper("sigma_k", sigma_k, (1e-6, 1.0)),
    })


class TestDeclarationValidation:
    def test_nonuniform_t_ref_rejected(self):
        with pytest.raises(NotImplementedError, match="stationar"):
            SpotRandomVariables({"lspot": 2.0},
                                t_ref=GaussianDistribution(5.0, 1.0))

    def test_nonuniform_phi_ref_rejected(self):
        with pytest.raises(NotImplementedError, match="UniformLongitude"):
            SpotRandomVariables({"lspot": 2.0}, phi_ref=object())

    def test_lam_ref_rejected(self):
        with pytest.raises(NotImplementedError, match="latitude quadrature"):
            SpotRandomVariables({"lspot": 2.0},
                                lam_ref=GaussianDistribution(0.5, 0.1))

    def test_geometry_key_rejected(self):
        with pytest.raises(ValueError, match="geometry"):
            SpotRandomVariables({"peq": Hyper("peq", 10.0)})

    def test_forward_reference_rejected(self):
        with pytest.raises(ValueError, match="declared before"):
            SpotRandomVariables({
                "lspot": Derived(lambda p: 2.0 * p["T"]),
                "T": 3.0,
            })

    def test_conflicting_hypers_rejected(self):
        with pytest.raises(ValueError, match="conflicting"):
            SpotRandomVariables({
                "a": LogNormalLatent(mean=Hyper("mu", 1.0), sigma=0.1),
                "b": LogNormalLatent(mean=Hyper("mu", 2.0), sigma=0.1),
            })

    def test_shared_hyper_instance_allowed(self):
        mu = Hyper("mu", 1.0)
        rv = SpotRandomVariables({
            "a": LogNormalLatent(mean=mu, sigma=0.1, n_quad=4),
            "b": LogNormalLatent(mean=mu, sigma=0.2, n_quad=4),
        })
        assert rv.hyper_keys == ("mu",)

    def test_param_hyper_name_collision_rejected(self):
        with pytest.raises(ValueError, match="both"):
            SpotRandomVariables({
                "x": LogNormalLatent(mean=Hyper("x", 1.0), sigma=0.1),
            })

    def test_bare_callable_becomes_derived(self):
        rv = SpotRandomVariables({"T": 3.0, "lspot": lambda p: 2 * p["T"]})
        ns, _ = rv.resolve(rv.hyper0)
        assert float(ns["lspot"]) == 6.0


class TestQuadrature:
    def test_lognormal_moments(self):
        rv = SpotRandomVariables({
            "T": LogNormalLatent(mean=Hyper("Tbar", 8.0),
                                 sigma=Hyper("sigma_T", 0.4), n_quad=16),
        })
        ns, w = rv.resolve(rv.hyper0)
        T, w = np.asarray(ns["T"]), np.asarray(w)
        np.testing.assert_allclose(np.sum(w), 1.0, rtol=1e-12)
        np.testing.assert_allclose(np.sum(w * T), 8.0, rtol=1e-8)
        np.testing.assert_allclose(np.sum(w * T ** 2),
                                   8.0 ** 2 * np.exp(0.4 ** 2), rtol=1e-8)

    def test_uniform_moments(self):
        rv = SpotRandomVariables({
            "x": UniformLatent(lo=Hyper("lo", 1.0), hi=Hyper("hi", 4.0),
                               n_quad=4),
        })
        ns, w = rv.resolve(rv.hyper0)
        x, w = np.asarray(ns["x"]), np.asarray(w)
        np.testing.assert_allclose(np.sum(w * x), 2.5, rtol=1e-12)
        np.testing.assert_allclose(np.sum(w * x ** 2),
                                   (1.0 + 4.0 + 16.0) / 3.0, rtol=1e-12)

    def test_tensor_product_nodes(self):
        rv = SpotRandomVariables({
            "a": UniformLatent(0.0, 1.0, n_quad=3),
            "b": UniformLatent(2.0, 3.0, n_quad=5),
        })
        assert rv.n_nodes == 15
        _, w = rv.resolve(rv.hyper0)
        np.testing.assert_allclose(float(jnp.sum(w)), 1.0, rtol=1e-12)

    def test_no_latents_single_node(self):
        rv = _delta_rv()
        assert rv.n_nodes == 1
        ns, w = rv.resolve(rv.hyper0)
        np.testing.assert_allclose(np.asarray(w), [1.0])
        assert float(ns["lspot"]) == 2.0


class TestGnevyshevWaldmeier:
    def test_hyper_layout(self):
        rv = gnevyshev_waldmeier(Tbar=8.0, sigma_T=0.4, sigma0=0.01, f=0.2)
        assert rv.hyper_keys == ("Tbar", "sigma_T", "sigma0")
        np.testing.assert_allclose(rv.hyper0, [8.0, 0.4, 0.01])
        assert "Tbar" in rv.hyper_bounds

    def test_fit_f_adds_hyper(self):
        rv = gnevyshev_waldmeier(fit_f=True)
        assert rv.hyper_keys == ("Tbar", "sigma_T", "sigma0", "f")

    def test_bad_fixed_f_rejected(self):
        with pytest.raises(ValueError, match="f must be"):
            gnevyshev_waldmeier(f=0.6)

    def test_couplings(self):
        """Lifetime = lspot + 2 tau_spot = T, amplitude proportional
        to T (the GW area-lifetime relation)."""
        rv = gnevyshev_waldmeier(Tbar=6.0, sigma_T=0.5, sigma0=0.02,
                                 f=0.2, n_quad=8)
        ns, _ = rv.resolve(rv.hyper0)
        T = np.asarray(ns["T"])
        np.testing.assert_allclose(
            np.asarray(ns["lspot"]) + 2 * np.asarray(ns["tau_spot"]),
            T, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(ns["sigma_k"]),
                                   0.02 * T / 6.0, rtol=1e-12)

    def test_sample_statistics(self):
        rv = gnevyshev_waldmeier(Tbar=8.0, sigma_T=0.4, sigma0=0.01, f=0.2)
        s = rv.sample(200_000, rng=np.random.default_rng(7))
        T = s["lspot"] + 2 * s["tau_spot"]
        np.testing.assert_allclose(np.mean(T), 8.0, rtol=0.02)
        # Deterministic GW coupling: size and lifetime perfectly rank-
        # correlated (correlation an independent model cannot express).
        r = np.corrcoef(np.log(s["sigma_k"]), np.log(T))[0, 1]
        assert r > 0.999999


class TestPopulationSpotTerm:
    def test_param_layout(self):
        term = PopulationSpotTerm(_model(), gnevyshev_waldmeier())
        assert term.base_keys == ("peq", "kappa", "inc",
                                  "Tbar", "sigma_T", "sigma0")
        np.testing.assert_allclose(term.theta0,
                                   [10.0, 0.2, np.pi / 4, 8.0, 0.4, 0.01])
        assert term.default_bounds["Tbar"] == (0.5, 50.0)
        assert term.default_bounds["peq"] == (0.5, 50.0)

    def test_missing_sigma_k_rejected(self):
        rv = SpotRandomVariables({"lspot": 2.0, "tau_spot": 0.5})
        with pytest.raises(ValueError, match="sigma_k"):
            PopulationSpotTerm(_model(), rv)

    def test_delta_declaration_matches_spot_term(self):
        """M = 1 node with hypers named like the physical keys: the
        marginalized kernel reduces to the plain single-spot kernel."""
        term = PopulationSpotTerm(_model(), _delta_rv())
        spot = SpotTerm(_model())
        assert term.base_keys == spot.base_keys
        np.testing.assert_allclose(term.theta0, spot.theta0)
        lag = jnp.linspace(0.0, 20.0, 200)
        np.testing.assert_allclose(
            np.asarray(term.k_of_lag(jnp.asarray(term.theta0), lag)),
            np.asarray(spot.k_of_lag(jnp.asarray(spot.theta0), lag)),
            rtol=1e-13, atol=1e-20)

    def test_marginalization_matches_manual_mixture(self):
        """The quadrature path equals a hand-built Gauss-Legendre
        mixture of fixed-parameter SpotTerm kernels."""
        rv = SpotRandomVariables({
            "lspot": UniformLatent(lo=Hyper("l_lo", 1.0, (0.5, 5.0)),
                                   hi=Hyper("l_hi", 4.0, (1.0, 10.0)),
                                   n_quad=4),
            "tau_spot": 0.5,
            "sigma_k": 0.01,
        })
        term = PopulationSpotTerm(_model(), rv)
        lag = jnp.linspace(0.0, 20.0, 150)
        k_pop = np.asarray(term.k_of_lag(jnp.asarray(term.theta0), lag))

        u, w = np.polynomial.legendre.leggauss(4)
        nodes = 1.0 + (4.0 - 1.0) * (u + 1.0) / 2.0
        k_manual = 0.0
        for lm, wm in zip(nodes, w / 2.0):
            st = SpotTerm(_model(dict(GEOM, lspot=float(lm),
                                      tau_spot=0.5, sigma_k=0.01)))
            k_manual = k_manual + wm * np.asarray(
                st.k_of_lag(jnp.asarray(st.theta0), lag))
        np.testing.assert_allclose(k_pop, k_manual, rtol=1e-10, atol=1e-18)

    def test_gw_narrow_limit_matches_point_kernel(self):
        """sigma_T -> 0: the GW population collapses to a single spot
        with lspot = (1-2f) Tbar, tau_spot = f Tbar, sigma_k = sigma0."""
        gw = gnevyshev_waldmeier(Tbar=6.0, sigma_T=1e-6, sigma0=0.01,
                                 f=0.2, n_quad=8)
        term = PopulationSpotTerm(_model(), gw)
        point = SpotTerm(_model(dict(GEOM, lspot=3.6, tau_spot=1.2,
                                     sigma_k=0.01)))
        lag = jnp.linspace(0.0, 20.0, 150)
        np.testing.assert_allclose(
            np.asarray(term.k_of_lag(jnp.asarray(term.theta0), lag)),
            np.asarray(point.k_of_lag(jnp.asarray(point.theta0), lag)),
            rtol=1e-6)

    def test_gradient_flows_to_hypers(self):
        term = PopulationSpotTerm(_model(), gnevyshev_waldmeier(n_quad=8))
        lag = jnp.linspace(0.0, 20.0, 100)
        grad = jax.grad(
            lambda th: jnp.sum(term.k_of_lag(th, lag)))(
                jnp.asarray(term.theta0))
        grad = np.asarray(grad)
        assert np.all(np.isfinite(grad))
        # Every hyper (Tbar, sigma_T, sigma0) moves the kernel.
        assert np.all(np.abs(grad[3:]) > 0)

    def test_jit_and_recall(self):
        term = PopulationSpotTerm(_model(), gnevyshev_waldmeier(n_quad=8))
        lag = jnp.linspace(0.0, 20.0, 100)
        k_jit = jax.jit(term.k_of_lag)
        theta1 = jnp.asarray(term.theta0)
        theta2 = theta1.at[3].set(12.0)  # new Tbar, same compiled fn
        # XLA fusion reorders float ops relative to eager mode, so
        # agreement is approximate, not bitwise.
        np.testing.assert_allclose(np.asarray(k_jit(theta1, lag)),
                                   np.asarray(term.k_of_lag(theta1, lag)),
                                   rtol=1e-8)
        assert np.all(np.isfinite(np.asarray(k_jit(theta2, lag))))

    def test_bandwidth_support_is_worst_case_lifetime(self):
        """With the GW structure, lspot + 2 tau_spot = T, so support =
        max node lifetime over the hyper bounds corners."""
        term = PopulationSpotTerm(_model(),
                                  gnevyshev_waldmeier(f=0.2, n_quad=16))
        keys = ["Tbar", "sigma_T", "sigma0"]
        rows = np.array([[4.0, 6.0], [0.05, 0.1], [1e-3, 1e-2]])
        support = term.bandwidth_support(keys, rows)
        z_max = np.sqrt(2.0) * np.max(np.polynomial.hermite.hermgauss(16)[0])
        t_max = 6.0 * np.exp(0.1 * z_max - 0.5 * 0.1 ** 2)
        np.testing.assert_allclose(support, t_max, rtol=1e-9)

    def test_psd_finite(self):
        term = PopulationSpotTerm(_model(), gnevyshev_waldmeier(n_quad=4))
        omega = np.linspace(0.1, 10.0, 32)
        freq, power = term.psd(omega)
        assert np.all(np.isfinite(np.asarray(power)))
        assert np.asarray(power).shape == omega.shape


class TestPopulationInSolver:
    def test_finite_logL_and_gradient(self):
        x, y, yerr = _data()
        term = PopulationSpotTerm(_model(), gnevyshev_waldmeier(n_quad=8))
        gp = GPSolver(x, y, yerr, term, matrix_solver="cholesky_full")
        assert gp.n_params == 6
        assert np.isfinite(float(gp.log_likelihood_fn(gp.theta0)))
        _, grad = gp.value_and_grad_log_posterior(gp.theta0)
        assert np.all(np.isfinite(np.asarray(grad)))

    def test_composes_in_kernel_sum(self):
        x, y, yerr = _data()
        ks = KernelSum(
            PopulationSpotTerm(_model(), gnevyshev_waldmeier(n_quad=4)),
            SHOTerm(S0=1e-4, Q=1 / np.sqrt(2), w0=2 * np.pi))
        assert "popspot0.Tbar" in ks.param_keys
        assert "sho0.S0" in ks.param_keys
        gp = GPSolver(x, y, yerr, ks, matrix_solver="cholesky_full")
        assert np.isfinite(float(gp.log_likelihood_fn(gp.theta0)))

    def test_banded_solver_with_bounded_hypers(self):
        x, y, yerr = _data()
        # Initial hypers must sit inside the prior box: the banded
        # support is computed from the bounds, and a theta0 outside
        # them would have wider support than the band covers.
        term = PopulationSpotTerm(
            _model(), gnevyshev_waldmeier(Tbar=4.0, sigma_T=0.08,
                                          sigma0=5e-3, f=0.2, n_quad=8))
        bounds = {"Tbar": (2.0, 6.0), "sigma_T": (0.05, 0.1),
                  "sigma0": (1e-3, 1e-2)}
        gp = GPSolver(x, y, yerr, term,
                      matrix_solver="cholesky_banded", bounds=bounds)
        assert 0 < gp.bandwidth <= len(x) - 1
        assert np.isfinite(float(gp.log_likelihood_fn(gp.theta0)))
