"""Tests for §3 (kernel consolidation) and §4 (polymorphic dispatch)."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from spotgp.analytic_kernel import AnalyticKernel, _kernel_eval
from spotgp.envelope import (
    TrapezoidSymmetricEnvelope,
    TrapezoidAsymmetricEnvelope,
    SkewedGaussianEnvelope,
    ExponentialEnvelope,
    ExponentialAsymmetricEnvelope,
    EnvelopeFunction,
    compute_R_Gamma_numerical,
)
from spotgp.spot_model import (
    SpotEvolutionModel,
    VisibilityFunction,
    EdgeOnVisibilityFunction,
)
from spotgp.latitude import LatitudeDistributionFunction


# =====================================================================
# §3: Kernel consolidation — class and functional paths must agree
# =====================================================================

class TestKernelConsolidation:
    """AnalyticKernel.kernel must agree with _kernel_eval to machine precision."""

    @pytest.fixture
    def model_symmetric(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
        return SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)

    @pytest.fixture
    def model_asymmetric(self):
        env = TrapezoidAsymmetricEnvelope(lspot=5.0, tau_em=0.5, tau_dec=1.5)
        vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
        return SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)

    @pytest.fixture
    def model_exponential(self):
        env = ExponentialEnvelope(tau_spot=2.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
        return SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)

    @pytest.fixture
    def model_edgeon(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = EdgeOnVisibilityFunction(peq=10.0)
        return SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)

    def _compare_class_and_functional(self, model, n_lat=64, quadrature="trapezoid"):
        """Assert AnalyticKernel.kernel and _kernel_eval agree."""
        ak = AnalyticKernel(model, n_lat=n_lat, quadrature=quadrature)
        lags = jnp.linspace(0, 15, 50)

        K_class = np.array(ak.kernel(lags))

        theta_arr = jnp.asarray(model.theta0)
        r_gamma_fn = model.get_r_gamma_func()

        if isinstance(model.visibility, EdgeOnVisibilityFunction):
            edgeon_cn_sq = jnp.array(
                model.visibility.cn_squared(0.0, ak.n_harmonics))
        else:
            edgeon_cn_sq = None

        # Pass same quadrature setup that the class uses internally
        if quadrature == "gauss-legendre":
            qn, qw = ak._quad_nodes, ak._quad_weights
            lat_dist = model.latitude_distribution
            user_w = jnp.array([lat_dist(float(p)) for p in qn])
            qw = user_w * qw
        else:
            qn, qw = None, None

        K_func = np.array(_kernel_eval(
            theta_arr, lags, ak.n_harmonics, n_lat, ak.lat_range,
            quad_nodes=qn, quad_weights=qw,
            r_gamma_func=r_gamma_fn,
            edgeon_cn_sq=edgeon_cn_sq,
        ))

        np.testing.assert_allclose(K_class, K_func, rtol=1e-10,
                                   err_msg="Class and functional kernel disagree")

    def test_symmetric_trapezoid(self, model_symmetric):
        self._compare_class_and_functional(model_symmetric)

    def test_asymmetric_trapezoid(self, model_asymmetric):
        self._compare_class_and_functional(model_asymmetric)

    def test_exponential(self, model_exponential):
        self._compare_class_and_functional(model_exponential)

    def test_edge_on(self, model_edgeon):
        self._compare_class_and_functional(model_edgeon)

    def test_gauss_legendre(self, model_symmetric):
        self._compare_class_and_functional(
            model_symmetric, n_lat=32, quadrature="gauss-legendre")


class TestNormalizationFix:
    """The old ~1.6% discrepancy must be gone."""

    def test_no_n_lat_over_n_lat_minus_1_factor(self):
        """At n_lat=64 the old discrepancy was ~1.6%. Now it must be < 1e-10."""
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)

        ak = AnalyticKernel(model, n_lat=64)
        lags = jnp.linspace(0, 6, 30)  # stay within kernel support
        K_class = np.array(ak.kernel(lags))

        theta_arr = jnp.asarray(model.theta0)
        r_gamma_fn = model.get_r_gamma_func()
        K_func = np.array(_kernel_eval(
            theta_arr, lags, ak.n_harmonics, 64, ak.lat_range,
            r_gamma_func=r_gamma_fn,
        ))

        np.testing.assert_allclose(K_class, K_func, rtol=1e-10,
                                   err_msg="Normalization discrepancy detected")


# =====================================================================
# §4: Polymorphic dispatch
# =====================================================================

class TestRGammaJax:
    """r_gamma_jax must match R_Gamma for each envelope type."""

    def test_trapezoid_symmetric(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        theta_env = jnp.array([5.0, 1.0])
        lags = jnp.linspace(0, 10, 50)
        R_method = np.array(env.R_Gamma(lags))
        R_jax = np.array(env.r_gamma_jax(theta_env, lags))
        np.testing.assert_allclose(R_method, R_jax, rtol=1e-12)

    def test_trapezoid_asymmetric(self):
        env = TrapezoidAsymmetricEnvelope(lspot=5.0, tau_em=0.5, tau_dec=1.5)
        theta_env = jnp.array([5.0, 0.5, 1.5])
        lags = jnp.linspace(0, 10, 50)
        R_method = np.array(env.R_Gamma(lags))
        R_jax = np.array(env.r_gamma_jax(theta_env, lags))
        np.testing.assert_allclose(R_method, R_jax, rtol=1e-12)

    def test_exponential(self):
        env = ExponentialEnvelope(tau_spot=2.0)
        theta_env = jnp.array([2.0])
        lags = jnp.linspace(0, 10, 50)
        R_method = np.array(env.R_Gamma(lags))
        R_jax = np.array(env.r_gamma_jax(theta_env, lags))
        np.testing.assert_allclose(R_method, R_jax, rtol=1e-12)

    def test_skewed_gaussian(self):
        env = SkewedGaussianEnvelope(sigma_sn=2.0, n_sn=-1.5)
        theta_env = jnp.array([2.0, -1.5, 0.0])
        lags = jnp.linspace(0, 20, 50)
        R_method = np.array(env.R_Gamma(lags))
        R_jax = np.array(env.r_gamma_jax(theta_env, lags))
        np.testing.assert_allclose(R_method, R_jax, rtol=1e-12)

    def test_exponential_asymmetric(self):
        env = ExponentialAsymmetricEnvelope(tau_em=1.0, tau_dec=3.0)
        theta_env = jnp.array([1.0, 3.0])
        lags = jnp.linspace(0, 15, 50)
        R_jax = np.array(env.r_gamma_jax(theta_env, lags))
        assert R_jax[0] > 0
        assert np.all(np.diff(R_jax) <= 1e-10)

    def test_exponential_asymmetric_symmetric_limit(self):
        """When tau_em == tau_dec, should match ExponentialEnvelope."""
        tau = 2.0
        env_sym = ExponentialEnvelope(tau_spot=tau)
        env_asym = ExponentialAsymmetricEnvelope(tau_em=tau, tau_dec=tau)
        lags = jnp.linspace(0, 10, 50)
        R_sym = np.array(env_sym.R_Gamma(lags))
        R_asym = np.array(env_asym.R_Gamma(lags))
        np.testing.assert_allclose(R_sym, R_asym, rtol=1e-8)

    def test_r_gamma_jax_differentiable(self):
        """Gradients must flow through r_gamma_jax."""
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        lag = jnp.array([1.0, 2.0, 3.0])

        def loss(theta_env):
            return jnp.sum(env.r_gamma_jax(theta_env, lag))

        grad = jax.grad(loss)(jnp.array([5.0, 1.0]))
        assert not np.any(np.isnan(grad))
        assert np.any(np.abs(grad) > 0)

    def test_custom_envelope_without_override_raises(self):
        """Custom envelope with free params but no r_gamma_jax must raise."""
        class BrokenEnvelope(EnvelopeFunction):
            @property
            def tau_spot(self):
                return 1.0

            @property
            def param_dict(self):
                return {"tau_spot": 1.0}

            def Gamma(self, t):
                return jnp.exp(-jnp.abs(t))

        env = BrokenEnvelope()
        with pytest.raises(NotImplementedError, match="r_gamma_jax"):
            env.r_gamma_jax(jnp.array([1.0]), jnp.array([1.0]))

    def test_custom_envelope_no_params_uses_default(self):
        """Custom envelope with no free params should use default R_Gamma."""
        class FixedEnvelope(EnvelopeFunction):
            @property
            def tau_spot(self):
                return 1.0

            def Gamma(self, t):
                return jnp.exp(-jnp.abs(t))

        env = FixedEnvelope()
        lags = jnp.linspace(0, 5, 20)
        R = env.r_gamma_jax(jnp.array([]), lags)
        assert R.shape == (20,)


class TestSupportFromBounds:
    """support_from_bounds must agree with the old isinstance logic."""

    def test_trapezoid_symmetric(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        upper_fn = lambda key, fallback: {"lspot": 20.0, "tau_spot": 10.0}.get(key, fallback)
        assert env.support_from_bounds(upper_fn) == 20.0 + 2.0 * 10.0

    def test_trapezoid_asymmetric(self):
        env = TrapezoidAsymmetricEnvelope(lspot=5.0, tau_em=0.5, tau_dec=1.5)
        upper_fn = lambda key, fallback: {"lspot": 10.0, "tau_em": 2.0, "tau_dec": 3.0}.get(key, fallback)
        assert env.support_from_bounds(upper_fn) == 10.0 + 2.0 + 3.0

    def test_exponential(self):
        env = ExponentialEnvelope(tau_spot=2.0)
        upper_fn = lambda key, fallback: {"tau_spot": 5.0}.get(key, fallback)
        assert env.support_from_bounds(upper_fn) == 6.0 * 5.0

    def test_skewed_gaussian(self):
        env = SkewedGaussianEnvelope(sigma_sn=2.0, n_sn=-1.5)
        upper_fn = lambda key, fallback: {"sigma_sn": 10.0}.get(key, fallback)
        assert env.support_from_bounds(upper_fn) == 12.0 * 10.0


class TestDispatchIntegration:
    """get_r_gamma_func and bandwidth_support should use polymorphic dispatch."""

    def test_get_r_gamma_func_symmetric(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)

        r_gamma = model.get_r_gamma_func()
        theta = jnp.asarray(model.theta0)
        lags = jnp.linspace(0, 10, 30)
        R = r_gamma(theta, lags)
        R_direct = env.R_Gamma(lags)
        np.testing.assert_allclose(np.array(R), np.array(R_direct), rtol=1e-12)

    def test_get_r_gamma_func_exponential(self):
        env = ExponentialEnvelope(tau_spot=2.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)

        r_gamma = model.get_r_gamma_func()
        theta = jnp.asarray(model.theta0)
        lags = jnp.linspace(0, 10, 30)
        R = r_gamma(theta, lags)
        R_direct = env.R_Gamma(lags)
        np.testing.assert_allclose(np.array(R), np.array(R_direct), rtol=1e-12)

    def test_bandwidth_support_matches_old_logic(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)

        keys = model.param_keys
        bounds = np.array([
            [0.5, 50.0],   # peq
            [0.001, 0.999], # kappa
            [0.01, 3.13],   # inc
            [0.1, 20.0],    # lspot
            [0.05, 10.0],   # tau_spot
            [1e-6, 1.0],    # sigma_k
        ])
        support = model.bandwidth_support(keys, bounds)
        assert support == 20.0 + 2.0 * 10.0


# =====================================================================
# §4: Bug fixes
# =====================================================================

class TestGaussLegendreLatRangeBug:
    """AnalyticKernel(model, quadrature='gauss-legendre') without explicit
    lat_range must not crash — it should use self.lat_range."""

    def test_no_crash_without_lat_range(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)
        ak = AnalyticKernel(model, quadrature="gauss-legendre")
        K = ak.kernel(jnp.array([0.0, 1.0, 2.0]))
        assert K.shape == (3,)
        assert float(K[0]) > 0


class TestLatWeightFuncWarning:
    """get_lat_weight_func must warn for custom distributions with free params."""

    def test_warns_on_custom_parameterized_distribution(self):
        class CustomLatDist(LatitudeDistributionFunction):
            @property
            def param_dict(self):
                return {"my_param": 1.0}

            def __call__(self, phi):
                return 1.0

        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
        model = SpotEvolutionModel(
            envelope=env, visibility=vis, sigma_k=0.01,
            latitude_distribution=CustomLatDist())

        with pytest.warns(UserWarning, match="zero gradient"):
            model.get_lat_weight_func()
