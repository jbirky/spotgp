"""Tests for spotgp.distributions — ParameterDistribution classes."""

import numpy as np
import pytest

from spotgp.distributions import (
    ParameterDistribution,
    DeltaDistribution,
    UniformDistribution,
    GaussianDistribution,
    LogNormalDistribution,
    as_distribution,
    is_distributed,
)
from spotgp.spot_model import SpotEvolutionModel
from spotgp.envelope import TrapezoidSymmetricEnvelope
from spotgp.visibility import VisibilityFunction
from spotgp.analytic_kernel import AnalyticKernel


# ── as_distribution / is_distributed ─────────────────────────

class TestAsDistribution:
    def test_float_becomes_delta(self):
        d = as_distribution(0.01)
        assert isinstance(d, DeltaDistribution)
        assert d.mean == 0.01

    def test_distribution_passes_through(self):
        u = UniformDistribution(1.0, 5.0)
        assert as_distribution(u) is u

    def test_is_distributed_false_for_float(self):
        assert not is_distributed(0.01)

    def test_is_distributed_false_for_delta(self):
        assert not is_distributed(DeltaDistribution(0.01))

    def test_is_distributed_true_for_uniform(self):
        assert is_distributed(UniformDistribution(1.0, 5.0))


# ── DeltaDistribution ───────────────────────────────────────

class TestDelta:
    def test_mean(self):
        d = DeltaDistribution(3.14)
        assert d.mean == 3.14

    def test_expectation(self):
        d = DeltaDistribution(2.0)
        assert d.expectation(lambda x: x ** 2) == 4.0

    def test_support(self):
        d = DeltaDistribution(5.0)
        assert d.support == (5.0, 5.0)

    def test_repr(self):
        assert "3.14" in repr(DeltaDistribution(3.14))


# ── UniformDistribution ─────────────────────────────────────

class TestUniform:
    def test_mean(self):
        d = UniformDistribution(2.0, 8.0)
        assert d.mean == 5.0

    def test_support(self):
        d = UniformDistribution(2.0, 8.0)
        assert d.support == (2.0, 8.0)

    def test_expectation_identity(self):
        d = UniformDistribution(0.0, 10.0)
        np.testing.assert_allclose(d.expectation(lambda x: x), 5.0, rtol=1e-6)

    def test_expectation_squared(self):
        d = UniformDistribution(0.0, 1.0)
        # E[x^2] for Uniform(0,1) = 1/3
        np.testing.assert_allclose(d.expectation(lambda x: x ** 2), 1 / 3, rtol=1e-6)

    def test_sample(self):
        d = UniformDistribution(2.0, 8.0)
        s = d.sample(100, rng=np.random.default_rng(42))
        assert len(s) == 100
        assert np.all(s >= 2.0) and np.all(s <= 8.0)


# ── GaussianDistribution ────────────────────────────────────

class TestGaussian:
    def test_mean(self):
        d = GaussianDistribution(5.0, 1.0)
        assert d.mean == 5.0

    def test_support(self):
        d = GaussianDistribution(5.0, 1.0, clip_sigma=3)
        assert d.support == (2.0, 8.0)

    def test_expectation_mean(self):
        d = GaussianDistribution(5.0, 1.0)
        np.testing.assert_allclose(d.expectation(lambda x: x), 5.0, rtol=1e-4)

    def test_expectation_variance(self):
        d = GaussianDistribution(0.0, 2.0)
        # E[x^2] for N(0, 2) ≈ 4
        np.testing.assert_allclose(d.expectation(lambda x: x ** 2), 4.0, rtol=0.02)

    def test_pdf_peak_at_mean(self):
        d = GaussianDistribution(5.0, 1.0)
        assert d(5.0) > d(3.0)
        assert d(5.0) > d(7.0)


# ── LogNormalDistribution ────────────────────────────────────

class TestLogNormal:
    def test_mean(self):
        d = LogNormalDistribution(0.0, 1.0)
        expected = np.exp(0.5)
        np.testing.assert_allclose(d.mean, expected, rtol=1e-10)

    def test_support_positive(self):
        d = LogNormalDistribution(0.0, 1.0)
        lo, hi = d.support
        assert lo > 0
        assert hi > lo

    def test_pdf_zero_at_negative(self):
        d = LogNormalDistribution(0.0, 1.0)
        assert d(-1.0) == 0.0

    def test_expectation_mean(self):
        d = LogNormalDistribution(1.0, 0.5)
        expected = np.exp(1.0 + 0.5 * 0.25)
        np.testing.assert_allclose(d.expectation(lambda x: x), expected, rtol=0.02)


# ── Integration with SpotEvolutionModel ──────────────────────

class TestSpotModelWithDistributions:
    def test_sigma_k_float_backward_compat(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)
        assert model.sigma_k == 0.01
        assert model.sigma_k_sq_expected == pytest.approx(0.0001)
        assert isinstance(model.sigma_k_distribution, DeltaDistribution)

    def test_sigma_k_distribution(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
        dist = GaussianDistribution(mu=0.01, sigma=0.002)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=dist)
        # mean should be mu
        assert model.sigma_k == pytest.approx(0.01)
        # E[sigma_k^2] = mu^2 + sigma^2 for Gaussian
        expected = 0.01 ** 2 + 0.002 ** 2
        np.testing.assert_allclose(model.sigma_k_sq_expected, expected, rtol=0.02)

    def test_sigma_k_setter(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)
        model.sigma_k = 0.02
        assert model.sigma_k == 0.02
        model.sigma_k = GaussianDistribution(0.01, 0.003)
        assert is_distributed(model.sigma_k_distribution)

    def test_kernel_with_sigma_k_distribution(self):
        """AnalyticKernel should work when sigma_k is a distribution."""
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
        dist = GaussianDistribution(mu=0.01, sigma=0.002)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=dist)
        ak = AnalyticKernel(model)
        K = ak.kernel(np.linspace(0, 10, 20))
        assert K.shape == (20,)
        assert float(K[0]) > 0

    def test_to_hparam_uses_mean(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
        dist = UniformDistribution(0.005, 0.015)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=dist)
        hp = model.to_hparam()
        assert hp["sigma_k"] == pytest.approx(0.01)

    def test_theta0_uses_mean(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
        dist = UniformDistribution(0.005, 0.015)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=dist)
        theta = model.theta0
        # sigma_k is the last element
        assert theta[-1] == pytest.approx(0.01)


# ── Envelope with distributed parameters ─────────────────────

class TestEnvelopeDistributions:
    def test_fixed_params_unchanged(self):
        """Float params should work exactly as before."""
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        assert env.lspot == 5.0
        assert env.tau_spot == 1.0
        R = env.R_Gamma(np.array([0.0, 1.0, 3.0]))
        assert R.shape == (3,)

    def test_distributed_lspot(self):
        env = TrapezoidSymmetricEnvelope(
            lspot=GaussianDistribution(mu=5.0, sigma=0.5),
            tau_spot=1.0,
        )
        assert env.lspot == pytest.approx(5.0)
        assert env.tau_spot == 1.0
        R = env.R_Gamma(np.array([0.0, 1.0, 3.0]))
        assert R.shape == (3,)
        assert float(R[0]) > 0

    def test_distributed_tau_spot(self):
        env = TrapezoidSymmetricEnvelope(
            lspot=5.0,
            tau_spot=UniformDistribution(0.5, 1.5),
        )
        assert env.tau_spot == pytest.approx(1.0)
        R = env.R_Gamma(np.array([0.0, 1.0, 3.0]))
        assert R.shape == (3,)
        assert float(R[0]) > 0

    def test_both_distributed(self):
        env = TrapezoidSymmetricEnvelope(
            lspot=GaussianDistribution(mu=5.0, sigma=0.5),
            tau_spot=GaussianDistribution(mu=1.0, sigma=0.2),
        )
        R = env.R_Gamma(np.array([0.0, 1.0, 3.0]))
        assert R.shape == (3,)
        assert float(R[0]) > 0

    def test_marginalized_smoother_than_fixed(self):
        """Marginalized R_Gamma should be smoother (less sharp cutoff)."""
        lag = np.linspace(0, 10, 100)
        env_fixed = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        env_dist = TrapezoidSymmetricEnvelope(
            lspot=GaussianDistribution(mu=5.0, sigma=1.0),
            tau_spot=1.0,
        )
        R_fixed = np.array(env_fixed.R_Gamma(lag))
        R_dist = np.array(env_dist.R_Gamma(lag))
        # At the support boundary of the fixed envelope,
        # the distributed version should still be nonzero
        idx_boundary = np.argmin(np.abs(lag - 7.0))
        assert R_fixed[idx_boundary] == pytest.approx(0.0, abs=1e-8)
        assert R_dist[idx_boundary] > 1e-6

    def test_kernel_support_uses_upper(self):
        env = TrapezoidSymmetricEnvelope(
            lspot=UniformDistribution(3.0, 7.0),
            tau_spot=UniformDistribution(0.5, 1.5),
        )
        ks = env.kernel_support()
        # Should use upper bounds: 7 + 2*1.5 = 10
        assert ks == pytest.approx(10.0)

    def test_distribution_properties(self):
        dist_l = GaussianDistribution(mu=5.0, sigma=0.5)
        dist_t = UniformDistribution(0.5, 1.5)
        env = TrapezoidSymmetricEnvelope(lspot=dist_l, tau_spot=dist_t)
        assert env.lspot_distribution is dist_l
        assert env.tau_spot_distribution is dist_t

    def test_full_model_with_all_distributions(self):
        """Full pipeline: distributed envelope + distributed sigma_k."""
        env = TrapezoidSymmetricEnvelope(
            lspot=GaussianDistribution(mu=5.0, sigma=0.5),
            tau_spot=GaussianDistribution(mu=1.0, sigma=0.2),
        )
        vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
        model = SpotEvolutionModel(
            envelope=env, visibility=vis,
            sigma_k=GaussianDistribution(mu=0.01, sigma=0.002),
        )
        ak = AnalyticKernel(model)
        K = ak.kernel(np.linspace(0, 10, 20))
        assert K.shape == (20,)
        assert float(K[0]) > 0


# ── get_sympy ────────────────────────────────────────────────

class TestDistributionSympy:
    def test_delta_sympy(self):
        pytest.importorskip("sympy")
        d = DeltaDistribution(3.0)
        expr = d.sympy_pdf()
        assert expr is not None

    def test_uniform_sympy(self):
        pytest.importorskip("sympy")
        d = UniformDistribution(1.0, 5.0)
        expr = d.sympy_pdf()
        assert expr is not None

    def test_gaussian_sympy(self):
        pytest.importorskip("sympy")
        d = GaussianDistribution(0.0, 1.0)
        expr = d.sympy_pdf()
        assert expr is not None

    def test_lognormal_sympy(self):
        pytest.importorskip("sympy")
        d = LogNormalDistribution(0.0, 1.0)
        expr = d.sympy_pdf()
        assert expr is not None

    def test_get_sympy_display_false(self):
        pytest.importorskip("sympy")
        d = GaussianDistribution(0.0, 1.0)
        result = d.get_sympy(display=False)
        assert "pdf" in result
        assert result["pdf"] is not None

    def test_get_sympy_display_true(self):
        pytest.importorskip("sympy")
        d = UniformDistribution(1.0, 5.0)
        result = d.get_sympy(var_name="\\ell", display=True)
        assert "pdf" in result

    def test_base_sympy_returns_none(self):
        """Base class sympy_pdf returns None."""
        d = ParameterDistribution()
        assert d.sympy_pdf() is None
