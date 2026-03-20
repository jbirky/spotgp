"""Tests for SpotEvolutionModel advanced methods: get_r_gamma_func, repr, asymmetric."""

import numpy as np
import pytest
import jax.numpy as jnp

from spotgp.spot_model import SpotEvolutionModel
from spotgp.visibility import VisibilityFunction
from spotgp.latitude import LatitudeDistributionFunction
from spotgp.envelope import (
    TrapezoidSymmetricEnvelope,
    TrapezoidAsymmetricEnvelope,
    SkewedGaussianEnvelope,
    ExponentialEnvelope,
)


@pytest.fixture
def sym_model():
    env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
    vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
    return SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)


@pytest.fixture
def asym_model():
    env = TrapezoidAsymmetricEnvelope(lspot=5.0, tau_em=0.5, tau_dec=1.5)
    vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
    return SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)


@pytest.fixture
def exp_model():
    env = ExponentialEnvelope(tau_spot=2.0)
    vis = VisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 2)
    return SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)


class TestGetRGammaFunc:
    def test_symmetric_envelope(self, sym_model):
        r_gamma = sym_model.get_r_gamma_func()
        lag = jnp.array([0.0, 1.0, 3.0])
        theta = sym_model.theta0
        R = r_gamma(jnp.array(theta), lag)
        assert R.shape == (3,)
        assert float(R[0]) > 0  # R(0) > 0

    def test_asymmetric_envelope(self, asym_model):
        r_gamma = asym_model.get_r_gamma_func()
        lag = jnp.array([0.0, 1.0, 3.0])
        theta = asym_model.theta0
        R = r_gamma(jnp.array(theta), lag)
        assert R.shape == (3,)
        assert float(R[0]) > 0

    def test_exponential_envelope(self, exp_model):
        r_gamma = exp_model.get_r_gamma_func()
        lag = jnp.array([0.0, 1.0, 5.0])
        theta = exp_model.theta0
        R = r_gamma(jnp.array(theta), lag)
        assert R.shape == (3,)
        assert float(R[0]) > 0
        # Should decay with lag
        assert float(R[2]) < float(R[0])

    def test_no_envelope(self):
        vis = VisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 2)
        model = SpotEvolutionModel(envelope=None, visibility=vis, sigma_k=0.01)
        r_gamma = model.get_r_gamma_func()
        lag = jnp.array([0.0, 5.0, 10.0])
        R = r_gamma(jnp.zeros(1), lag)
        # Should be all ones when no envelope
        np.testing.assert_allclose(np.array(R), 1.0)


class TestBandwidthSupport:
    def test_symmetric(self, sym_model):
        bounds = np.array([[5, 15], [0, 0.5], [0.1, 1.5],
                           [1, 10], [0.1, 3], [0.001, 0.1]])
        bw = sym_model.bandwidth_support(sym_model.param_keys, bounds)
        assert bw > 0
        # Should be lspot_upper + 2 * tau_upper = 10 + 2*3 = 16
        assert bw == pytest.approx(16.0)

    def test_asymmetric(self, asym_model):
        keys = asym_model.param_keys
        bounds = np.array([[5, 15], [0, 0.5], [0.1, 1.5],
                           [1, 10], [0.1, 2], [0.1, 3], [0.001, 0.1]])
        bw = asym_model.bandwidth_support(keys, bounds)
        assert bw > 0

    def test_no_envelope(self):
        vis = VisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 2)
        model = SpotEvolutionModel(envelope=None, visibility=vis, sigma_k=0.01)
        bw = model.bandwidth_support((), np.empty((0, 2)))
        assert bw == 0.0


class TestRepr:
    def test_repr_with_components(self, sym_model):
        r = repr(sym_model)
        assert "SpotEvolutionModel" in r
        assert "TrapezoidSymmetricEnvelope" in r
        assert "VisibilityFunction" in r
        assert "sigma_k" in r

    def test_repr_no_envelope(self):
        vis = VisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 2)
        model = SpotEvolutionModel(envelope=None, visibility=vis, sigma_k=0.01)
        r = repr(model)
        assert "None" in r

    def test_repr_no_visibility(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        model = SpotEvolutionModel(envelope=env, visibility=None, sigma_k=0.01)
        r = repr(model)
        assert "None" in r


class TestAmplitudeInit:
    def test_physical_rate(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 2)
        model = SpotEvolutionModel(
            envelope=env, visibility=vis,
            nspot_rate=1.0, alpha_max=0.1, fspot=0.0)
        expected = np.sqrt(1.0) * 1.0 * 0.01
        np.testing.assert_allclose(model.sigma_k, expected)

    def test_missing_amplitude_raises(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 2)
        with pytest.raises(ValueError, match="sigma_k"):
            SpotEvolutionModel(envelope=env, visibility=vis)


class TestCustomLatitude:
    def test_custom_distribution(self):
        class NarrowBand(LatitudeDistributionFunction):
            @property
            def lat_range(self):
                return (-np.pi / 6, np.pi / 6)

        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 2)
        lat = NarrowBand()
        model = SpotEvolutionModel(
            envelope=env, visibility=vis, sigma_k=0.01,
            latitude_distribution=lat)
        assert model.latitude_distribution.lat_range == (-np.pi / 6, np.pi / 6)

    def test_bad_type_raises(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 2)
        with pytest.raises(TypeError):
            SpotEvolutionModel(
                envelope=env, visibility=vis, sigma_k=0.01,
                latitude_distribution="not a lat dist")
