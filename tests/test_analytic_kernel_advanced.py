"""Tests for AnalyticKernel advanced methods: build_jax, PSD, solid_body, asymmetric."""

import numpy as np
import pytest
import jax.numpy as jnp

from spotgp.analytic_kernel import AnalyticKernel
from spotgp.envelope import (
    TrapezoidSymmetricEnvelope,
    TrapezoidAsymmetricEnvelope,
    SkewedGaussianEnvelope,
    ExponentialEnvelope,
)
from spotgp.spot_model import SpotEvolutionModel
from spotgp.visibility import VisibilityFunction, EdgeOnVisibilityFunction


@pytest.fixture
def sym_ak():
    env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
    vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
    model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)
    return AnalyticKernel(model)


class TestBuildJax:
    def test_returns_self(self, sym_ak):
        result = sym_ak.build_jax(n_lag=32)
        assert result is sym_ak

    def test_kernel_works_after_build(self, sym_ak):
        sym_ak.build_jax(n_lag=32)
        K = sym_ak.kernel(jnp.linspace(0, 10, 20))
        assert K.shape == (20,)


class TestAsymmetricKernel:
    def test_kernel_shape(self):
        env = TrapezoidAsymmetricEnvelope(lspot=5.0, tau_em=0.5, tau_dec=1.5)
        vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)
        ak = AnalyticKernel(model)
        K = ak.kernel(jnp.linspace(0, 10, 30))
        assert K.shape == (30,)
        assert float(K[0]) > 0

    def test_psd(self):
        env = TrapezoidAsymmetricEnvelope(lspot=5.0, tau_em=0.5, tau_dec=1.5)
        vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)
        ak = AnalyticKernel(model)
        freq, power = ak.compute_psd(jnp.linspace(0.01, 5, 30))
        assert freq.shape == (30,)
        assert np.all(np.array(power) >= 0)


class TestSkewNormalKernel:
    def test_kernel_shape(self):
        env = SkewedGaussianEnvelope(sigma_sn=2.0, n_sn=-3.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 2)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)
        ak = AnalyticKernel(model)
        K = ak.kernel(jnp.linspace(0, 20, 30))
        assert K.shape == (30,)
        assert float(K[0]) > 0

    def test_psd_uses_gamma_hat_sq(self):
        env = SkewedGaussianEnvelope(sigma_sn=2.0, n_sn=-3.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 2)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)
        ak = AnalyticKernel(model)
        freq, power = ak.compute_psd(jnp.linspace(0.01, 5, 30))
        assert np.all(np.array(power) >= 0)


class TestExponentialKernel:
    def test_kernel_shape(self):
        env = ExponentialEnvelope(tau_spot=2.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 2)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)
        ak = AnalyticKernel(model)
        K = ak.kernel(jnp.linspace(0, 20, 30))
        assert K.shape == (30,)
        assert float(K[0]) > 0

    def test_psd(self):
        env = ExponentialEnvelope(tau_spot=2.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 2)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)
        ak = AnalyticKernel(model)
        freq, power = ak.compute_psd(jnp.linspace(0.01, 5, 30))
        assert np.all(np.array(power) >= 0)


class TestEdgeOnKernel:
    def test_kernel(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = EdgeOnVisibilityFunction(peq=10.0)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)
        ak = AnalyticKernel(model)
        K = ak.kernel(jnp.linspace(0, 20, 30))
        assert K.shape == (30,)
        assert float(K[0]) > 0

    def test_psd(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = EdgeOnVisibilityFunction(peq=10.0)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)
        ak = AnalyticKernel(model)
        freq, power = ak.compute_psd(jnp.linspace(0.01, 5, 30))
        assert np.all(np.array(power) >= 0)


class TestSolidBodyGaussLegendre:
    def test_kernel_solid_body_gl(self, default_hparam):
        lat_range = (-np.pi / 2, np.pi / 2)
        ak = AnalyticKernel(default_hparam, quadrature="gauss-legendre",
                            lat_range=lat_range)
        K = ak.kernel_solid_body(jnp.linspace(0, 10, 20))
        assert K.shape == (20,)
        assert float(K[0]) > 0

    def test_psd_gauss_legendre(self, default_hparam):
        lat_range = (-np.pi / 2, np.pi / 2)
        ak = AnalyticKernel(default_hparam, quadrature="gauss-legendre",
                            lat_range=lat_range)
        freq, power = ak.compute_psd(jnp.linspace(0.01, 5, 30))
        assert freq.shape == (30,)


class TestKernelSingleLatitude:
    def test_shape(self, default_hparam):
        ak = AnalyticKernel(default_hparam)
        K = ak.kernel_single_latitude(jnp.linspace(0, 10, 20), 0.3)
        assert K.shape == (20,)

    def test_positive_at_zero(self, default_hparam):
        ak = AnalyticKernel(default_hparam)
        K0 = float(ak.kernel_single_latitude(jnp.array([0.0]), 0.3)[0])
        assert K0 > 0
