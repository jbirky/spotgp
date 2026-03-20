"""Tests for src.analytic_kernel — AnalyticKernel."""

import numpy as np
import pytest
import jax.numpy as jnp

from src.analytic_kernel import AnalyticKernel
from src.envelope import TrapezoidSymmetricEnvelope, TrapezoidAsymmetricEnvelope
from src.spot_model import (
    SpotEvolutionModel, VisibilityFunction, EdgeOnVisibilityFunction,
)


class TestAnalyticKernelInit:
    def test_from_hparam(self, default_hparam):
        ak = AnalyticKernel(default_hparam)
        assert ak.sigma_k == 0.01
        assert ak.peq == 10.0

    def test_from_physical_hparam(self, physical_hparam):
        ak = AnalyticKernel(physical_hparam)
        expected = np.sqrt(5) * (1 - 0.1) * 0.01 / np.pi
        np.testing.assert_allclose(ak.sigma_k, expected)

    def test_from_spot_model(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi / 4)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)
        ak = AnalyticKernel(model)
        assert ak.sigma_k == 0.01
        assert ak.peq == 10.0

    def test_missing_keys_raises(self):
        with pytest.raises(ValueError, match="missing required"):
            AnalyticKernel({"peq": 10.0})

    def test_not_dict_raises(self):
        with pytest.raises(TypeError):
            AnalyticKernel([1, 2, 3])

    def test_asymmetric_envelope(self, asymmetric_hparam):
        ak = AnalyticKernel(asymmetric_hparam)
        assert ak.envelope_type == "trapezoid_asymmetric"


class TestAnalyticKernelEval:
    def test_kernel_shape(self, default_hparam):
        ak = AnalyticKernel(default_hparam)
        lags = jnp.linspace(0, 10, 50)
        K = ak.kernel(lags)
        assert K.shape == (50,)

    def test_kernel_zero_lag_positive(self, default_hparam):
        ak = AnalyticKernel(default_hparam)
        K0 = ak.kernel(jnp.array([0.0]))
        assert float(K0[0]) > 0

    def test_kernel_symmetry(self, default_hparam):
        ak = AnalyticKernel(default_hparam)
        lags = jnp.array([-3.0, -1.0, 1.0, 3.0])
        K = np.array(ak.kernel(lags))
        np.testing.assert_allclose(K[0], K[3], rtol=1e-8)
        np.testing.assert_allclose(K[1], K[2], rtol=1e-8)

    def test_kernel_2d_input(self, default_hparam):
        ak = AnalyticKernel(default_hparam)
        lag_matrix = np.abs(np.subtract.outer(np.arange(5.0), np.arange(5.0)))
        K = ak.kernel(lag_matrix)
        assert K.shape == (5, 5)

    def test_callable(self, default_hparam):
        ak = AnalyticKernel(default_hparam)
        lags = jnp.linspace(0, 5, 10)
        np.testing.assert_allclose(np.array(ak(lags)), np.array(ak.kernel(lags)))

    def test_kernel_solid_body(self, default_hparam):
        ak = AnalyticKernel(default_hparam)
        lags = jnp.linspace(0, 10, 30)
        K = ak.kernel_solid_body(lags)
        assert K.shape == (30,)
        assert float(K[0]) > 0


class TestAnalyticKernelQuadrature:
    def test_gauss_legendre(self, default_hparam):
        lat_range = (-np.pi / 2, np.pi / 2)
        ak = AnalyticKernel(default_hparam, quadrature="gauss-legendre", lat_range=lat_range)
        K = ak.kernel(jnp.array([0.0, 1.0, 2.0]))
        assert K.shape == (3,)
        assert float(K[0]) > 0

    def test_trapezoid_vs_gauss_legendre(self, default_hparam):
        """Both quadrature methods should give similar results."""
        lat_range = (-np.pi / 2, np.pi / 2)
        ak_trap = AnalyticKernel(default_hparam, quadrature="trapezoid", n_lat=128)
        ak_gl = AnalyticKernel(default_hparam, quadrature="gauss-legendre", n_lat=64,
                               lat_range=lat_range)
        lags = jnp.linspace(0, 10, 20)
        K_trap = np.array(ak_trap.kernel(lags))
        K_gl = np.array(ak_gl.kernel(lags))
        np.testing.assert_allclose(K_trap, K_gl, rtol=0.1)

    def test_invalid_quadrature_raises(self, default_hparam):
        with pytest.raises(ValueError, match="Unknown quadrature"):
            AnalyticKernel(default_hparam, quadrature="simpson")


class TestAnalyticKernelEdgeOn:
    def test_edge_on_fast_path(self):
        """EdgeOnVisibilityFunction should use the fast path (no latitude loop)."""
        vis = EdgeOnVisibilityFunction(peq=10.0)
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)
        ak = AnalyticKernel(model)
        K = ak.kernel(jnp.linspace(0, 10, 20))
        assert K.shape == (20,)
        assert float(K[0]) > 0


class TestAnalyticKernelPSD:
    def test_compute_psd(self, default_hparam):
        ak = AnalyticKernel(default_hparam)
        omega = jnp.linspace(0.01, 5, 50)
        freq, power = ak.compute_psd(omega)
        assert freq.shape == (50,)
        assert power.shape == (50,)
        assert np.all(np.array(power) >= 0)

    def test_psd_stored(self, default_hparam):
        ak = AnalyticKernel(default_hparam)
        omega = jnp.linspace(0.01, 5, 50)
        ak.compute_psd(omega)
        assert hasattr(ak, "psd_freq")
        assert hasattr(ak, "psd_power")
        assert len(ak.psd_freq) == 50
