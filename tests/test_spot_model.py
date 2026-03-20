"""Tests for src.spot_model — visibility, latitude distributions, and SpotEvolutionModel."""

import numpy as np
import pytest
import jax.numpy as jnp

from spotgp.spot_model import (
    VisibilityFunction,
    EdgeOnVisibilityFunction,
    LatitudeDistributionFunction,
    SpotEvolutionModel,
    _cn_general_jax,
    _cn_squared_coefficients_jax,
    _gauss_legendre_grid,
)
from spotgp.envelope import TrapezoidSymmetricEnvelope


# =====================================================================
# _cn_general_jax
# =====================================================================

class TestCnGeneralJax:
    def test_c0_pole_on(self):
        """For inc=0 (pole-on), c_0 = cos(inc) * sin(phi) = sin(phi)."""
        c0 = float(_cn_general_jax(0, inc=0.0, phi=np.pi / 4))
        expected = np.cos(0.0) * np.sin(np.pi / 4)
        np.testing.assert_allclose(c0, expected, atol=1e-10)

    def test_cn_zero_for_pole_on(self):
        """For inc=0, c_n=0 for n>=1 (pure DC visibility)."""
        for n in [1, 2, 3]:
            cn = float(_cn_general_jax(n, inc=0.0, phi=np.pi / 4))
            np.testing.assert_allclose(cn, 0.0, atol=1e-10)

    def test_edge_on_nonzero_harmonics(self):
        """For inc=pi/2, higher harmonics should be nonzero at mid-latitudes."""
        c1 = float(_cn_general_jax(1, inc=np.pi / 2, phi=np.pi / 4))
        assert abs(c1) > 0


class TestCnSquaredCoefficients:
    def test_shape(self):
        cn_sq = _cn_squared_coefficients_jax(inc=np.pi / 4, phi=0.3, n_harmonics=3)
        assert cn_sq.shape == (4,)

    def test_non_negative(self):
        cn_sq = np.array(_cn_squared_coefficients_jax(inc=np.pi / 4, phi=0.3, n_harmonics=3))
        assert np.all(cn_sq >= -1e-15)

    def test_different_n_harmonics(self):
        cn2 = _cn_squared_coefficients_jax(inc=np.pi / 4, phi=0.3, n_harmonics=2)
        cn5 = _cn_squared_coefficients_jax(inc=np.pi / 4, phi=0.3, n_harmonics=5)
        assert cn2.shape == (3,)
        assert cn5.shape == (6,)


# =====================================================================
# _gauss_legendre_grid
# =====================================================================

class TestGaussLegendreGrid:
    def test_nodes_in_range(self):
        nodes, weights = _gauss_legendre_grid(16, -1.0, 1.0)
        assert np.all(np.array(nodes) >= -1.0) and np.all(np.array(nodes) <= 1.0)

    def test_weights_sum(self):
        nodes, weights = _gauss_legendre_grid(32, 0.0, np.pi)
        np.testing.assert_allclose(float(jnp.sum(weights)), np.pi, rtol=1e-12)

    def test_integrates_constant(self):
        """Integral of f(x)=1 over [a,b] should equal b-a."""
        nodes, weights = _gauss_legendre_grid(16, 2.0, 5.0)
        integral = float(jnp.sum(weights))
        np.testing.assert_allclose(integral, 3.0, rtol=1e-12)


# =====================================================================
# VisibilityFunction
# =====================================================================

class TestVisibilityFunction:
    def test_init(self):
        vis = VisibilityFunction(peq=10.0, kappa=0.3, inc=np.pi / 3)
        assert vis.peq == 10.0
        assert vis.kappa == 0.3
        assert vis.inc == pytest.approx(np.pi / 3)

    def test_omega0_equator(self):
        vis = VisibilityFunction(peq=10.0, kappa=0.3, inc=np.pi / 3)
        omega = float(vis.omega0(0.0))
        np.testing.assert_allclose(omega, 2 * np.pi / 10.0, rtol=1e-10)

    def test_omega0_differential_rotation(self):
        vis = VisibilityFunction(peq=10.0, kappa=0.3, inc=np.pi / 3)
        omega_eq = float(vis.omega0(0.0))
        omega_pole = float(vis.omega0(np.pi / 2))
        # kappa > 0 means poles rotate slower
        assert omega_pole < omega_eq

    def test_cn_squared_shape(self):
        vis = VisibilityFunction(peq=10.0, kappa=0.3, inc=np.pi / 3)
        cn_sq = vis.cn_squared(0.3, n_harmonics=3)
        assert cn_sq.shape == (4,)

    def test_param_dict(self):
        vis = VisibilityFunction(peq=10.0, kappa=0.3, inc=np.pi / 3)
        pd = vis.param_dict
        assert pd == {"peq": 10.0, "kappa": 0.3, "inc": pytest.approx(np.pi / 3)}

    def test_param_keys(self):
        vis = VisibilityFunction(peq=10.0, kappa=0.3, inc=np.pi / 3)
        assert vis.param_keys == ("peq", "kappa", "inc")


# =====================================================================
# EdgeOnVisibilityFunction
# =====================================================================

class TestEdgeOnVisibilityFunction:
    def test_init(self):
        vis = EdgeOnVisibilityFunction(peq=10.0)
        assert vis.peq == 10.0
        assert vis.kappa == 0.0
        assert vis.inc == pytest.approx(np.pi / 2)

    def test_cn_squared_closed_form(self):
        vis = EdgeOnVisibilityFunction(peq=10.0)
        cn_sq = vis.cn_squared(0.0, n_harmonics=3)
        assert cn_sq.shape == (4,)
        assert np.all(np.array(cn_sq) >= -1e-15)


# =====================================================================
# LatitudeDistributionFunction
# =====================================================================

class TestLatitudeDistributionFunction:
    def test_default_uniform(self):
        lat = LatitudeDistributionFunction()
        assert lat.lat_range == (-np.pi / 2, np.pi / 2)
        # Uniform distribution should return constant
        assert lat(0.0) == lat(0.5)

    def test_callable(self):
        lat = LatitudeDistributionFunction()
        val = lat(0.3)
        assert isinstance(val, (int, float))
        assert val > 0


# =====================================================================
# SpotEvolutionModel
# =====================================================================

class TestSpotEvolutionModel:
    def test_init_from_components(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.3, inc=np.pi / 3)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)
        assert model.peq == 10.0
        assert model.kappa == 0.3
        assert model.sigma_k == 0.01
        assert model.lspot == 5.0
        assert model.tau_spot == 1.0

    def test_param_keys(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.3, inc=np.pi / 3)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)
        assert model.param_keys == ("peq", "kappa", "inc", "lspot", "tau_spot", "sigma_k")

    def test_theta0(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.3, inc=np.pi / 3)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)
        theta = model.theta0
        assert len(theta) == 6
        np.testing.assert_allclose(theta[0], 10.0)  # peq
        np.testing.assert_allclose(theta[5], 0.01)  # sigma_k

    def test_to_hparam(self):
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.3, inc=np.pi / 3)
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)
        hp = model.to_hparam()
        assert hp["peq"] == 10.0
        assert hp["sigma_k"] == 0.01
        assert hp["tau_spot"] == 1.0

    def test_from_hparam(self, default_hparam):
        model = SpotEvolutionModel.from_hparam(default_hparam)
        assert model.peq == 10.0
        assert model.sigma_k == 0.01
        assert model.tau_spot == 1.0

    def test_from_hparam_physical(self, physical_hparam):
        model = SpotEvolutionModel.from_hparam(physical_hparam)
        expected = np.sqrt(5) * (1 - 0.1) * 0.01 / np.pi
        np.testing.assert_allclose(model.sigma_k, expected)

    def test_roundtrip_hparam(self, default_hparam):
        model = SpotEvolutionModel.from_hparam(default_hparam)
        hp = model.to_hparam()
        for key in ("peq", "kappa", "inc", "lspot", "tau_spot", "sigma_k"):
            np.testing.assert_allclose(hp[key], default_hparam[key], rtol=1e-10)

    def test_theta_from_hparam(self, default_hparam):
        model = SpotEvolutionModel.from_hparam(default_hparam)
        theta = model.theta_from_hparam(default_hparam)
        np.testing.assert_allclose(theta, model.theta0)
