"""Tests for spotgp.visibility — FullGeometryVisibilityFunction."""

import numpy as np
import pytest
import jax.numpy as jnp

from spotgp.visibility import (
    VisibilityFunction,
    FullGeometryVisibilityFunction,
    _cn_general_jax,
)


class TestFullGeometryProjectedArea:
    def test_fully_visible_matches_formula(self):
        """In fully-visible regime, A = pi sin^2(alpha) cos(beta)."""
        alpha = 0.1
        beta = 0.3  # well within fully-visible
        A = float(FullGeometryVisibilityFunction.projected_area(alpha, beta))
        expected = np.pi * np.sin(alpha)**2 * np.cos(beta)
        np.testing.assert_allclose(A, expected, rtol=1e-10)

    def test_hidden_is_zero(self):
        """Spot on far side has zero projected area."""
        A = float(FullGeometryVisibilityFunction.projected_area(0.1, np.pi * 0.9))
        assert A == pytest.approx(0.0, abs=1e-12)

    def test_partial_visibility_intermediate(self):
        """Near the limb, area should be between 0 and fully-visible value."""
        alpha = 0.3
        beta = np.pi / 2  # right at the limb
        A = float(FullGeometryVisibilityFunction.projected_area(alpha, beta))
        assert A > 0
        A_full = np.pi * np.sin(alpha)**2 * np.cos(beta)
        assert A > A_full  # partial area exceeds the (negative) cos approximation

    def test_zero_alpha_gives_zero(self):
        A = float(FullGeometryVisibilityFunction.projected_area(0.0, 0.5))
        assert A == pytest.approx(0.0, abs=1e-12)

    def test_vectorized(self):
        alpha = 0.1
        betas = jnp.linspace(0, jnp.pi, 100)
        A = FullGeometryVisibilityFunction.projected_area(alpha, betas)
        assert A.shape == (100,)
        # Should be non-negative everywhere
        assert np.all(np.array(A) >= -1e-12)

    def test_monotone_decrease_fully_visible(self):
        """Area decreases as beta increases in the fully-visible region."""
        alpha = 0.05
        betas = jnp.linspace(0, jnp.pi / 2 - alpha - 0.01, 50)
        A = np.array(FullGeometryVisibilityFunction.projected_area(alpha, betas))
        assert np.all(np.diff(A) <= 1e-10)


class TestFullGeometryCosBeta:
    def test_pole_on(self):
        """inc=0 (pole-on): cos(beta) = sin(phi) regardless of longitude."""
        vis = FullGeometryVisibilityFunction(peq=10.0, kappa=0.0, inc=0.0)
        cb = float(vis.cos_beta(np.pi / 4, 0.0))
        np.testing.assert_allclose(cb, np.sin(np.pi / 4), rtol=1e-10)

    def test_edge_on_equator(self):
        """inc=pi/2, phi=0: cos(beta) = cos(longitude)."""
        vis = FullGeometryVisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 2)
        lon = 0.5
        cb = float(vis.cos_beta(0.0, lon))
        np.testing.assert_allclose(cb, np.cos(lon), rtol=1e-10)


class TestFullGeometryVisibilityProfile:
    def test_shape(self):
        vis = FullGeometryVisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 3)
        lon, A = vis.visibility_profile(np.pi / 6, 0.1)
        assert lon.shape == (512,)
        assert A.shape == (512,)

    def test_custom_n_lon(self):
        vis = FullGeometryVisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 3)
        lon, A = vis.visibility_profile(np.pi / 6, 0.1, n_lon=64)
        assert lon.shape == (64,)

    def test_non_negative(self):
        vis = FullGeometryVisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 3)
        _, A = vis.visibility_profile(np.pi / 6, 0.1)
        assert np.all(np.array(A) >= -1e-12)


class TestFullGeometryCnSquared:
    def test_shape(self):
        vis = FullGeometryVisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 3)
        cn = vis.cn_squared(0.3, n_harmonics=3)
        assert cn.shape == (4,)

    def test_non_negative(self):
        vis = FullGeometryVisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 3)
        cn = np.array(vis.cn_squared(0.3, n_harmonics=3))
        assert np.all(cn >= 0)

    def test_small_spot_matches_base(self):
        """For small alpha_ref, full geometry should match small-spot approx."""
        inc = np.pi / 3
        phi = 0.3
        vis_small = VisibilityFunction(peq=10.0, kappa=0.0, inc=inc)
        vis_full = FullGeometryVisibilityFunction(
            peq=10.0, kappa=0.0, inc=inc, alpha_ref=0.01)
        cn_small = np.array(vis_small.cn_squared(phi, 3))
        cn_full = np.array(vis_full.cn_squared(phi, 3))
        np.testing.assert_allclose(cn_full, cn_small, rtol=0.1)

    def test_large_spot_diverges(self):
        """For large alpha_ref, results should differ from small-spot approx."""
        inc = np.pi / 3
        phi = 0.3
        vis_small = VisibilityFunction(peq=10.0, kappa=0.0, inc=inc)
        vis_full = FullGeometryVisibilityFunction(
            peq=10.0, kappa=0.0, inc=inc, alpha_ref=0.8)
        cn_small = np.array(vis_small.cn_squared(phi, 3))
        cn_full = np.array(vis_full.cn_squared(phi, 3))
        # At least one coefficient should differ by > 10%
        rel_diff = np.abs(cn_full - cn_small) / np.maximum(cn_small, 1e-15)
        assert np.any(rel_diff > 0.1)

    def test_inherits_omega0(self):
        vis = FullGeometryVisibilityFunction(peq=10.0, kappa=0.3, inc=np.pi / 3)
        omega = float(vis.omega0(0.0))
        np.testing.assert_allclose(omega, 2 * np.pi / 10.0, rtol=1e-10)
