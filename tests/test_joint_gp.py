"""Tests for joint_gp.py — joint photometric-spectroscopic GP."""

import numpy as np
import pytest

from spotgp.joint_gp import JointCovarianceBuilder
from spotgp.spectral_temporal_kernel import SpectralTemporalKernel
from spotgp.analytic_kernel import AnalyticKernel


@pytest.fixture
def joint_setup(default_hparam):
    """Create kernel instances and data for joint GP tests."""
    dv_grid = np.linspace(-10, 10, 21)
    sk = SpectralTemporalKernel(
        default_hparam, dv_grid, sigma_H=2.0, vsini=5.0,
        n_harmonics=2, n_lat=16, n_theta=64, n_v_internal=256,
    )
    ak = AnalyticKernel(default_hparam, n_harmonics=2, n_lat=16)
    builder = JointCovarianceBuilder(sk, ak)
    t_phot = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    t_spec = np.array([0.5, 2.5, 4.5])
    return builder, sk, ak, t_phot, t_spec, dv_grid


class TestPhotPhotBlock:
    def test_shape(self, joint_setup):
        builder, _, _, t_phot, _, _ = joint_setup
        K_pp = builder.build_phot_phot(t_phot)
        assert K_pp.shape == (5, 5)

    def test_symmetric(self, joint_setup):
        builder, _, _, t_phot, _, _ = joint_setup
        K_pp = builder.build_phot_phot(t_phot)
        np.testing.assert_allclose(K_pp, K_pp.T, atol=1e-12)

    def test_positive_diagonal(self, joint_setup):
        builder, _, _, t_phot, _, _ = joint_setup
        K_pp = builder.build_phot_phot(t_phot)
        assert np.all(np.diag(K_pp) > 0)

    def test_noise_added(self, joint_setup):
        builder, _, _, t_phot, _, _ = joint_setup
        sigma = np.full(5, 0.001)
        K_pp = builder.build_phot_phot(t_phot, sigma_phot=sigma)
        K_pp_clean = builder.build_phot_phot(t_phot)
        diff = np.diag(K_pp) - np.diag(K_pp_clean)
        np.testing.assert_allclose(diff, 0.001 ** 2, rtol=1e-10)


class TestSpecSpecBlock:
    def test_shape(self, joint_setup):
        builder, _, _, _, t_spec, dv_grid = joint_setup
        K_ss = builder.build_spec_spec(t_spec)
        N_v = len(dv_grid)
        assert K_ss.shape == (3 * N_v, 3 * N_v)

    def test_symmetric(self, joint_setup):
        builder, _, _, _, t_spec, _ = joint_setup
        K_ss = builder.build_spec_spec(t_spec)
        np.testing.assert_allclose(K_ss, K_ss.T, atol=1e-12)

    def test_subset_v_indices(self, joint_setup):
        builder, _, _, _, t_spec, _ = joint_setup
        v_idx = np.array([5, 10, 15])
        K_ss = builder.build_spec_spec(t_spec, v_indices=v_idx)
        assert K_ss.shape == (3 * 3, 3 * 3)


class TestPhotSpecBlock:
    def test_shape(self, joint_setup):
        builder, _, _, t_phot, t_spec, dv_grid = joint_setup
        K_ps = builder.build_phot_spec(t_phot, t_spec)
        N_v = len(dv_grid)
        assert K_ps.shape == (5, 3 * N_v)


class TestJointCovariance:
    def test_shape(self, joint_setup):
        builder, _, _, t_phot, t_spec, dv_grid = joint_setup
        N_v = len(dv_grid)
        K = builder.build_joint_covariance(t_phot, t_spec)
        total = 5 + 3 * N_v
        assert K.shape == (total, total)

    def test_symmetric(self, joint_setup):
        builder, _, _, t_phot, t_spec, _ = joint_setup
        K = builder.build_joint_covariance(t_phot, t_spec)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_psd_with_noise(self, joint_setup):
        """Joint covariance + sufficient noise should be PSD.

        The cross-covariance uses a leading-order approximation that may
        slightly violate the Schur complement condition, requiring noise
        or jitter to regularize.
        """
        builder, _, _, t_phot, t_spec, dv_grid = joint_setup
        sigma_phot = np.full(5, 0.05)
        sigma_spec = 0.05
        K = builder.build_joint_covariance(
            t_phot, t_spec,
            sigma_phot=sigma_phot, sigma_spec=sigma_spec,
            jitter=1e-3,
        )
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals > -1e-6)


class TestJointLogLikelihood:
    def test_finite(self, joint_setup):
        builder, _, _, t_phot, t_spec, dv_grid = joint_setup
        N_v = len(dv_grid)
        rng = np.random.default_rng(42)

        y_phot = 1.0 + 0.001 * rng.standard_normal(5)
        yerr_phot = np.full(5, 0.05)
        y_spec = 0.001 * rng.standard_normal(3 * N_v)
        yerr_spec = 0.05

        logL = builder.joint_log_likelihood(
            t_phot, y_phot, yerr_phot,
            t_spec, y_spec, yerr_spec,
        )
        assert np.isfinite(logL)

    def test_worse_with_larger_noise(self, joint_setup):
        """Higher noise should give a less negative log-likelihood."""
        builder, _, _, t_phot, t_spec, dv_grid = joint_setup
        N_v = len(dv_grid)
        rng = np.random.default_rng(42)

        y_phot = 1.0 + 0.001 * rng.standard_normal(5)
        y_spec = 0.001 * rng.standard_normal(3 * N_v)

        logL_small = builder.joint_log_likelihood(
            t_phot, y_phot, np.full(5, 0.001),
            t_spec, y_spec, 0.001,
        )
        logL_large = builder.joint_log_likelihood(
            t_phot, y_phot, np.full(5, 0.1),
            t_spec, y_spec, 0.1,
        )
        assert logL_large > logL_small
