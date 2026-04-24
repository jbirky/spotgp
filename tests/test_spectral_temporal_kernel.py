"""Tests for SpectralTemporalKernel — spectral-temporal GP kernel K_S(τ, Δv)."""

import numpy as np
import pytest
import jax.numpy as jnp

from spotgp.spectral_temporal_kernel import SpectralTemporalKernel
from spotgp.analytic_kernel import AnalyticKernel
from spotgp.spot_model import SpotEvolutionModel


@pytest.fixture
def dv_grid():
    return np.linspace(-15, 15, 61)


@pytest.fixture
def spectral_kernel(default_hparam, dv_grid):
    return SpectralTemporalKernel(
        default_hparam,
        dv_grid=dv_grid,
        sigma_H=2.0,
        vsini=5.0,
        n_harmonics=3,
        n_lat=32,
        n_theta=128,
        n_v_internal=512,
    )


class TestSpectralTemporalKernelInit:
    def test_from_hparam(self, default_hparam, dv_grid):
        sk = SpectralTemporalKernel(default_hparam, dv_grid, sigma_H=2.0, vsini=5.0)
        assert sk.peq == 10.0
        assert sk.sigma_H == 2.0
        assert sk.vsini == 5.0

    def test_from_spot_model(self, default_hparam, dv_grid):
        model = SpotEvolutionModel.from_hparam(default_hparam)
        sk = SpectralTemporalKernel(model, dv_grid, sigma_H=2.0, vsini=5.0)
        assert sk.peq == 10.0

    def test_custom_sigma_S(self, default_hparam, dv_grid):
        sk = SpectralTemporalKernel(
            default_hparam, dv_grid, sigma_H=2.0, vsini=5.0, sigma_S=0.05)
        assert sk.sigma_S == 0.05

    def test_default_sigma_S_from_model(self, default_hparam, dv_grid):
        sk = SpectralTemporalKernel(
            default_hparam, dv_grid, sigma_H=2.0, vsini=5.0)
        assert sk.sigma_S == default_hparam["sigma_k"]

    def test_precomputed_amplitudes_shape(self, spectral_kernel, dv_grid):
        A = spectral_kernel._A_n_precomputed
        assert A.shape == (32, 4, len(dv_grid))


class TestSpectralTemporalKernelEval:
    def test_kernel_shape(self, spectral_kernel, dv_grid):
        lags = np.linspace(0, 10, 20)
        K = spectral_kernel.kernel(lags)
        assert K.shape == (20, len(dv_grid))

    def test_kernel_zero_lag_positive(self, spectral_kernel):
        K = spectral_kernel.kernel(np.array([0.0]))
        assert K.shape[0] == 1
        assert np.max(K) > 0

    def test_kernel_symmetry_in_lag(self, spectral_kernel):
        """K_S(τ, Δv) = K_S(-τ, Δv) — kernel is even in τ."""
        lags = np.array([1.0, 3.0, 5.0])
        K_pos = spectral_kernel.kernel(lags)
        K_neg = spectral_kernel.kernel(-lags)
        np.testing.assert_allclose(K_pos, K_neg, atol=1e-10)

    def test_kernel_symmetry_in_dv(self, spectral_kernel):
        """K_S(τ, Δv) = K_S(τ, -Δv) — kernel is even in Δv."""
        dv = spectral_kernel.dv_grid
        lags = np.array([0.0, 2.0, 5.0])
        K = spectral_kernel.kernel(lags)
        mid = len(dv) // 2
        K_pos = K[:, mid:]
        K_neg = K[:, :mid + 1][:, ::-1]
        np.testing.assert_allclose(K_pos, K_neg, rtol=1e-4)

    def test_kernel_single_latitude_shape(self, spectral_kernel, dv_grid):
        lags = np.linspace(0, 5, 10)
        K = spectral_kernel.kernel_single_latitude(lags, phi_idx=0)
        assert K.shape == (10, len(dv_grid))


class TestPhotometricRecovery:
    """Integrating K_S over Δv should recover the 1D photometric kernel."""

    def test_photometric_kernel_shape(self, spectral_kernel):
        lags = np.linspace(0, 10, 20)
        K_phot = spectral_kernel.photometric_kernel(lags)
        assert K_phot.shape == (20,)

    def test_photometric_recovery_consistency(self, default_hparam, dv_grid):
        """∫ K_S(τ, Δv) dΔv should be proportional to K(τ) from AnalyticKernel.

        The absolute amplitudes may differ because of the line profile
        normalization, but the temporal *shape* must match.
        """
        sk = SpectralTemporalKernel(
            default_hparam, dv_grid, sigma_H=2.0, vsini=5.0,
            n_harmonics=3, n_lat=64, n_theta=256, n_v_internal=1024,
        )
        ak = AnalyticKernel(default_hparam, n_harmonics=3, n_lat=64)

        lags = np.linspace(0, 15, 30)
        K_phot = sk.photometric_kernel(lags)
        K_analytic = np.array(ak.kernel(lags))

        K_phot_norm = K_phot / K_phot[0]
        K_analytic_norm = K_analytic / K_analytic[0]

        np.testing.assert_allclose(K_phot_norm, K_analytic_norm, atol=0.15)


class TestDerivedQuantities:
    def test_rv_autocovariance_shape(self, spectral_kernel):
        lags = np.linspace(0, 10, 15)
        C_RV = spectral_kernel.rv_autocovariance(lags)
        assert C_RV.shape == (15,)

    def test_rv_autocovariance_zero_lag_positive(self, spectral_kernel):
        C_RV = spectral_kernel.rv_autocovariance(np.array([0.0]))
        assert float(C_RV[0]) > 0

    def test_ccf_covariance_shape(self, spectral_kernel, dv_grid):
        lags = np.linspace(0, 10, 15)
        C_CCF = spectral_kernel.ccf_covariance(lags, sum_w_sq=1.0)
        assert C_CCF.shape == (15, len(dv_grid))

    def test_ccf_covariance_scaling(self, spectral_kernel):
        """CCF covariance should scale linearly with sum_w_sq."""
        lags = np.array([0.0, 2.0, 5.0])
        C1 = spectral_kernel.ccf_covariance(lags, sum_w_sq=1.0)
        C2 = spectral_kernel.ccf_covariance(lags, sum_w_sq=3.0)
        np.testing.assert_allclose(C2, 3.0 * C1, rtol=1e-12)


class TestPositiveSemiDefiniteness:
    def test_zero_lag_slice_nonnegative(self, spectral_kernel):
        """K_S(0, Δv) should be non-negative for all Δv."""
        K0 = spectral_kernel.kernel(np.array([0.0]))
        assert np.all(K0 >= -1e-15)

    def test_covariance_matrix_psd(self, spectral_kernel):
        """The τ-covariance matrix at a fixed Δv should be PSD."""
        lags = np.linspace(0, 10, 15)
        K = spectral_kernel.kernel(lags)
        mid = len(spectral_kernel.dv_grid) // 2
        K_at_dv0 = K[:, mid]
        lag_matrix = np.abs(np.subtract.outer(lags, lags))
        K_full = spectral_kernel.kernel(lag_matrix.ravel())
        K_matrix = K_full[:, mid].reshape(len(lags), len(lags))
        eigvals = np.linalg.eigvalsh(K_matrix)
        assert np.all(eigvals > -1e-10)
