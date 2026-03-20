"""Tests for src.numerical_kernel — NumericalKernel and helpers."""

import numpy as np
import pytest

from spotgp.numerical_kernel import NumericalKernel, generate_sims, avg_covariance_tlag


class TestGenerateSims:
    def test_output_shape(self):
        np.random.seed(42)
        theta = (10.0, 0.0, np.pi / 2, 3)
        fluxes = generate_sims(theta, nsim=5, tsim=10, tsamp=0.5)
        assert fluxes.shape[0] == 5
        assert fluxes.shape[1] == len(np.arange(0, 10, 0.5))

    def test_fluxes_near_one(self):
        np.random.seed(42)
        theta = (10.0, 0.0, np.pi / 2, 2)
        fluxes = generate_sims(theta, nsim=3, tsim=10, tsamp=0.5, alpha_max=0.01)
        assert np.all(np.abs(fluxes - 1.0) < 0.5)


class TestAvgCovarianceTlag:
    def test_shape(self):
        K = np.eye(10)
        acf = avg_covariance_tlag(K)
        assert len(acf) == 10

    def test_identity_matrix(self):
        K = np.eye(10)
        acf = avg_covariance_tlag(K)
        # Diagonal average should be 1.0
        np.testing.assert_allclose(acf[0], 1.0)
        # Off-diagonals of identity are 0
        np.testing.assert_allclose(acf[1], 0.0)


class TestNumericalKernel:
    def test_init(self, default_hparam):
        np.random.seed(42)
        hp = dict(default_hparam)
        hp["nspot"] = 3
        hp["alpha_max"] = 0.1
        nk = NumericalKernel(hp, tsim=10, tsamp=0.5, nsim=10, verbose=False)
        assert hasattr(nk, "autocor")
        assert hasattr(nk, "kernel")

    def test_requires_nspot(self, default_hparam):
        with pytest.raises(ValueError, match="nspot"):
            NumericalKernel(default_hparam, nsim=5, verbose=False)

    def test_get_acf(self, default_hparam):
        np.random.seed(42)
        hp = dict(default_hparam)
        hp["nspot"] = 3
        hp["alpha_max"] = 0.1
        nk = NumericalKernel(hp, tsim=10, tsamp=0.5, nsim=10, verbose=False)
        tarr, autocor = nk.get_acf()
        assert len(tarr) == len(autocor)
        # Autocorrelation at zero lag should be 1.0
        np.testing.assert_allclose(autocor[0], 1.0)
