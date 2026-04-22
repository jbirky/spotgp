"""Tests for multi-band GP solver with chromatic spot contrast."""

import numpy as np
import pytest
import jax.numpy as jnp

from spotgp.contrast import spot_contrast, contrast_factor
from spotgp.multiband import MultiBandData, MultiBandGPSolver


# =====================================================================
# Contrast function tests
# =====================================================================

class TestSpotContrast:
    def test_equal_temperatures(self):
        """f_spot = 1 when T_spot = T_phot (no contrast)."""
        f = spot_contrast(jnp.array(6000.0), 5000.0, 5000.0)
        np.testing.assert_allclose(float(f), 1.0, atol=1e-10)

    def test_contrast_positive(self):
        """f_spot < 1 when T_spot < T_phot."""
        f = spot_contrast(jnp.array(6000.0), 3500.0, 5800.0)
        assert 0.0 < float(f) < 1.0

    def test_rayleigh_jeans_limit(self):
        """f_spot → (T_spot/T_phot) in the long-wavelength limit."""
        T_spot, T_phot = 4000.0, 5800.0
        f = spot_contrast(jnp.array(5e5), T_spot, T_phot)  # 50 microns
        np.testing.assert_allclose(float(f), T_spot / T_phot, rtol=0.02)

    def test_wavelength_dependence(self):
        """Contrast is deeper (f_spot smaller) at shorter wavelengths."""
        T_spot, T_phot = 3500.0, 5800.0
        f_blue = spot_contrast(jnp.array(4000.0), T_spot, T_phot)
        f_red = spot_contrast(jnp.array(8000.0), T_spot, T_phot)
        assert float(f_blue) < float(f_red)

    def test_contrast_factor_complement(self):
        """c(λ) = 1 - f_spot(λ)."""
        lam = jnp.array(6000.0)
        f = spot_contrast(lam, 3800.0, 5500.0)
        c = contrast_factor(lam, 3800.0, 5500.0)
        np.testing.assert_allclose(float(c), 1.0 - float(f), atol=1e-12)

    def test_contrast_factor_limits(self):
        """c → 1 at short λ (spots dark), c → small at long λ."""
        T_spot, T_phot = 3500.0, 5800.0
        c_uv = contrast_factor(jnp.array(2000.0), T_spot, T_phot)
        c_ir = contrast_factor(jnp.array(1e7), T_spot, T_phot)  # 1 mm, deep RJ
        assert float(c_uv) > 0.95
        assert float(c_ir) < 0.45  # RJ limit: c → 1 - T_spot/T_phot ≈ 0.40

    def test_jax_differentiable(self):
        """Contrast is differentiable w.r.t. T_spot."""
        grad_fn = jax.grad(
            lambda T: contrast_factor(jnp.array(6000.0), T, 5800.0))
        g = grad_fn(4000.0)
        assert np.isfinite(float(g))
        # dc/dT_spot < 0: hotter spots → less contrast
        assert float(g) < 0


# =====================================================================
# MultiBandData tests
# =====================================================================

class TestMultiBandData:
    @pytest.fixture
    def two_band_data(self):
        rng = np.random.default_rng(42)
        return {
            "kepler": {
                "x": np.linspace(0, 30, 50),
                "y": 1.0 + 0.005 * rng.standard_normal(50),
                "yerr": np.full(50, 0.001),
                "wavelength": 6400.0,
            },
            "tess": {
                "x": np.linspace(0, 30, 30),
                "y": 1.0 + 0.003 * rng.standard_normal(30),
                "yerr": np.full(30, 0.002),
                "wavelength": 7865.0,
            },
        }

    def test_merge_shape(self, two_band_data):
        mbd = MultiBandData(two_band_data)
        assert mbd.N == 80
        assert mbd.n_bands == 2
        assert len(mbd.band_wavelengths) == 2

    def test_sorted_by_time(self, two_band_data):
        mbd = MultiBandData(two_band_data)
        assert np.all(np.diff(mbd.x) >= 0)

    def test_band_indices(self, two_band_data):
        mbd = MultiBandData(two_band_data)
        assert set(mbd.band_indices) == {0, 1}
        assert np.sum(mbd.band_indices == 0) == 50  # kepler
        assert np.sum(mbd.band_indices == 1) == 30  # tess

    def test_wavelengths(self, two_band_data):
        mbd = MultiBandData(two_band_data)
        np.testing.assert_allclose(mbd.band_wavelengths, [6400.0, 7865.0])

    def test_single_band(self):
        """Single-band MultiBandData degenerates cleanly."""
        data = {"V": {
            "x": np.arange(10.0),
            "y": np.ones(10),
            "yerr": np.full(10, 0.01),
            "wavelength": 5510.0,
        }}
        mbd = MultiBandData(data)
        assert mbd.N == 10
        assert mbd.n_bands == 1


# =====================================================================
# MultiBandGPSolver tests
# =====================================================================

class TestMultiBandGPSolver:
    @pytest.fixture
    def hparam(self):
        return dict(
            peq=10.0, kappa=0.2, inc=np.pi / 4,
            lspot=5.0, tau_spot=1.0, sigma_k=0.01,
        )

    @pytest.fixture
    def two_band_data(self):
        rng = np.random.default_rng(42)
        return MultiBandData({
            "kepler": {
                "x": np.linspace(0, 20, 30),
                "y": 1.0 + 0.005 * rng.standard_normal(30),
                "yerr": np.full(30, 0.001),
                "wavelength": 6400.0,
            },
            "tess": {
                "x": np.linspace(0, 20, 20),
                "y": 1.0 + 0.003 * rng.standard_normal(20),
                "yerr": np.full(20, 0.002),
                "wavelength": 7865.0,
            },
        })

    def test_init(self, two_band_data, hparam):
        gp = MultiBandGPSolver(
            two_band_data, hparam, T_phot=5800.0, T_spot_init=4000.0)
        assert gp.N == 50
        assert "T_spot" in gp.param_keys
        assert gp.n_params == 7  # 6 kernel + T_spot

    def test_param_keys_with_sigma_n(self, two_band_data, hparam):
        gp = MultiBandGPSolver(
            two_band_data, hparam, T_phot=5800.0, T_spot_init=4000.0,
            fit_sigma_n=True)
        assert gp.n_params == 8
        assert gp.param_keys[-1] == "sigma_n"
        assert gp.param_keys[-2] == "T_spot"

    def test_log_likelihood_finite(self, two_band_data, hparam):
        gp = MultiBandGPSolver(
            two_band_data, hparam, T_phot=5800.0, T_spot_init=4000.0)
        ll = gp.log_likelihood_at(gp.theta0)
        assert np.isfinite(ll)

    def test_log_posterior_finite(self, two_band_data, hparam):
        gp = MultiBandGPSolver(
            two_band_data, hparam, T_phot=5800.0, T_spot_init=4000.0)
        lp = float(gp.log_posterior(gp.theta0))
        assert np.isfinite(lp)

    def test_gradient_finite(self, two_band_data, hparam):
        gp = MultiBandGPSolver(
            two_band_data, hparam, T_phot=5800.0, T_spot_init=4000.0)
        grad = gp.grad_log_posterior(gp.theta0)
        assert all(np.isfinite(grad))

    def test_full_vs_banded_solver(self, two_band_data, hparam):
        """Full and banded solvers should agree."""
        gp_banded = MultiBandGPSolver(
            two_band_data, hparam, T_phot=5800.0, T_spot_init=4000.0,
            matrix_solver="cholesky_banded")
        gp_full = MultiBandGPSolver(
            two_band_data, hparam, T_phot=5800.0, T_spot_init=4000.0,
            matrix_solver="cholesky_full")
        ll_banded = gp_banded.log_likelihood_at(gp_banded.theta0)
        ll_full = gp_full.log_likelihood_at(gp_full.theta0)
        np.testing.assert_allclose(ll_banded, ll_full, rtol=1e-4)

    def test_single_band_matches_gpsolver(self, hparam):
        """With one band, multi-band solver should match single-band."""
        rng = np.random.default_rng(42)
        N = 30
        x = np.linspace(0, 20, N)
        y = 1.0 + 0.005 * rng.standard_normal(N)
        yerr = np.full(N, 0.001)

        # Single-band GPSolver
        from spotgp.gp_solver import GPSolver
        gp_single = GPSolver(x, y, yerr, hparam,
                             matrix_solver="cholesky_full")
        ll_single = gp_single.log_likelihood()

        # Multi-band with one band — T_spot chosen so c(λ) = some value,
        # but sigma_k is adjusted so c(λ)*sigma_k matches the single-band.
        # For a fair comparison, set T_spot = T_phot (c=0, no contrast).
        # Actually, when T_spot = T_phot, c(λ) = 0 and the kernel vanishes.
        # Instead, test that the structure is correct by using the full
        # solver and checking the kernel scaling directly.

        # Use a specific T_spot and adjust sigma_k so the effective
        # amplitude matches: sigma_k_eff = sigma_k_geom * c(λ)
        T_phot = 5800.0
        T_spot = 4000.0
        lam = 6400.0  # Angstroms
        c_val = float(contrast_factor(jnp.array(lam), T_spot, T_phot))

        # Multi-band with sigma_k_geom = sigma_k_single / c_val
        hparam_mb = dict(hparam)
        hparam_mb["sigma_k"] = hparam["sigma_k"] / c_val

        mbd = MultiBandData({
            "test": {"x": x, "y": y, "yerr": yerr, "wavelength": lam}
        })
        gp_multi = MultiBandGPSolver(
            mbd, hparam_mb, T_phot=T_phot, T_spot_init=T_spot,
            matrix_solver="cholesky_full")
        ll_multi = gp_multi.log_likelihood_at(gp_multi.theta0)

        np.testing.assert_allclose(ll_multi, ll_single, rtol=5e-3)

    def test_amplitude_ratio(self, two_band_data, hparam):
        """Amplitude ratio between bands depends on T_spot."""
        gp = MultiBandGPSolver(
            two_band_data, hparam, T_phot=5800.0, T_spot_init=4000.0)
        ratio = gp.amplitude_ratio(4000.0, 8000.0)
        assert ratio > 1.0  # bluer band has larger amplitude

    def test_predict_shape(self, two_band_data, hparam):
        gp = MultiBandGPSolver(
            two_band_data, hparam, T_phot=5800.0, T_spot_init=4000.0,
            matrix_solver="cholesky_full")
        xpred = np.linspace(0, 20, 15)
        mu, var = gp.predict(xpred, band_wavelength=6400.0)
        assert mu.shape == (15,)
        assert var.shape == (15,)
        assert all(np.isfinite(mu))
        assert all(var >= 0)

    def test_T_spot_affects_likelihood(self, two_band_data, hparam):
        """Changing T_spot should change the log-likelihood."""
        gp = MultiBandGPSolver(
            two_band_data, hparam, T_phot=5800.0, T_spot_init=4000.0)
        ll1 = gp.log_likelihood_at(gp.theta0)

        theta2 = gp.theta0.at[gp._n_kernel].set(3500.0)
        ll2 = gp.log_likelihood_at(theta2)
        assert ll1 != ll2


# Need jax import for the differentiable test
import jax
