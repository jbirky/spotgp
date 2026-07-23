"""Tests for spectral contrast model and synthetic photometry."""

import numpy as np
import pytest
import jax.numpy as jnp


# =====================================================================
# BlackbodyProvider tests
# =====================================================================

class TestBlackbodyProvider:
    def test_spectrum_shape(self):
        from spotgp.spectral import BlackbodyProvider
        provider = BlackbodyProvider(wl_range=(3000, 10000), n_points=500)
        wl, flux = provider.spectrum(5800.0)
        assert wl.shape == (500,)
        assert flux.shape == (500,)
        assert np.all(flux > 0)

    def test_hotter_star_brighter(self):
        from spotgp.spectral import BlackbodyProvider
        provider = BlackbodyProvider()
        _, flux_hot = provider.spectrum(6000.0)
        _, flux_cool = provider.spectrum(4000.0)
        assert np.sum(flux_hot) > np.sum(flux_cool)

    def test_wavelength_range(self):
        from spotgp.spectral import BlackbodyProvider
        provider = BlackbodyProvider(wl_range=(4000, 9000), n_points=100)
        wl, _ = provider.spectrum(5000.0)
        assert wl[0] == pytest.approx(4000.0)
        assert wl[-1] == pytest.approx(9000.0)


# =====================================================================
# BandpassSet tests
# =====================================================================

class TestBandpassSet:
    @pytest.fixture
    def custom_bandpass_set(self):
        from spotgp.spectral import BandpassSet
        wl1 = np.linspace(5500, 7500, 200)
        tp1 = np.exp(-0.5 * ((wl1 - 6500) / 400) ** 2)
        wl2 = np.linspace(7000, 10000, 200)
        tp2 = np.exp(-0.5 * ((wl2 - 8500) / 500) ** 2)
        return BandpassSet({
            "band_V": {"wavelength": wl1, "throughput": tp1},
            "band_I": {"wavelength": wl2, "throughput": tp2},
        })

    def test_n_bands(self, custom_bandpass_set):
        assert custom_bandpass_set.n_bands == 2

    def test_effective_wavelengths(self, custom_bandpass_set):
        wls = custom_bandpass_set.effective_wavelengths
        assert len(wls) == 2
        assert wls[0] < wls[1]
        assert wls[0] == pytest.approx(6500, abs=100)
        assert wls[1] == pytest.approx(8500, abs=100)

    def test_integrated_flux_positive(self, custom_bandpass_set):
        wl = np.linspace(4000, 11000, 1000)
        flux = np.ones_like(wl)
        f = custom_bandpass_set.integrated_flux(wl, flux, "band_V")
        assert f > 0

    def test_contrast_ratio_equal_spectra(self, custom_bandpass_set):
        wl = np.linspace(4000, 11000, 1000)
        flux = np.ones_like(wl)
        ratio = custom_bandpass_set.contrast_ratio(wl, flux, flux, "band_V")
        assert ratio == pytest.approx(1.0, abs=1e-10)

    def test_contrast_ratio_dimmer_spot(self, custom_bandpass_set):
        wl = np.linspace(4000, 11000, 1000)
        flux_phot = np.ones_like(wl)
        flux_spot = 0.5 * np.ones_like(wl)
        ratio = custom_bandpass_set.contrast_ratio(
            wl, flux_phot, flux_spot, "band_V")
        assert ratio == pytest.approx(0.5, abs=1e-10)


# =====================================================================
# BandpassSet with pyphot (if available)
# =====================================================================

class TestBandpassSetPyphot:
    @pytest.fixture
    def pyphot_bandpass_set(self):
        pytest.importorskip("pyphot")
        from spotgp.spectral import BandpassSet
        return BandpassSet({
            "kepler": "KEPLER_Kp",
            "tess": "TESS",
            "sdss_g": "SDSS_g",
        })

    def test_pyphot_loads(self, pyphot_bandpass_set):
        assert pyphot_bandpass_set.n_bands == 3

    def test_pyphot_effective_wavelengths(self, pyphot_bandpass_set):
        wls = pyphot_bandpass_set.effective_wavelengths
        assert len(wls) == 3
        # SDSS_g < KEPLER < TESS in effective wavelength
        assert wls[2] < wls[0] < wls[1]

    def test_pyphot_integrated_flux(self, pyphot_bandpass_set):
        from spotgp.spectral import BlackbodyProvider
        provider = BlackbodyProvider(wl_range=(3000, 11000))
        wl, flux = provider.spectrum(5800.0)
        f = pyphot_bandpass_set.integrated_flux(wl, flux, "tess")
        assert f > 0


# =====================================================================
# SpectralContrastModel tests
# =====================================================================

class TestSpectralContrastModel:
    @pytest.fixture
    def model(self):
        from spotgp.spectral import BlackbodyProvider, BandpassSet, SpectralContrastModel
        provider = BlackbodyProvider(wl_range=(3000, 11000), n_points=2000)
        wl_kep = np.linspace(4200, 9000, 200)
        tp_kep = np.exp(-0.5 * ((wl_kep - 6400) / 800) ** 2)
        wl_tess = np.linspace(6000, 10500, 200)
        tp_tess = np.exp(-0.5 * ((wl_tess - 7865) / 900) ** 2)
        bps = BandpassSet({
            "kepler": {"wavelength": wl_kep, "throughput": tp_kep},
            "tess": {"wavelength": wl_tess, "throughput": tp_tess},
        })
        return SpectralContrastModel(
            provider, bps, T_phot=5800.0,
            T_spot_grid=np.arange(2500, 5800, 100))

    def test_table_shape(self, model):
        assert model.table.shape == (33, 2)  # (5800-2500)/100 = 33 steps

    def test_f_spot_bounds(self, model):
        """f_spot should be between 0 and 1 for T_spot < T_phot."""
        assert np.all(model.table >= 0)
        assert np.all(model.table <= 1.0)

    def test_f_spot_increases_with_T(self, model):
        """Warmer spots have higher f_spot (less contrast)."""
        for j in range(model.n_bands):
            assert np.all(np.diff(model.table[:, j]) >= 0)

    def test_contrast_factor_shape(self, model):
        wls = jnp.array([6400.0, 7865.0])
        c = model.contrast_factor(wls, 4000.0, 5800.0)
        assert c.shape == (2,)

    def test_contrast_factor_positive(self, model):
        wls = jnp.array([6400.0, 7865.0])
        c = model.contrast_factor(wls, 4000.0, 5800.0)
        assert np.all(np.array(c) > 0)
        assert np.all(np.array(c) < 1)

    def test_bluer_band_higher_contrast(self, model):
        """Bluer band should have higher contrast factor."""
        wls = jnp.array([6400.0, 7865.0])
        c = model.contrast_factor(wls, 4000.0, 5800.0)
        assert float(c[0]) > float(c[1])

    def test_contrast_matches_planck_for_blackbody(self, model):
        """With BlackbodyProvider, spectral model should approximate
        the pure Planck contrast_factor from contrast.py."""
        from spotgp.contrast import contrast_factor as planck_cf
        wls = jnp.array([6400.0, 7865.0])
        c_spectral = np.array(model.contrast_factor(wls, 4000.0, 5800.0))
        c_planck = np.array(planck_cf(wls, 4000.0, 5800.0))
        # Should agree to ~5% (differences from bandpass width vs point eval)
        np.testing.assert_allclose(c_spectral, c_planck, rtol=0.05)

    def test_cache_save_load(self, model, tmp_path):
        from spotgp.spectral import BlackbodyProvider, BandpassSet, SpectralContrastModel
        cache_path = str(tmp_path / "contrast_cache.npz")
        provider = model.provider
        bps = model.bandpass_set

        # Save
        m1 = SpectralContrastModel(
            provider, bps, T_phot=5800.0,
            T_spot_grid=model.T_grid, cache_path=cache_path)

        # Load
        m2 = SpectralContrastModel(
            provider, bps, T_phot=5800.0,
            T_spot_grid=model.T_grid, cache_path=cache_path)

        np.testing.assert_array_equal(m1.table, m2.table)


# =====================================================================
# Integration with MultiBandGPSolver
# =====================================================================

class TestSpectralContrastIntegration:
    @pytest.fixture
    def solver_with_spectral_model(self):
        from spotgp.spectral import BlackbodyProvider, BandpassSet, SpectralContrastModel
        from spotgp.multiband import MultiBandData, MultiBandGPSolver

        rng = np.random.default_rng(42)
        data = MultiBandData({
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

        provider = BlackbodyProvider(wl_range=(3000, 11000), n_points=2000)
        wl_kep = np.linspace(4200, 9000, 200)
        tp_kep = np.exp(-0.5 * ((wl_kep - 6400) / 800) ** 2)
        wl_tess = np.linspace(6000, 10500, 200)
        tp_tess = np.exp(-0.5 * ((wl_tess - 7865) / 900) ** 2)
        bps = BandpassSet({
            "kepler": {"wavelength": wl_kep, "throughput": tp_kep},
            "tess": {"wavelength": wl_tess, "throughput": tp_tess},
        })
        scm = SpectralContrastModel(
            provider, bps, T_phot=5800.0,
            T_spot_grid=np.arange(2500, 5800, 50))

        hparam = dict(
            peq=10.0, kappa=0.2, inc=np.pi / 4,
            lspot=5.0, tau_spot=1.0, sigma_k=0.01,
        )
        gp = MultiBandGPSolver(
            data, hparam, T_phot=5800.0, T_spot_init=4000.0,
            contrast_model=scm, matrix_solver="cholesky_full")
        return gp

    def test_likelihood_finite(self, solver_with_spectral_model):
        gp = solver_with_spectral_model
        ll = gp.log_likelihood_at(gp.theta0)
        assert np.isfinite(ll)

    def test_gradient_finite(self, solver_with_spectral_model):
        gp = solver_with_spectral_model
        grad = gp.grad_log_posterior(gp.theta0)
        assert all(np.isfinite(grad))

    def test_spectral_vs_planck_likelihood(self):
        """Spectral model with BlackbodyProvider should give similar
        likelihood to the default Planck contrast."""
        from spotgp.spectral import BlackbodyProvider, BandpassSet, SpectralContrastModel
        from spotgp.multiband import MultiBandData, MultiBandGPSolver

        rng = np.random.default_rng(42)
        data = MultiBandData({
            "kepler": {
                "x": np.linspace(0, 20, 25),
                "y": 1.0 + 0.005 * rng.standard_normal(25),
                "yerr": np.full(25, 0.001),
                "wavelength": 6400.0,
            },
            "tess": {
                "x": np.linspace(0, 20, 15),
                "y": 1.0 + 0.003 * rng.standard_normal(15),
                "yerr": np.full(15, 0.002),
                "wavelength": 7865.0,
            },
        })

        hparam = dict(
            peq=10.0, kappa=0.2, inc=np.pi / 4,
            lspot=5.0, tau_spot=1.0, sigma_k=0.01,
        )

        # Default Planck contrast
        gp_planck = MultiBandGPSolver(
            data, hparam, T_phot=5800.0, T_spot_init=4000.0,
            matrix_solver="cholesky_full")

        # Spectral model with BlackbodyProvider (should be very similar)
        provider = BlackbodyProvider(wl_range=(3000, 11000), n_points=3000)
        wl_kep = np.linspace(4200, 9000, 300)
        tp_kep = np.exp(-0.5 * ((wl_kep - 6400) / 800) ** 2)
        wl_tess = np.linspace(6000, 10500, 300)
        tp_tess = np.exp(-0.5 * ((wl_tess - 7865) / 900) ** 2)
        bps = BandpassSet({
            "kepler": {"wavelength": wl_kep, "throughput": tp_kep},
            "tess": {"wavelength": wl_tess, "throughput": tp_tess},
        })
        scm = SpectralContrastModel(
            provider, bps, T_phot=5800.0,
            T_spot_grid=np.arange(2500, 5800, 25))

        gp_spectral = MultiBandGPSolver(
            data, hparam, T_phot=5800.0, T_spot_init=4000.0,
            contrast_model=scm, matrix_solver="cholesky_full")

        ll_planck = gp_planck.log_likelihood_at(gp_planck.theta0)
        ll_spectral = gp_spectral.log_likelihood_at(gp_spectral.theta0)

        # Should be close (not exact due to bandpass integration vs point eval)
        np.testing.assert_allclose(ll_planck, ll_spectral, rtol=0.05)
