"""Tests for spotgp.observations — TimeSeriesData."""

import numpy as np
import pytest

from spotgp.observations import TimeSeriesData


class TestTimeSeriesDataInit:
    def test_basic(self):
        x = np.linspace(0, 10, 50)
        y = np.ones(50) * 100.0
        yerr = np.full(50, 1.0)
        ts = TimeSeriesData(x, y, yerr, normalize=False)
        assert ts.N == 50
        np.testing.assert_array_equal(ts.x, x)
        np.testing.assert_array_equal(ts.y, y)

    def test_scalar_yerr(self):
        x = np.linspace(0, 10, 20)
        y = np.ones(20)
        ts = TimeSeriesData(x, y, 0.01, normalize=False)
        assert ts.yerr.shape == (20,)
        np.testing.assert_allclose(ts.yerr, 0.01)

    def test_mismatched_shapes_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            TimeSeriesData(np.arange(10), np.arange(5), 0.01)

    def test_mismatched_yerr_raises(self):
        with pytest.raises(ValueError):
            TimeSeriesData(np.arange(10), np.arange(10), np.arange(5))

    def test_masks_nan_in_y(self):
        x = np.arange(10.0)
        y = np.ones(10)
        y[3] = np.nan
        y[7] = np.nan
        ts = TimeSeriesData(x, y, 0.01, normalize=False)
        assert ts.N == 8
        assert np.all(np.isfinite(ts.y))

    def test_masks_nan_in_x(self):
        x = np.arange(10.0)
        x[5] = np.nan
        y = np.ones(10)
        ts = TimeSeriesData(x, y, 0.01, normalize=False)
        assert ts.N == 9

    def test_masks_inf_in_yerr(self):
        x = np.arange(10.0)
        y = np.ones(10)
        yerr = np.full(10, 0.01)
        yerr[2] = np.inf
        ts = TimeSeriesData(x, y, yerr, normalize=False)
        assert ts.N == 9


class TestNormalize:
    def test_default_normalizes(self):
        x = np.linspace(0, 10, 50)
        y = np.ones(50) * 500.0
        yerr = np.full(50, 5.0)
        ts = TimeSeriesData(x, y, yerr, normalize=True)
        np.testing.assert_allclose(np.median(ts.y), 1.0, rtol=1e-10)
        np.testing.assert_allclose(ts.yerr, 5.0 / 500.0, rtol=1e-10)

    def test_normalize_false(self):
        x = np.linspace(0, 10, 50)
        y = np.ones(50) * 500.0
        ts = TimeSeriesData(x, y, 5.0, normalize=False)
        np.testing.assert_allclose(np.median(ts.y), 500.0)

    def test_normalize_method_idempotent(self):
        x = np.linspace(0, 10, 50)
        y = np.ones(50) * 200.0
        ts = TimeSeriesData(x, y, 2.0, normalize=True)
        y_after_first = ts.y.copy()
        ts.normalize()
        np.testing.assert_allclose(ts.y, y_after_first, rtol=1e-10)

    def test_normalize_scales_yerr(self):
        x = np.linspace(0, 10, 50)
        y = np.ones(50) * 1000.0
        yerr = np.full(50, 10.0)
        ts = TimeSeriesData(x, y, yerr, normalize=True)
        np.testing.assert_allclose(ts.yerr, 0.01, rtol=1e-10)


class TestSigmaClip:
    def test_removes_outliers(self):
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 100)
        y = np.ones(100)
        y[50] = 10.0   # high outlier
        y[75] = -8.0   # low outlier
        ts = TimeSeriesData(x, y, 0.01, normalize=False)
        ts.sigma_clip(lower=3.0, upper=3.0)
        assert ts.N == 98
        assert np.all(ts.y < 5.0)
        assert np.all(ts.y > -5.0)

    def test_no_outliers_unchanged(self):
        x = np.linspace(0, 10, 50)
        y = np.ones(50)
        ts = TimeSeriesData(x, y, 0.01, normalize=False)
        ts.sigma_clip()
        assert ts.N == 50

    def test_asymmetric_clip(self):
        x = np.arange(100.0)
        y = np.ones(100)
        y[10] = 100.0   # high outlier
        y[20] = -100.0  # low outlier
        ts = TimeSeriesData(x, y, 0.01, normalize=False)
        ts.sigma_clip(lower=1.0, upper=100.0)
        # Only the low outlier should be removed (lower=1 is tight)
        assert 20.0 not in ts.x
        # High outlier kept (upper=100 is very loose)
        assert 10.0 in ts.x

    def test_arrays_stay_aligned(self):
        x = np.arange(10.0)
        y = np.array([1, 1, 1, 100, 1, 1, 1, 1, 1, 1], dtype=float)
        yerr = np.arange(10.0) * 0.1
        ts = TimeSeriesData(x, y, yerr, normalize=False)
        ts.sigma_clip(upper=2.0)
        # x=3 (the outlier) should be gone
        assert 3.0 not in ts.x
        # Remaining yerr should match the original indices
        assert ts.yerr[0] == pytest.approx(0.0)
        assert ts.yerr[1] == pytest.approx(0.1)


class TestTimeSeriesDataProperties:
    def test_baseline(self):
        ts = TimeSeriesData(np.array([0, 5, 10]), np.ones(3), 0.01)
        assert ts.baseline == 10.0

    def test_median_dt(self):
        ts = TimeSeriesData(np.array([0, 1, 2, 3, 4]), np.ones(5), 0.01)
        assert ts.median_dt == pytest.approx(1.0)

    def test_repr(self):
        ts = TimeSeriesData(np.linspace(0, 10, 50), np.ones(50), 0.01)
        r = repr(ts)
        assert "TimeSeriesData" in r
        assert "N=50" in r


class TestComputePSD:
    def test_returns_freq_power(self):
        rng = np.random.default_rng(42)
        t = np.linspace(0, 20, 200)
        y = 1.0 + 0.01 * np.sin(2 * np.pi * t / 3.0) + 0.001 * rng.standard_normal(200)
        ts = TimeSeriesData(t, y, 0.001)
        freq, power = ts.compute_psd()
        assert len(freq) == len(power)
        assert len(freq) > 0

    def test_stores_results(self):
        t = np.linspace(0, 10, 100)
        y = 1.0 + 0.01 * np.sin(t)
        ts = TimeSeriesData(t, y, 0.001)
        ts.compute_psd()
        assert hasattr(ts, "psd_freq")
        assert hasattr(ts, "psd_power")

    def test_n_freq(self):
        t = np.linspace(0, 10, 100)
        y = 1.0 + 0.01 * np.sin(t)
        ts = TimeSeriesData(t, y, 0.001)
        freq, power = ts.compute_psd(n_freq=50)
        assert len(freq) == 50


class TestComputeACF:
    def test_returns_lags_acf(self):
        rng = np.random.default_rng(42)
        t = np.linspace(0, 50, 500)
        y = 1.0 + 0.01 * np.sin(2 * np.pi * t / 5.0) + 0.001 * rng.standard_normal(500)
        ts = TimeSeriesData(t, y, 0.001)
        lags, acf = ts.compute_acf()
        assert len(lags) == len(acf)
        assert len(lags) > 0

    def test_acf_at_zero_lag_near_one(self):
        rng = np.random.default_rng(42)
        t = np.linspace(0, 50, 500)
        y = 1.0 + 0.01 * np.sin(2 * np.pi * t / 5.0) + 0.0001 * rng.standard_normal(500)
        ts = TimeSeriesData(t, y, 0.001)
        lags, acf = ts.compute_acf(n_bins=100)
        assert acf[0] > 0.5

    def test_custom_n_bins(self):
        t = np.linspace(0, 20, 100)
        y = 1.0 + 0.01 * np.sin(t)
        ts = TimeSeriesData(t, y, 0.001)
        lags, acf = ts.compute_acf(n_bins=30)
        assert len(lags) == 30

    def test_custom_max_lag(self):
        t = np.linspace(0, 20, 100)
        y = 1.0 + 0.01 * np.sin(t)
        ts = TimeSeriesData(t, y, 0.001)
        lags, acf = ts.compute_acf(max_lag=5.0)
        assert lags[-1] < 5.0

    def test_stores_results(self):
        t = np.linspace(0, 20, 100)
        y = 1.0 + 0.01 * np.sin(t)
        ts = TimeSeriesData(t, y, 0.001)
        ts.compute_acf()
        assert hasattr(ts, "acf_lags")
        assert hasattr(ts, "acf_values")
        assert hasattr(ts, "acf_counts")

    def test_periodic_signal_has_periodic_acf(self):
        period = 5.0
        t = np.linspace(0, 50, 1000)
        y = 1.0 + 0.01 * np.sin(2 * np.pi * t / period)
        ts = TimeSeriesData(t, y, 0.0001)
        lags, acf = ts.compute_acf(n_bins=200, max_lag=15.0)
        near_period = np.abs(lags - period) < 1.5
        assert np.any(near_period)
        assert np.max(acf[near_period]) > 0.5


class TestFromLightkurve:
    def _mock_lc(self, time, flux, flux_err=None):
        class MockQuantity:
            def __init__(self, arr):
                self.value = np.asarray(arr, dtype=float)
        class MockLC:
            pass
        lc = MockLC()
        lc.time = MockQuantity(time)
        lc.flux = MockQuantity(flux)
        lc.flux_err = MockQuantity(flux_err) if flux_err is not None else None
        return lc

    def test_basic(self):
        lc = self._mock_lc(np.arange(100.0), np.ones(100) * 1000, np.full(100, 10.0))
        ts = TimeSeriesData.from_lightkurve(lc, normalize=True)
        assert ts.N == 100
        np.testing.assert_allclose(np.median(ts.y), 1.0, rtol=1e-10)
        np.testing.assert_allclose(ts.yerr, 0.01, rtol=1e-10)

    def test_no_normalize(self):
        lc = self._mock_lc(np.arange(100.0), np.ones(100) * 1000, np.full(100, 10.0))
        ts = TimeSeriesData.from_lightkurve(lc, normalize=False)
        np.testing.assert_allclose(ts.y, 1000.0)

    def test_removes_nans(self):
        time = np.arange(50.0)
        flux = np.ones(50) * 100
        flux[10] = np.nan
        flux[20] = np.nan
        flux_err = np.full(50, 1.0)
        lc = self._mock_lc(time, flux, flux_err)
        ts = TimeSeriesData.from_lightkurve(lc)
        assert ts.N == 48
        assert np.all(np.isfinite(ts.y))

    def test_no_flux_err_estimates(self):
        lc = self._mock_lc(
            np.arange(50.0),
            1.0 + 0.01 * np.random.default_rng(0).standard_normal(50))
        ts = TimeSeriesData.from_lightkurve(lc)
        assert ts.N == 50
        assert np.all(ts.yerr > 0)
