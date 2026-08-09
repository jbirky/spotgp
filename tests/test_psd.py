"""Tests for src.psd — Lomb-Scargle PSD computation."""

import numpy as np
import pytest

from spotgp.psd import compute_psd


class TestComputePSD:
    def test_basic_with_time_array(self):
        rng = np.random.default_rng(0)
        t = np.linspace(0, 10, 100)
        y = np.sin(2 * np.pi * t / 2.0) + 0.1 * rng.standard_normal(100)
        freq, power = compute_psd(y, t=t)
        assert len(freq) == len(power)
        assert len(freq) > 0

    def test_with_dt(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(100)
        freq, power = compute_psd(y, dt=0.5)
        assert len(freq) > 0

    def test_with_n_freq(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(100)
        freq, power = compute_psd(y, dt=1.0, n_freq=50)
        assert len(freq) == 50

    def test_peak_near_injected_frequency(self):
        """PSD should peak near the frequency of a strong sinusoidal signal."""
        rng = np.random.default_rng(0)
        f_inject = 0.5  # cycles/day
        t = np.linspace(0, 50, 500)
        y = np.sin(2 * np.pi * f_inject * t) + 0.01 * rng.standard_normal(500)
        freq, power = compute_psd(y, t=t, freq_max=2.0)
        peak_freq = freq[np.argmax(power)]
        np.testing.assert_allclose(peak_freq, f_inject, atol=0.05)

    def test_freq_limits(self):
        rng = np.random.default_rng(0)
        t = np.linspace(0, 20, 200)
        y = rng.standard_normal(200)
        freq, power = compute_psd(y, t=t, freq_min=0.1, freq_max=1.0)
        assert freq[0] >= 0.1
        assert freq[-1] <= 1.0

    def test_defaults_to_dt_one_without_time(self):
        """When neither t nor dt is given, dt defaults to 1."""
        rng = np.random.default_rng(0)
        y = rng.standard_normal(50)
        freq, power = compute_psd(y)
        assert len(freq) > 0

    def test_n_bins_reduces_point_count(self):
        rng = np.random.default_rng(0)
        t = np.linspace(0, 100, 5000)
        y = rng.standard_normal(5000)
        freq, power = compute_psd(y, t=t, n_bins=40)
        assert len(freq) <= 40
        assert len(freq) == len(power)

    def test_n_bins_reduces_scatter(self):
        """Binning should reduce the raw periodogram's point-to-point scatter."""
        rng = np.random.default_rng(0)
        t = np.linspace(0, 100, 5000)
        y = rng.standard_normal(5000)
        freq_raw, power_raw = compute_psd(y, t=t)
        freq_bin, power_bin = compute_psd(y, t=t, n_bins=40)
        rel_std_raw = np.std(power_raw) / np.mean(power_raw)
        rel_std_bin = np.std(power_bin) / np.mean(power_bin)
        assert rel_std_bin < rel_std_raw

    def test_n_bins_linear_spacing(self):
        rng = np.random.default_rng(0)
        t = np.linspace(0, 50, 500)
        y = rng.standard_normal(500)
        freq, power = compute_psd(y, t=t, n_bins=20, log_bins=False)
        assert len(freq) <= 20
        diffs = np.diff(freq)
        # linear bins should have roughly uniform spacing (unlike log bins)
        assert np.std(diffs) / np.mean(diffs) < 0.5

    def test_freq_grid_used_exactly(self):
        rng = np.random.default_rng(0)
        t = np.linspace(0, 100, 3000)
        y = rng.standard_normal(3000)
        freq_grid = np.logspace(-2, 1, 200)
        freq, power = compute_psd(y, t=t, freq_grid=freq_grid)
        np.testing.assert_array_equal(freq, freq_grid)
        assert len(power) == len(freq_grid)

    def test_freq_grid_overrides_other_grid_args(self):
        """freq_grid takes priority over n_freq/freq_min/freq_max."""
        rng = np.random.default_rng(0)
        t = np.linspace(0, 100, 3000)
        y = rng.standard_normal(3000)
        freq_grid = np.logspace(-2, 1, 150)
        freq, power = compute_psd(
            y, t=t, freq_grid=freq_grid,
            n_freq=999, freq_min=50.0, freq_max=60.0)
        np.testing.assert_array_equal(freq, freq_grid)
