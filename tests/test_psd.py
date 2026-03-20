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
