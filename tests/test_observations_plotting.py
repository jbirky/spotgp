"""Tests for TimeSeriesData plotting methods."""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

from spotgp.observations import TimeSeriesData


@pytest.fixture
def ts():
    t = np.linspace(0, 20, 200)
    y = 1.0 + 0.01 * np.sin(2 * np.pi * t / 5.0)
    return TimeSeriesData(t, y, 0.001)


class TestPlot:
    def test_returns_axes(self, ts):
        ax = ts.plot()
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_with_existing_axes(self, ts):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = ts.plot(ax=ax)
        assert result is ax
        plt.close("all")


class TestPlotPSD:
    def test_returns_axes(self, ts):
        ax = ts.plot_psd()
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_loglog_false(self, ts):
        ax = ts.plot_psd(loglog=False)
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_auto_computes_psd(self, ts):
        assert not hasattr(ts, "psd_freq")
        ts.plot_psd()
        assert hasattr(ts, "psd_freq")
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_with_kwargs(self, ts):
        ax = ts.plot_psd(n_freq=50)
        assert len(ts.psd_freq) == 50
        import matplotlib.pyplot as plt
        plt.close("all")


class TestPlotACF:
    def test_returns_axes(self, ts):
        ax = ts.plot_acf()
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_auto_computes_acf(self, ts):
        assert not hasattr(ts, "acf_lags")
        ts.plot_acf()
        assert hasattr(ts, "acf_lags")
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_with_kwargs(self, ts):
        ax = ts.plot_acf(n_bins=30, max_lag=8.0)
        assert len(ts.acf_lags) == 30
        import matplotlib.pyplot as plt
        plt.close("all")
