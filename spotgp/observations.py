"""
observations.py — Time series data container with PSD and ACF computation.
"""

import numpy as np
from .psd import compute_psd

__all__ = ["TimeSeriesData"]


class TimeSeriesData:
    """
    Container for observed time series data.

    Parameters
    ----------
    x : array_like, shape (N,)
        Observation times.
    y : array_like, shape (N,)
        Observed values (e.g. flux).
    yerr : array_like, shape (N,) or float
        Measurement uncertainties. A scalar is broadcast to all points.
    normalize : bool
        If True (default), normalize the flux so that the median is 1
        and scale yerr accordingly.
    """

    def __init__(self, x, y, yerr, normalize=True):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        yerr = np.asarray(yerr, dtype=float)
        yerr = np.broadcast_to(yerr, x.shape).copy()

        if x.shape != y.shape:
            raise ValueError(
                f"x and y must have the same shape, "
                f"got {x.shape} and {y.shape}")
        if x.shape != yerr.shape:
            raise ValueError(
                f"x and yerr must have the same shape, "
                f"got {x.shape} and {yerr.shape}")

        # Mask out non-finite values
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
        self.x = x[mask]
        self.y = y[mask]
        self.yerr = yerr[mask]

        if normalize:
            self.normalize()

    @property
    def N(self) -> int:
        """Number of data points."""
        return len(self.x)

    @property
    def baseline(self) -> float:
        """Total time baseline."""
        return float(self.x[-1] - self.x[0])

    @property
    def median_dt(self) -> float:
        """Median time step."""
        return float(np.median(np.diff(self.x)))

    def normalize(self):
        """
        Normalize the flux so that the median equals 1.

        Divides ``y`` and ``yerr`` by the median of ``y``.  This is
        idempotent: calling it on already-normalized data (median ~ 1)
        has negligible effect.
        """
        median = np.median(self.y)
        if median != 0:
            self.yerr = self.yerr / np.abs(median)
            self.y = self.y / median

    def sigma_clip(self, lower=3.0, upper=3.0):
        """
        Remove outliers beyond a sigma threshold.

        Masks out points where ``y`` falls below
        ``mean(y) - |lower| * std(y)`` or above
        ``mean(y) + |upper| * std(y)``.

        Parameters
        ----------
        lower : float
            Number of standard deviations below the mean (default 3).
        upper : float
            Number of standard deviations above the mean (default 3).
        """
        mean = np.mean(self.y)
        std = np.std(self.y)
        lo = mean - np.abs(lower) * std
        hi = mean + np.abs(upper) * std
        mask = (self.y >= lo) & (self.y <= hi)
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.yerr = self.yerr[mask]

    def compute_psd(self, normalization="psd", freq_min=None, freq_max=None,
                    n_freq=None, samples_per_peak=5):
        """
        Compute the Lomb-Scargle power spectral density.

        Parameters
        ----------
        normalization : str
            LombScargle normalization mode (default "psd").
        freq_min, freq_max : float, optional
            Frequency bounds.
        n_freq : int, optional
            Number of frequency grid points.
        samples_per_peak : float, optional
            Frequency grid density (default 5).

        Returns
        -------
        freq : ndarray
            Frequencies [cycles per unit time].
        power : ndarray
            PSD at each frequency.
        """
        freq, power = compute_psd(
            self.y, t=self.x,
            normalization=normalization,
            freq_min=freq_min, freq_max=freq_max,
            n_freq=n_freq, samples_per_peak=samples_per_peak,
        )
        self.psd_freq = freq
        self.psd_power = power
        return freq, power

    def compute_acf(self, n_bins=None, max_lag=None):
        """
        Compute the empirical autocorrelation function from irregularly
        sampled data using binned lag pairs.

        For each pair of observations (i, j), the lag ``|x_i - x_j|``
        and product ``(y_i - mean)(y_j - mean)`` are accumulated into
        bins.  The ACF is the mean product in each bin, normalized by
        the variance.

        Parameters
        ----------
        n_bins : int, optional
            Number of lag bins (default: N // 2).
        max_lag : float, optional
            Maximum lag to compute (default: half the baseline).

        Returns
        -------
        lag_centers : ndarray, shape (n_bins,)
            Center of each lag bin.
        acf : ndarray, shape (n_bins,)
            Normalized autocorrelation in each bin.
        """
        x, y = self.x, self.y
        N = self.N
        y_centered = y - np.mean(y)
        var = np.var(y)

        if max_lag is None:
            max_lag = self.baseline / 2.0
        if n_bins is None:
            n_bins = max(N // 2, 10)

        bin_edges = np.linspace(0, max_lag, n_bins + 1)
        lag_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        acf_sum = np.zeros(n_bins)
        acf_count = np.zeros(n_bins)

        for i in range(N):
            lags = np.abs(x[i + 1:] - x[i])
            products = y_centered[i] * y_centered[i + 1:]
            bin_idx = np.searchsorted(bin_edges[1:], lags)
            valid = bin_idx < n_bins
            np.add.at(acf_sum, bin_idx[valid], products[valid])
            np.add.at(acf_count, bin_idx[valid], 1)

        nonempty = acf_count > 0
        acf = np.zeros(n_bins)
        acf[nonempty] = acf_sum[nonempty] / (acf_count[nonempty] * var)

        self.acf_lags = lag_centers
        self.acf_values = acf
        self.acf_counts = acf_count.astype(int)
        return lag_centers, acf

    @classmethod
    def from_lightkurve(cls, lc, normalize=True):
        """
        Create a TimeSeriesData from a lightkurve LightCurve object.

        Extracts time, flux, and flux_err arrays, removes NaN entries,
        and converts Astropy quantities to plain floats if needed.

        Parameters
        ----------
        lc : lightkurve.LightCurve
            A LightCurve object (e.g. from ``lk.search_lightcurve(...).download()``).
        normalize : bool
            If True (default), normalize flux to median of 1.

        Returns
        -------
        TimeSeriesData
        """
        # Extract arrays, handling Astropy Quantity objects
        time = np.asarray(lc.time.value, dtype=float)
        flux = np.asarray(lc.flux.value, dtype=float) if hasattr(lc.flux, 'value') else np.asarray(lc.flux, dtype=float)

        if lc.flux_err is not None and np.any(np.isfinite(
                np.asarray(lc.flux_err.value if hasattr(lc.flux_err, 'value') else lc.flux_err))):
            flux_err = np.asarray(
                lc.flux_err.value if hasattr(lc.flux_err, 'value') else lc.flux_err,
                dtype=float)
        else:
            flux_err = np.full_like(flux, np.nanmedian(np.abs(np.diff(flux))))

        # NaN/inf masking is handled by __init__
        return cls(time, flux, flux_err, normalize=normalize)

    def plot(self, ax=None, color="k", alpha=0.6, marker=".", ms=2,
             xlabel="Time", ylabel="Flux", **kwargs):
        """
        Plot the time series.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Axes to plot on. If None, creates a new figure.
        color, alpha, marker, ms : plot style options.
        xlabel, ylabel : str
            Axis labels.
        **kwargs
            Extra keyword arguments passed to ``ax.errorbar``.

        Returns
        -------
        ax : matplotlib Axes
        """
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        ax.errorbar(self.x, self.y, yerr=self.yerr, fmt=marker,
                    color=color, alpha=alpha, ms=ms, elinewidth=0.5, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    def plot_psd(self, ax=None, color="k", lw=1.0, loglog=True,
                 xlabel="Frequency", ylabel="Power", **psd_kwargs):
        """
        Plot the Lomb-Scargle PSD.

        Calls ``compute_psd()`` if it hasn't been run yet (or if
        ``psd_kwargs`` are provided to override previous settings).

        Parameters
        ----------
        ax : matplotlib Axes, optional
        color, lw : plot style options.
        loglog : bool
            If True (default), use log-log axes.
        xlabel, ylabel : str
            Axis labels.
        **psd_kwargs
            Passed to ``compute_psd()``.

        Returns
        -------
        ax : matplotlib Axes
        """
        import matplotlib.pyplot as plt
        if psd_kwargs or not hasattr(self, "psd_freq"):
            self.compute_psd(**psd_kwargs)
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        if loglog:
            ax.loglog(self.psd_freq, self.psd_power, color=color, lw=lw)
        else:
            ax.plot(self.psd_freq, self.psd_power, color=color, lw=lw)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    def plot_acf(self, ax=None, color="k", lw=1.5,
                 xlabel="Time lag", ylabel="ACF", **acf_kwargs):
        """
        Plot the empirical autocorrelation function.

        Calls ``compute_acf()`` if it hasn't been run yet (or if
        ``acf_kwargs`` are provided to override previous settings).

        Parameters
        ----------
        ax : matplotlib Axes, optional
        color, lw : plot style options.
        xlabel, ylabel : str
            Axis labels.
        **acf_kwargs
            Passed to ``compute_acf()``.

        Returns
        -------
        ax : matplotlib Axes
        """
        import matplotlib.pyplot as plt
        if acf_kwargs or not hasattr(self, "acf_lags"):
            self.compute_acf(**acf_kwargs)
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.acf_lags, self.acf_values, color=color, lw=lw)
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    def __repr__(self) -> str:
        return (f"TimeSeriesData(N={self.N}, "
                f"baseline={self.baseline:.2f}, "
                f"median_dt={self.median_dt:.4f})")
