import numpy as np

__all__ = ["compute_psd"]


def _bin_psd(freq, power, n_bins, log_bins):
    """
    Average (freq, power) into ``n_bins`` frequency bins.

    A raw periodogram estimate at each frequency has ~100% fractional
    scatter (it is approximately chi-squared with 2 degrees of
    freedom) regardless of how much data went into it; averaging
    several adjacent estimates together reduces that scatter by
    roughly ``1 / sqrt(points per bin)``.  Power is averaged linearly
    (not in log space), matching Bartlett/Welch-style periodogram
    averaging.  Bins with no points are dropped.
    """
    freq = np.asarray(freq, dtype=float)
    power = np.asarray(power, dtype=float)
    positive = freq > 0
    freq, power = freq[positive], power[positive]

    if log_bins:
        edges = np.logspace(np.log10(freq[0]), np.log10(freq[-1]), n_bins + 1)
    else:
        edges = np.linspace(freq[0], freq[-1], n_bins + 1)

    bin_idx = np.clip(np.digitize(freq, edges) - 1, 0, n_bins - 1)

    freq_out, power_out = [], []
    for i in range(n_bins):
        mask = bin_idx == i
        if not np.any(mask):
            continue
        freq_out.append(np.mean(freq[mask]))
        power_out.append(np.mean(power[mask]))

    return np.array(freq_out), np.array(power_out)


def compute_psd(y, t=None, dt=None,
                normalization="psd",
                freq_min=None, freq_max=None, n_freq=None,
                samples_per_peak=5, n_bins=None, log_bins=True,
                freq_grid=None):
    """
    Compute the Power Spectral Density of a time series using
    astropy.timeseries.LombScargle.

    Works for both evenly and unevenly sampled data.

    Parameters
    ----------
    y : array-like, shape (N,)
        Time series values.
    t : array-like, shape (N,), optional
        Sample times.  If None, integer indices scaled by ``dt`` are used.
    dt : float, optional
        Sampling interval.  Used only when ``t`` is None (default: 1).
    normalization : {"psd", "standard", "model", "log"}
        Passed directly to LombScargle.autopower / power.
    freq_min : float, optional
        Minimum frequency to evaluate.  Ignored when ``freq_grid`` is
        given.
    freq_max : float, optional
        Maximum frequency to evaluate.  Ignored when ``freq_grid`` is
        given.
    n_freq : int, optional
        Number of frequency grid points.  Ignored when ``freq_grid``
        is given.
    samples_per_peak : float, optional
        Controls the frequency grid density (default 5).  Ignored
        when ``freq_grid`` or ``n_freq`` is given.
    freq_grid : array-like, optional
        Evaluate the periodogram at exactly these frequencies instead
        of building a grid internally (e.g. a custom ``np.logspace``
        grid).  Takes priority over ``freq_min``/``freq_max``/
        ``n_freq``/``samples_per_peak``, which are all ignored when
        this is given.
    n_bins : int, optional
        If given, bin the raw periodogram into ``n_bins`` frequency
        bins and average the power within each bin (see
        ``_bin_psd``).  Use this to get a smoothed spectrum (e.g. for
        comparing against published granulation-background fits)
        instead of the raw, highly-scattered periodogram.
    log_bins : bool
        If True (default), bin edges are log-spaced, matching the
        usual log-log presentation of granulation/oscillation power
        spectra; if False, bin edges are linear.  Ignored when
        ``n_bins`` is None.

    Returns
    -------
    freq  : ndarray
        Frequencies in cycles per unit time.
    power : ndarray
        PSD evaluated at each frequency.
    """
    y = np.asarray(y, dtype=float)
    N = len(y)

    # --- build time array if not supplied ---
    if t is None:
        dt = float(dt) if dt is not None else 1.0
        t = np.arange(N, dtype=float) * dt
    else:
        t = np.asarray(t, dtype=float)
        if dt is None:
            dt = float(np.median(np.diff(t)))

    # --- build LombScargle object ---
    from astropy.timeseries import LombScargle
    ls = LombScargle(t, y)

    # --- frequency grid ---
    if freq_grid is not None:
        freq = np.asarray(freq_grid, dtype=float)
        power = ls.power(freq, normalization=normalization)
    elif n_freq is not None:
        f_min = freq_min if freq_min is not None else 1.0 / (t[-1] - t[0])
        f_max = freq_max if freq_max is not None else 0.5 / dt
        freq = np.linspace(f_min, f_max, n_freq)
        power = ls.power(freq, normalization=normalization)
    else:
        freq, power = ls.autopower(
            normalization=normalization,
            minimum_frequency=freq_min,
            maximum_frequency=freq_max,
            samples_per_peak=samples_per_peak,
        )

    if n_bins is not None:
        freq, power = _bin_psd(freq, power, n_bins, log_bins)

    return freq, power
