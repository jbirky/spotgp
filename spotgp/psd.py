from astropy.timeseries import LombScargle
import numpy as np

__all__ = ["compute_psd"]


def compute_psd(y, t=None, dt=None,
                normalization="psd",
                freq_min=None, freq_max=None, n_freq=None,
                samples_per_peak=5):
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
        Minimum frequency to evaluate.
    freq_max : float, optional
        Maximum frequency to evaluate.
    n_freq : int, optional
        Number of frequency grid points.
    samples_per_peak : float, optional
        Controls the frequency grid density (default 5).

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
    ls = LombScargle(t, y)

    # --- frequency grid ---
    if n_freq is not None:
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

    return freq, power
