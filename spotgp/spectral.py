"""
spectral.py — Model-atmosphere spectral contrast and synthetic photometry.

Extends the blackbody contrast model in ``contrast.py`` with realistic
stellar spectra (via Korg) and bandpass-integrated photometry (via pyphot).

The workflow has two phases:

1. **Precomputation** (slow, ~minutes): synthesize spectra on a grid of
   T_spot values, integrate through bandpasses, and cache the resulting
   contrast table.

2. **Inference** (fast, JAX-traceable): interpolate the cached table at
   arbitrary T_spot using ``jnp.interp``, producing a drop-in replacement
   for ``contrast.contrast_factor``.

Dependencies
------------
- ``pyphot`` (optional): bandpass library and synthetic photometry.
  Install via ``pip install pyphot``.
- ``juliacall`` + ``Korg.jl`` (optional): 1D LTE spectral synthesis.
  Install via ``pip install juliacall juliapkg``, then from Python::

      import juliapkg
      juliapkg.add("Korg", "acafc109-a718-429c-b0e5-afd7f8c7ae46")
      juliapkg.resolve()
"""

import os
import warnings
import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = np

__all__ = [
    "KorgProvider",
    "BandpassSet",
    "SpectralContrastModel",
]


# =====================================================================
# Spectrum provider: Korg (Julia via juliacall)
# =====================================================================

class KorgProvider:
    """
    Spectral synthesis via Korg.jl (called through juliacall).

    Produces absolute surface flux F_λ(T_eff) over a specified wavelength
    range.  The Julia runtime and Korg package are initialized lazily on
    first use (~30-60 s JIT overhead, then fast).

    Parameters
    ----------
    logg : float
        Surface gravity (log10 cgs).
    M_H : float
        Metallicity [metals/H].
    wl_range : tuple of float
        (λ_start, λ_stop) in Angstroms.
    linelist : str
        Built-in linelist name: ``"vald_solar"`` (3000-9000 Å) or
        ``"ges"`` (wider, denser).
    """

    def __init__(self, logg=4.44, M_H=0.0, wl_range=(3000, 11000),
                 linelist="vald_solar"):
        self.logg = logg
        self.M_H = M_H
        self.wl_range = tuple(wl_range)
        self._linelist_name = linelist
        self._Korg = None
        self._linelist = None
        self._cache = {}

    def _init_korg(self):
        if self._Korg is not None:
            return
        try:
            from juliacall import Main as jl
        except ImportError:
            raise ImportError(
                "juliacall is required for KorgProvider. "
                "Install with: pip install juliacall juliapkg\n"
                "Then add Korg: python -c \"import juliapkg; "
                "juliapkg.add('Korg', 'acafc109-a718-429c-b0e5-afd7f8c7ae46'); "
                "juliapkg.resolve()\"")
        jl.seval("using Korg")
        self._Korg = jl.Korg
        if self._linelist_name == "ges":
            self._linelist = self._Korg.get_GES_linelist()
        else:
            self._linelist = self._Korg.get_VALD_solar_linelist()

    def spectrum(self, Teff):
        """
        Compute absolute surface flux spectrum at given Teff.

        Parameters
        ----------
        Teff : float
            Effective temperature [K].

        Returns
        -------
        wavelength : ndarray
            Wavelengths in Angstroms.
        flux : ndarray
            Surface flux in erg/s/cm²/Å (not continuum-normalized).
        """
        Teff = float(Teff)
        if Teff in self._cache:
            return self._cache[Teff]

        self._init_korg()

        wls, flux, cntm = self._Korg.synth(
            Teff=Teff, logg=self.logg, M_H=self.M_H,
            linelist=self._linelist,
            wavelengths=self.wl_range,
            rectify=False)

        wls = np.array(wls, dtype=np.float64)
        flux = np.array(flux, dtype=np.float64)
        self._cache[Teff] = (wls, flux)
        return wls, flux

    def clear_cache(self):
        self._cache.clear()


# =====================================================================
# Spectrum provider: blackbody (no external dependency)
# =====================================================================

class BlackbodyProvider:
    """
    Simple Planck-function spectrum provider (no external dependencies).

    Useful for testing and as a fallback when Korg is not installed.

    Parameters
    ----------
    wl_range : tuple of float
        (λ_start, λ_stop) in Angstroms.
    n_points : int
        Number of wavelength samples.
    """

    _h = 6.62607015e-27    # erg·s
    _c = 2.99792458e10     # cm/s
    _kB = 1.380649e-16     # erg/K

    def __init__(self, wl_range=(3000, 11000), n_points=5000):
        self.wavelength = np.linspace(wl_range[0], wl_range[1], n_points)
        self._lam_cm = self.wavelength * 1e-8

    def spectrum(self, Teff):
        """
        Planck function B_λ(T) in erg/s/cm²/Å/sr, scaled by π to give
        surface flux F_λ = π B_λ.

        Returns
        -------
        wavelength : ndarray
            Wavelengths in Angstroms.
        flux : ndarray
            Surface flux in erg/s/cm²/Å.
        """
        h, c, kB = self._h, self._c, self._kB
        lam = self._lam_cm
        x = h * c / (lam * kB * float(Teff))
        B_lam = (2 * h * c**2 / lam**5) / np.expm1(x)
        flux = np.pi * B_lam * 1e-8  # per Å instead of per cm
        return self.wavelength.copy(), flux


# =====================================================================
# Bandpass integration via pyphot
# =====================================================================

class BandpassSet:
    """
    Collection of photometric bandpasses for synthetic photometry.

    Wraps ``pyphot`` for filter loading and flux integration.  Falls back
    to a simple numpy trapezoidal integrator if pyphot is not installed.

    Parameters
    ----------
    bands : dict
        ``{name: filter_spec}`` where ``filter_spec`` is one of:

        - A string matching a pyphot library name (e.g. ``"TESS"``,
          ``"KEPLER_Kp"``, ``"SDSS_g"``).
        - A dict ``{"wavelength": array, "throughput": array}`` for a
          custom filter curve (wavelength in Angstroms).

    Examples
    --------
    >>> bps = BandpassSet({
    ...     "kepler": "KEPLER_Kp",
    ...     "tess": "TESS",
    ...     "sdss_g": "SDSS_g",
    ... })
    >>> bps.integrated_flux(wave, flux, "tess")
    """

    def __init__(self, bands):
        self._filters = {}
        self._eff_wavelengths = {}
        self._use_pyphot = False

        try:
            import pyphot
            self._lib = pyphot.get_library()
            self._use_pyphot = True
        except ImportError:
            self._lib = None

        for name, spec in bands.items():
            if isinstance(spec, str):
                if not self._use_pyphot:
                    raise ImportError(
                        f"pyphot is required to load filter '{spec}'. "
                        "Install with: pip install pyphot")
                filt = self._lib[spec]
                self._filters[name] = filt
                self._eff_wavelengths[name] = self._extract_wavelength(filt)
            elif isinstance(spec, dict):
                wl = np.asarray(spec["wavelength"], dtype=np.float64)
                tp = np.asarray(spec["throughput"], dtype=np.float64)
                if self._use_pyphot:
                    import pyphot
                    filt = pyphot.Filter(
                        wl, tp, name=name, dtype="photon", unit="Angstrom")
                    self._filters[name] = filt
                    self._eff_wavelengths[name] = self._extract_wavelength(filt)
                else:
                    self._filters[name] = (wl, tp)
                    tp_norm = tp / tp.max() if tp.max() > 0 else tp
                    self._eff_wavelengths[name] = float(
                        np.trapz(wl * tp_norm, wl) / np.trapz(tp_norm, wl))
            else:
                raise ValueError(
                    f"Unrecognized filter spec for '{name}': {type(spec)}")

        self.band_names = list(bands.keys())
        self.n_bands = len(bands)

    @staticmethod
    def _extract_wavelength(filt):
        """Extract effective wavelength as a plain float from a pyphot filter."""
        for attr in ('leff', 'lpivot', 'cl'):
            val = getattr(filt, attr, None)
            if val is not None:
                # pyphot returns astropy Quantities; extract .value
                return float(getattr(val, 'value', val))
        return 0.0

    def effective_wavelength(self, band_name):
        """Effective wavelength of a band in Angstroms."""
        return self._eff_wavelengths[band_name]

    @property
    def effective_wavelengths(self):
        """Array of effective wavelengths for all bands, in order."""
        return np.array([self._eff_wavelengths[n] for n in self.band_names])

    def integrated_flux(self, wavelength, flux, band_name):
        """
        Mean flux density through a bandpass: ∫F·T·dλ / ∫T·dλ.

        Parameters
        ----------
        wavelength : array_like
            Wavelengths in Angstroms.
        flux : array_like
            Flux density (any consistent units).
        band_name : str
            Name of the band (as given in the constructor).

        Returns
        -------
        mean_flux : float
        """
        filt = self._filters[band_name]
        wavelength = np.asarray(wavelength, dtype=np.float64)
        flux = np.asarray(flux, dtype=np.float64)

        if self._use_pyphot and not isinstance(filt, tuple):
            result = filt.get_flux(wavelength, flux)
            return float(getattr(result, 'value', result))
        else:
            wl_bp, tp_bp = filt if isinstance(filt, tuple) else (
                np.asarray(filt.wavelength), np.asarray(filt.transmit))
            T = np.interp(wavelength, wl_bp, tp_bp, left=0.0, right=0.0)
            return float(
                np.trapz(flux * T, wavelength) / np.trapz(T, wavelength))

    def contrast_ratio(self, wavelength, flux_phot, flux_spot, band_name):
        """
        Bandpass-integrated spot contrast: f_spot = <F_spot> / <F_phot>.

        Parameters
        ----------
        wavelength : array_like
            Wavelengths in Angstroms (common to both spectra).
        flux_phot : array_like
            Photosphere flux.
        flux_spot : array_like
            Spot flux.
        band_name : str

        Returns
        -------
        f_spot : float
            Flux ratio (0 = black spot, 1 = no contrast).
        """
        f_phot = self.integrated_flux(wavelength, flux_phot, band_name)
        f_spot = self.integrated_flux(wavelength, flux_spot, band_name)
        if f_phot == 0:
            return 0.0
        return f_spot / f_phot


# =====================================================================
# Precomputed spectral contrast model
# =====================================================================

class SpectralContrastModel:
    """
    Precomputed contrast table for bandpass-integrated spectral models.

    Computes ``f_spot(T_spot, band)`` on a temperature grid using a
    spectrum provider (Korg or blackbody) and a bandpass set (pyphot).
    At inference time, interpolates the cached table with JAX-compatible
    linear interpolation, providing a drop-in replacement for
    ``contrast.contrast_factor``.

    Parameters
    ----------
    provider : KorgProvider or BlackbodyProvider
        Spectrum source.
    bandpass_set : BandpassSet
        Bandpass collection for synthetic photometry.
    T_phot : float
        Photosphere temperature [K].
    T_spot_grid : array_like or None
        Grid of spot temperatures [K] to precompute.  Default is
        ``np.arange(2500, T_phot, 25)``.
    cache_path : str or None
        If given, save/load the precomputed table to/from this ``.npz``
        file.  If the file exists and matches the current configuration,
        the table is loaded from disk instead of recomputed.

    Examples
    --------
    >>> provider = BlackbodyProvider()
    >>> bps = BandpassSet({"kepler": "KEPLER_Kp", "tess": "TESS"})
    >>> scm = SpectralContrastModel(provider, bps, T_phot=5800.0)
    >>> scm.contrast_factor(jnp.array([6400.0, 7865.0]), 4000.0, 5800.0)
    """

    def __init__(self, provider, bandpass_set, T_phot,
                 T_spot_grid=None, cache_path=None):
        self.provider = provider
        self.bandpass_set = bandpass_set
        self.T_phot = float(T_phot)

        if T_spot_grid is None:
            T_spot_grid = np.arange(2500, self.T_phot, 25)
        self._T_grid = np.asarray(T_spot_grid, dtype=np.float64)

        self._band_names = list(bandpass_set.band_names)
        self._band_eff_wl = bandpass_set.effective_wavelengths
        self.n_bands = bandpass_set.n_bands

        loaded = False
        if cache_path is not None and os.path.exists(cache_path):
            loaded = self._load_cache(cache_path)

        if not loaded:
            self._compute_table()
            if cache_path is not None:
                self._save_cache(cache_path)

        # Convert to JAX arrays for tracing
        self._T_grid_jax = jnp.asarray(self._T_grid)
        self._f_table_jax = jnp.asarray(self._f_table)
        self._band_eff_wl_jax = jnp.asarray(self._band_eff_wl)

    def _compute_table(self):
        """Build the f_spot(T_spot, band) table."""
        n_T = len(self._T_grid)
        n_b = self.n_bands
        self._f_table = np.zeros((n_T, n_b), dtype=np.float64)

        # Photosphere spectrum (computed once)
        wl_phot, flux_phot = self.provider.spectrum(self.T_phot)

        for i, T_spot in enumerate(self._T_grid):
            wl_spot, flux_spot = self.provider.spectrum(T_spot)

            # Ensure common wavelength grid
            if not np.array_equal(wl_spot, wl_phot):
                flux_spot = np.interp(wl_phot, wl_spot, flux_spot,
                                      left=0.0, right=0.0)

            for j, bname in enumerate(self._band_names):
                self._f_table[i, j] = self.bandpass_set.contrast_ratio(
                    wl_phot, flux_phot, flux_spot, bname)

    def _save_cache(self, path):
        np.savez(path,
                 T_grid=self._T_grid,
                 f_table=self._f_table,
                 T_phot=np.array([self.T_phot]),
                 band_eff_wl=self._band_eff_wl)

    def _load_cache(self, path):
        try:
            data = np.load(path)
            if (np.allclose(data["T_phot"], self.T_phot)
                    and np.array_equal(data["T_grid"], self._T_grid)
                    and np.allclose(data["band_eff_wl"], self._band_eff_wl)):
                self._f_table = data["f_table"]
                return True
        except Exception:
            pass
        return False

    def match_bands(self, wavelengths):
        """
        Map wavelengths (Angstroms) to band indices via nearest match.

        This should be called once at setup time (not inside JIT-traced
        code) to resolve which precomputed column each observation band
        corresponds to.

        Returns
        -------
        indices : ndarray of int
        """
        wavelengths = np.atleast_1d(np.asarray(wavelengths))
        dists = np.abs(wavelengths[:, None] - np.asarray(self._band_eff_wl))
        return np.argmin(dists, axis=1)

    def f_spot(self, T_spot, band_index):
        """
        Interpolated spot contrast ratio for a single band.

        Parameters
        ----------
        T_spot : float
            Spot temperature [K].
        band_index : int
            Index into the bandpass set.

        Returns
        -------
        f : float
            Flux ratio B(T_spot)/B(T_phot) integrated through the bandpass.
        """
        return jnp.interp(T_spot, self._T_grid_jax,
                          self._f_table_jax[:, band_index])

    def make_contrast_fn(self, band_wavelengths):
        """
        Build a JAX-traceable contrast function for a fixed set of bands.

        The band-to-column mapping is resolved eagerly (not inside JIT),
        and the returned closure interpolates the precomputed table over
        T_spot only.

        Parameters
        ----------
        band_wavelengths : array_like
            Effective wavelengths in Angstroms identifying each band.

        Returns
        -------
        contrast_fn : callable
            ``contrast_fn(band_wavelengths, T_spot, T_phot) -> c_array``
            with the same signature as ``contrast.contrast_factor``.
            Fully JAX-traceable w.r.t. T_spot.
        """
        indices = self.match_bands(band_wavelengths)
        # Gather the table columns for the matched bands: shape (n_T, n_matched)
        sub_table = self._f_table_jax[:, indices]
        T_grid = self._T_grid_jax

        def _contrast_fn(band_wavelengths_unused, T_spot, T_phot_unused=None):
            # Interpolate each band column over T_spot
            def _interp_col(col):
                return 1.0 - jnp.interp(T_spot, T_grid, col)
            # vmap over columns (axis 1 of sub_table transposed)
            return jax.vmap(_interp_col)(sub_table.T)

        return _contrast_fn

    def contrast_factor(self, band_wavelengths, T_spot, T_phot=None):
        """
        Compute contrast factors for given bands and T_spot.

        For use outside of JIT-compiled code (e.g., in predict() or
        amplitude_ratio()).  For JIT contexts, use ``make_contrast_fn``
        to build a traceable closure.

        Parameters
        ----------
        band_wavelengths : array_like
            Effective wavelengths in Angstroms (used to identify bands).
        T_spot : float
            Spot temperature [K].
        T_phot : float or None
            Ignored (T_phot is fixed at construction time).

        Returns
        -------
        c : array_like
            Contrast factors c(λ) = 1 - f_spot(λ) for each band.
        """
        indices = self.match_bands(np.atleast_1d(np.asarray(band_wavelengths)))
        c_vals = np.array([
            1.0 - float(jnp.interp(
                T_spot, self._T_grid_jax, self._f_table_jax[:, idx]))
            for idx in indices
        ])
        return jnp.asarray(c_vals)

    @property
    def table(self):
        """The precomputed f_spot(T_spot, band) table as a numpy array."""
        return np.asarray(self._f_table)

    @property
    def T_grid(self):
        """The T_spot grid used for precomputation."""
        return np.asarray(self._T_grid)

    def summary(self):
        """Print a summary of the precomputed contrast table."""
        print(f"SpectralContrastModel: T_phot = {self.T_phot:.0f} K")
        print(f"  T_spot grid: {self._T_grid[0]:.0f} -- "
              f"{self._T_grid[-1]:.0f} K ({len(self._T_grid)} points)")
        print(f"  Bands ({self.n_bands}):")
        for j, bname in enumerate(self._band_names):
            wl = self._band_eff_wl[j]
            f_lo = self._f_table[0, j]
            f_hi = self._f_table[-1, j]
            print(f"    {bname:<15s} λ_eff={wl:8.1f} Å  "
                  f"f_spot=[{f_lo:.4f}, {f_hi:.4f}]")
