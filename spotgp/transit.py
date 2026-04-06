"""
transit.py — Keplerian orbit and transit light curve computation.

Implements a Keplerian orbit model for binary or planetary systems,
following the parameterization of the ``exoplanet`` package
(https://docs.exoplanet.codes).  All heavy computation uses JAX for
GPU/CPU acceleration and JIT compilation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

jax.config.update("jax_enable_x64", True)

try:
    from .lightcurve import LightcurveModel
    from .gp_solver import GPSolver
except ImportError:
    from lightcurve import LightcurveModel
    from gp_solver import GPSolver

__all__ = [
    "KeplerianOrbit",
    "QuadLimbDarkLightCurve",
    "SpotTransitModel",
]


# =====================================================================
# Kepler's equation solver
# =====================================================================

@jax.jit
def _kepler_starter(M, ecc):
    """Markley (1995) starter for Kepler's equation."""
    E0 = M + ecc * jnp.sin(M) / (1.0 - jnp.sin(M) + jnp.sin(M + ecc))
    return E0


@jax.jit
def _kepler_refine(E, M, ecc):
    """One Newton-Raphson refinement step."""
    f = E - ecc * jnp.sin(E) - M
    fp = 1.0 - ecc * jnp.cos(E)
    return E - f / fp


@jax.jit
def _solve_kepler(M, ecc):
    """Solve Kepler's equation  M = E - e sin(E)  for eccentric anomaly E.

    Uses a starter plus several Newton-Raphson iterations, all
    implemented with pure JAX operations so the solver is JIT-compatible
    and differentiable.

    Parameters
    ----------
    M : array_like
        Mean anomaly [rad].
    ecc : float
        Eccentricity (0 <= ecc < 1).

    Returns
    -------
    E : array_like
        Eccentric anomaly [rad].
    """
    # Starter
    E = _kepler_starter(M, ecc)
    # Newton iterations (unrolled for JIT compatibility)
    E = _kepler_refine(E, M, ecc)
    E = _kepler_refine(E, M, ecc)
    E = _kepler_refine(E, M, ecc)
    E = _kepler_refine(E, M, ecc)
    E = _kepler_refine(E, M, ecc)
    return E


# =====================================================================
# True anomaly from eccentric anomaly
# =====================================================================

@jax.jit
def _E_to_true_anomaly(E, ecc):
    """Convert eccentric anomaly to true anomaly.

    Parameters
    ----------
    E : array_like
        Eccentric anomaly [rad].
    ecc : float
        Eccentricity.

    Returns
    -------
    f : array_like
        True anomaly [rad].
    """
    half_angle = 2.0 * jnp.arctan2(
        jnp.sqrt(1.0 + ecc) * jnp.sin(E / 2.0),
        jnp.sqrt(1.0 - ecc) * jnp.cos(E / 2.0),
    )
    return half_angle


# =====================================================================
# KeplerianOrbit
# =====================================================================

class KeplerianOrbit:
    """Keplerian orbit for a binary or planetary system.

    This class computes the 3-D position of the secondary body relative
    to the primary (host star) as a function of time.  It also provides
    the projected sky-plane separation needed for transit light curve
    evaluation.

    The parameterization follows the ``exoplanet`` package convention:
    the reference direction is along the line of sight, with the sky
    plane defined by (x, y) and the z-axis pointing *toward* the
    observer.  A transit occurs when z > 0 and the sky-plane separation
    is less than the sum of the stellar and companion radii.

    Parameters
    ----------
    period : float
        Orbital period [days].
    semi_major_axis : float or None
        Semi-major axis in units of stellar radii.  If ``None``, it is
        computed from ``period`` and ``stellar_mass`` via Kepler's third
        law.
    ecc : float
        Orbital eccentricity (default 0).
    incl : float or None
        Orbital inclination [rad].  If ``None``, it is computed from
        ``impact_param``.
    impact_param : float or None
        Impact parameter (0 = central transit).  Ignored when ``incl``
        is provided directly.
    omega : float
        Argument of periastron of the *secondary* [rad] (default 0).
    Omega : float
        Longitude of the ascending node [rad] (default 0).
    t_periastron : float or None
        Time of periastron passage [days].  Exactly one of
        ``t_periastron`` and ``t0`` must be given.
    t0 : float or None
        Reference transit mid-time [days].  If provided instead of
        ``t_periastron``, the periastron time is derived so that transit
        center falls at ``t0``.
    radius_ratio : float
        Companion-to-star radius ratio  Rp / R*  (default 0.1).
    stellar_mass : float
        Stellar mass in solar masses (default 1.0).  Used only when
        ``semi_major_axis`` is ``None``.
    stellar_radius : float
        Stellar radius in solar radii (default 1.0).  Used for
        converting semi-major axis to physical units.

    Notes
    -----
    Angles follow the standard celestial mechanics convention:

    * ``omega`` is the argument of periastron of the *orbiting body*.
    * ``incl = pi/2`` corresponds to an edge-on orbit.
    """

    # Gravitational constant * solar mass  [R_sun^3 / (M_sun * day^2)]
    _G_MSUN_RSUN3_DAY2 = 2942.2062

    def __init__(
        self,
        period,
        semi_major_axis=None,
        ecc=0.0,
        incl=None,
        impact_param=None,
        omega=0.0,
        Omega=0.0,
        t_periastron=None,
        t0=None,
        radius_ratio=0.1,
        stellar_mass=1.0,
        stellar_radius=1.0,
    ):
        self.period = float(period)
        self.ecc = float(ecc)
        self.omega = float(omega)
        self.Omega = float(Omega)
        self.radius_ratio = float(radius_ratio)
        self.stellar_mass = float(stellar_mass)
        self.stellar_radius = float(stellar_radius)

        # Semi-major axis --------------------------------------------------
        if semi_major_axis is not None:
            self.semi_major_axis = float(semi_major_axis)
        else:
            # Kepler's third law:  a^3 = G M P^2 / (4 pi^2)
            a_cubed = (
                self._G_MSUN_RSUN3_DAY2
                * self.stellar_mass
                * self.period ** 2
                / (4.0 * np.pi ** 2)
            )
            self.semi_major_axis = float(a_cubed ** (1.0 / 3.0))
        # Store in units of stellar radii
        self.a_over_Rstar = self.semi_major_axis / self.stellar_radius

        # Inclination ------------------------------------------------------
        if incl is not None:
            self.incl = float(incl)
            self.impact_param = float(
                self.a_over_Rstar
                * np.cos(self.incl)
                * (1.0 - self.ecc ** 2)
                / (1.0 + self.ecc * np.sin(self.omega))
            )
        elif impact_param is not None:
            self.impact_param = float(impact_param)
            cos_incl = (
                self.impact_param
                * (1.0 + self.ecc * np.sin(self.omega))
                / (self.a_over_Rstar * (1.0 - self.ecc ** 2))
            )
            cos_incl = np.clip(cos_incl, -1.0, 1.0)
            self.incl = float(np.arccos(cos_incl))
        else:
            # Default to edge-on
            self.incl = 0.5 * np.pi
            self.impact_param = 0.0

        # Reference time ---------------------------------------------------
        if t_periastron is not None and t0 is not None:
            raise ValueError(
                "Specify exactly one of t_periastron and t0, not both."
            )
        if t_periastron is not None:
            self.t_periastron = float(t_periastron)
            self.t0 = self._t_periastron_to_t0(self.t_periastron)
        elif t0 is not None:
            self.t0 = float(t0)
            self.t_periastron = self._t0_to_t_periastron(self.t0)
        else:
            self.t0 = 0.0
            self.t_periastron = self._t0_to_t_periastron(self.t0)

    # -----------------------------------------------------------------
    # Time reference conversions
    # -----------------------------------------------------------------

    def _true_anomaly_at_transit(self):
        """True anomaly at inferior conjunction (transit center)."""
        return 0.5 * np.pi - self.omega

    def _t_periastron_to_t0(self, t_peri):
        """Compute transit mid-time from time of periastron."""
        f_transit = self._true_anomaly_at_transit()
        E_transit = 2.0 * np.arctan2(
            np.sqrt(1.0 - self.ecc) * np.sin(f_transit / 2.0),
            np.sqrt(1.0 + self.ecc) * np.cos(f_transit / 2.0),
        )
        M_transit = E_transit - self.ecc * np.sin(E_transit)
        return t_peri + M_transit * self.period / (2.0 * np.pi)

    def _t0_to_t_periastron(self, t0):
        """Compute time of periastron from transit mid-time."""
        f_transit = self._true_anomaly_at_transit()
        E_transit = 2.0 * np.arctan2(
            np.sqrt(1.0 - self.ecc) * np.sin(f_transit / 2.0),
            np.sqrt(1.0 + self.ecc) * np.cos(f_transit / 2.0),
        )
        M_transit = E_transit - self.ecc * np.sin(E_transit)
        return t0 - M_transit * self.period / (2.0 * np.pi)

    # -----------------------------------------------------------------
    # Orbital position
    # -----------------------------------------------------------------

    def _get_true_anomaly(self, t):
        """Compute true anomaly at times *t*.

        Parameters
        ----------
        t : array_like
            Times [days].

        Returns
        -------
        f : jnp.ndarray
            True anomaly [rad].
        """
        t = jnp.asarray(t, dtype=jnp.float64)
        M = 2.0 * jnp.pi * (t - self.t_periastron) / self.period
        E = _solve_kepler(M, self.ecc)
        return _E_to_true_anomaly(E, self.ecc)

    def get_position(self, t):
        """Compute the 3-D position of the companion relative to the star.

        The coordinate system is:

        * **x** — on the sky, in the direction of increasing right
          ascension (or equivalently, the descending-node direction when
          ``Omega = 0``).
        * **y** — on the sky, toward celestial north (ascending-node
          direction when ``Omega = 0``).
        * **z** — along the line of sight, *toward* the observer.

        A transit occurs when ``z > 0`` (companion is in front of star).

        Parameters
        ----------
        t : array_like
            Times [days].

        Returns
        -------
        x, y, z : jnp.ndarray
            Position of the companion in units of stellar radii.
        """
        f = self._get_true_anomaly(t)

        # Radial distance in units of semi-major axis
        r = self.a_over_Rstar * (1.0 - self.ecc ** 2) / (
            1.0 + self.ecc * jnp.cos(f)
        )

        # Position in the orbital plane (X toward pericenter, Y perp)
        cos_f_w = jnp.cos(f + self.omega)
        sin_f_w = jnp.sin(f + self.omega)

        cos_Omega = jnp.cos(self.Omega)
        sin_Omega = jnp.sin(self.Omega)
        cos_incl = jnp.cos(self.incl)
        sin_incl = jnp.sin(self.incl)

        # Thiele-Innes rotation to observer frame
        x = r * (
            -sin_f_w * cos_Omega * cos_incl
            + cos_f_w * sin_Omega
        )  # not used for transit, kept for completeness
        y = r * (
            sin_f_w * sin_Omega * cos_incl
            + cos_f_w * cos_Omega
        )  # not used for transit, kept for completeness
        # z positive = toward observer (transit geometry)
        z = r * sin_f_w * sin_incl

        # Sky-plane rotation by Omega
        x_sky = -r * (
            cos_Omega * cos_f_w
            - sin_Omega * sin_f_w * cos_incl
        )
        y_sky = -r * (
            sin_Omega * cos_f_w
            + cos_Omega * sin_f_w * cos_incl
        )

        return x_sky, y_sky, z

    def get_sky_separation(self, t):
        """Projected sky-plane separation between companion and star.

        Parameters
        ----------
        t : array_like
            Times [days].

        Returns
        -------
        rho : jnp.ndarray
            Sky-plane separation in units of stellar radii.
        """
        x, y, z = self.get_position(t)
        return jnp.sqrt(x ** 2 + y ** 2)

    def get_radial_velocity(self, t, K=None):
        """Radial velocity of the star due to the companion.

        Parameters
        ----------
        t : array_like
            Times [days].
        K : float or None
            RV semi-amplitude [m/s].  If ``None``, the returned values
            are in natural units (useful for relative comparison only).

        Returns
        -------
        rv : jnp.ndarray
            Radial velocity [m/s if K is given, else natural units].
        """
        f = self._get_true_anomaly(t)
        rv = jnp.cos(f + self.omega) + self.ecc * jnp.cos(self.omega)
        if K is not None:
            rv = K * rv
        return rv

    def in_transit(self, t):
        """Boolean mask for times during which a transit is occurring.

        A point is in transit when the companion is in front of the star
        (``z > 0``) and the sky-plane separation is less than
        ``1 + radius_ratio`` stellar radii.

        Parameters
        ----------
        t : array_like
            Times [days].

        Returns
        -------
        mask : jnp.ndarray of bool
        """
        x, y, z = self.get_position(t)
        rho = jnp.sqrt(x ** 2 + y ** 2)
        return (z > 0) & (rho < 1.0 + self.radius_ratio)


# =====================================================================
# Quadratic limb-darkened transit light curve
# =====================================================================

class QuadLimbDarkLightCurve:
    """Compute a transit light curve with quadratic limb darkening.

    Uses the analytic Mandel & Agol (2002) uniform-source solution with a
    polynomial limb-darkening correction, following the ``exoplanet``
    implementation style.

    Parameters
    ----------
    u1 : float
        First quadratic limb-darkening coefficient.
    u2 : float
        Second quadratic limb-darkening coefficient.

        The specific intensity profile is:

        .. math::

            I(\\mu) / I(1) = 1 - u_1 (1 - \\mu) - u_2 (1 - \\mu)^2

        where :math:`\\mu = \\cos\\theta` and :math:`\\theta` is the
        angle from disk center.
    """

    def __init__(self, u1, u2):
        self.u1 = float(u1)
        self.u2 = float(u2)

    # -----------------------------------------------------------------
    # Mandel & Agol uniform-source occultation
    # -----------------------------------------------------------------

    @staticmethod
    @jax.jit
    def _uniform_occultation(z, p):
        """Fractional flux blocked by a uniform-source occultation.

        Parameters
        ----------
        z : array_like
            Projected separation in units of stellar radii.
        p : float
            Companion-to-star radius ratio.

        Returns
        -------
        delta_F : array_like
            Fraction of stellar flux blocked (>= 0).
        """
        z = jnp.abs(z)
        p = jnp.abs(p)

        # Case 1: complete occultation (companion fully covers disk)
        full = p >= 1.0
        full_occ = jnp.where(z <= p - 1.0, 1.0, 0.0)

        # Case 2: no overlap
        no_overlap = z >= 1.0 + p

        # Case 3: companion fully on disk
        on_disk = z <= 1.0 - p
        on_disk_flux = p ** 2

        # Case 4: partial overlap — area of intersection of two circles
        # of radii 1 and p separated by distance z
        z_safe = jnp.clip(z, 1e-12, None)
        kappa1 = jnp.arccos(
            jnp.clip((1.0 - p ** 2 + z_safe ** 2) / (2.0 * z_safe), -1, 1)
        )
        kappa0 = jnp.arccos(
            jnp.clip((p ** 2 + z_safe ** 2 - 1.0) / (2.0 * p * z_safe), -1, 1)
        )
        partial_flux = (
            p ** 2 * kappa0
            + kappa1
            - 0.5
            * jnp.sqrt(
                jnp.clip(
                    4.0 * z_safe ** 2 - (1.0 + z_safe ** 2 - p ** 2) ** 2,
                    0.0,
                    None,
                )
            )
        ) / jnp.pi

        delta = jnp.where(
            full & (z <= p - 1.0),
            full_occ,
            jnp.where(
                no_overlap,
                0.0,
                jnp.where(on_disk, on_disk_flux, partial_flux),
            ),
        )
        return delta

    # -----------------------------------------------------------------
    # Limb-darkening correction
    # -----------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def _limb_dark_factor(self, z, p):
        """Approximate limb-darkening correction factor.

        For a quadratic limb-darkening law, the transit depth depends on
        where the companion sits on the stellar disk.  This computes a
        multiplicative correction to the uniform-source solution using a
        first-order expansion in the limb-darkening coefficients.

        Parameters
        ----------
        z : array_like
            Projected separation in stellar radii.
        p : float
            Radius ratio.

        Returns
        -------
        factor : array_like
            Multiplicative correction (1.0 = no limb darkening).
        """
        u1 = self.u1
        u2 = self.u2

        # Normalization: integral of I(mu) over the disk
        I0 = 1.0 - u1 / 3.0 - u2 / 6.0

        # Mean mu under the companion shadow  (approximate for small p)
        mu_eff = jnp.sqrt(jnp.clip(1.0 - z ** 2, 0.0, 1.0))

        # Intensity at the shadow position
        I_shadow = 1.0 - u1 * (1.0 - mu_eff) - u2 * (1.0 - mu_eff) ** 2

        return I_shadow / I0

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def _compute_flux(self, t, x, y, z_pos, radius_ratio):
        """JIT-compiled flux computation."""
        rho = jnp.sqrt(x ** 2 + y ** 2)
        delta_uniform = self._uniform_occultation(rho, radius_ratio)
        ld_corr = self._limb_dark_factor(rho, radius_ratio)

        # Only block flux when companion is in front (z > 0)
        in_front = z_pos > 0.0
        delta = jnp.where(in_front, delta_uniform * ld_corr, 0.0)

        return 1.0 - delta

    def get_light_curve(self, orbit, t):
        """Compute the transit light curve for a Keplerian orbit.

        Parameters
        ----------
        orbit : KeplerianOrbit
            Orbital model instance.
        t : array_like
            Times [days].

        Returns
        -------
        flux : jnp.ndarray
            Relative flux (1.0 = out of transit).
        """
        t = jnp.asarray(t, dtype=jnp.float64)
        x, y, z_pos = orbit.get_position(t)
        return self._compute_flux(t, x, y, z_pos, orbit.radius_ratio)


# =====================================================================
# Combined spot + transit model
# =====================================================================

class SpotTransitModel:
    """Combined stellar variability (spots) + planetary transit model.

    The total flux is the product of the spot-modulated stellar flux and
    the transit flux:

    .. math::

        F(t) = F_{\\mathrm{spots}}(t) \\times F_{\\mathrm{transit}}(t)

    This is the standard multiplicative model: the transit removes a
    fraction of the *current* stellar brightness (including any spot
    modulation on the visible hemisphere).

    The spot component can be supplied in two ways:

    1. **Explicit spots** via a ``LightcurveModel`` instance — a forward
       simulation with individual spots drawn from the spot evolution
       model.
    2. **GP-based** via a ``GPSolver`` instance — draws from the GP
       prior or posterior to represent the stellar variability
       stochastically.

    Parameters
    ----------
    orbit : KeplerianOrbit
        Keplerian orbit of the transiting companion.
    limbdark : QuadLimbDarkLightCurve
        Limb-darkened transit light curve calculator.
    spot_model : LightcurveModel or GPSolver or None
        Source of stellar variability.  If ``None``, the spot component
        is unity (transit only).
    """

    def __init__(self, orbit, limbdark, spot_model=None):
        self.orbit = orbit
        self.limbdark = limbdark
        self.spot_model = spot_model

    # -----------------------------------------------------------------
    # Transit-only component
    # -----------------------------------------------------------------

    def get_transit_flux(self, t):
        """Compute the transit light curve alone (no spots).

        Parameters
        ----------
        t : array_like
            Times [days].

        Returns
        -------
        flux_transit : jnp.ndarray
            Relative flux from the transit (1.0 = out of transit).
        """
        return self.limbdark.get_light_curve(self.orbit, t)

    # -----------------------------------------------------------------
    # Spot-only component
    # -----------------------------------------------------------------

    def get_spot_flux(self, t):
        """Compute the spot-modulated stellar flux alone (no transit).

        Uses whichever spot model was provided at construction time.

        Parameters
        ----------
        t : array_like
            Times [days].

        Returns
        -------
        flux_spots : ndarray
            Relative flux from stellar variability.

        Raises
        ------
        ValueError
            If no spot model was provided.
        """
        if self.spot_model is None:
            raise ValueError("No spot model was provided.")

        if isinstance(self.spot_model, LightcurveModel):
            t_arr = np.asarray(t, dtype=float)
            return np.asarray(self.spot_model.Flux(t_arr))

        if isinstance(self.spot_model, GPSolver):
            mu, _ = self.spot_model.predict(np.asarray(t))
            return np.asarray(mu)

        raise TypeError(
            f"spot_model must be a LightcurveModel or GPSolver, "
            f"got {type(self.spot_model).__name__}"
        )

    # -----------------------------------------------------------------
    # Combined light curve
    # -----------------------------------------------------------------

    def get_light_curve(self, t):
        """Compute the combined spot + transit light curve.

        Parameters
        ----------
        t : array_like
            Times [days].

        Returns
        -------
        flux : jnp.ndarray
            Combined relative flux.
        """
        t = jnp.asarray(t, dtype=jnp.float64)
        flux_transit = self.get_transit_flux(t)

        if self.spot_model is None:
            return flux_transit

        flux_spots = jnp.asarray(self.get_spot_flux(t))
        return flux_spots * flux_transit

    # -----------------------------------------------------------------
    # Sampling (GP mode)
    # -----------------------------------------------------------------

    def sample_light_curves(self, t, n_samples=5, source="prior",
                            rng=None):
        """Draw combined spot + transit light curve samples.

        Only available when the spot model is a ``GPSolver``.  Each
        sample draws an independent realization of the spot variability
        and multiplies it by the deterministic transit model.

        Parameters
        ----------
        t : array_like
            Times [days].
        n_samples : int
            Number of samples to draw (default 5).
        source : {'prior', 'posterior'}
            Draw from the GP prior or posterior (default ``'prior'``).
        rng : numpy.random.Generator or None
            Random number generator for reproducibility.

        Returns
        -------
        t : jnp.ndarray, shape (M,)
            Evaluation times.
        samples : jnp.ndarray, shape (n_samples, M)
            Combined flux samples.

        Raises
        ------
        TypeError
            If the spot model is not a ``GPSolver``.
        """
        if not isinstance(self.spot_model, GPSolver):
            raise TypeError(
                "sample_light_curves requires a GPSolver spot model, "
                f"got {type(self.spot_model).__name__}"
            )

        t = jnp.asarray(t, dtype=jnp.float64)
        t_np = np.asarray(t)

        # Draw spot variability samples
        _, spot_samples = self.spot_model.sample_lightcurves(
            xpred=t_np, n_samples=n_samples, source=source, rng=rng,
        )
        spot_samples = jnp.asarray(spot_samples)

        # Deterministic transit component (same for every sample)
        flux_transit = self.get_transit_flux(t)

        # Multiply each spot sample by the transit
        combined = spot_samples * flux_transit[None, :]

        return t, combined

    # -----------------------------------------------------------------
    # Animation
    # -----------------------------------------------------------------

    def animate_lightcurve(self, t=None, fps=30, duration=10.0,
                           outfile=None, dpi=150, show_spots=True,
                           show_grid=True, show_params=True,
                           figsize=(14, 5.5), save_last_frame=None,
                           show_dr=True, label_size=18):
        """Animate the spotted star with transiting planet and combined
        light curve.

        Left panel shows the rotating stellar disk with spots and the
        planet transiting across it.  Right panel traces the combined
        (spots x transit) light curve over time, with the transit-only
        and spot-only components shown as faint references.

        Requires a ``LightcurveModel`` as the spot model.

        Parameters
        ----------
        t : array_like or None
            Time grid [days].  If ``None``, uses the spot model's
            internal grid ``spot_model.t``.
        fps : int
            Frames per second (default 30).
        duration : float
            Animation duration in seconds (default 10).
        outfile : str or None
            Output file path (.mp4 or .gif).  If ``None``, returns the
            animation object without saving.
        dpi : int
            Resolution (default 150).
        show_spots : bool
            If ``True``, show individual spot contributions on the
            light curve panel (default ``True``).
        show_grid : bool
            If ``True``, draw latitude/longitude grid on the star
            (default ``True``).
        show_params : bool
            If ``True``, show parameter annotation above the figure
            (default ``True``).
        figsize : tuple
            Figure size (default ``(14, 5.5)``).
        save_last_frame : str or None
            If provided, save the last frame as a static image.
        show_dr : bool
            If ``True``, color the stellar disk by differential
            rotation rate (default ``True``).
        label_size : int or float
            Font size for labels and tick marks (default 18).

        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
        """
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.patches import Circle

        # -- lazy imports from lightcurve helpers -------------------------
        try:
            from .lightcurve import (
                _projected_spot_patch, _alphak, _betak,
            )
        except ImportError:
            from lightcurve import (
                _projected_spot_patch, _alphak, _betak,
            )

        if not isinstance(self.spot_model, LightcurveModel):
            raise TypeError(
                "animate_lightcurve requires a LightcurveModel spot "
                f"model, got {type(self.spot_model).__name__}"
            )

        sm = self.spot_model
        orbit = self.orbit

        # -- time grid & fluxes -------------------------------------------
        if t is None:
            t = sm.t
        t = np.asarray(t, dtype=float)
        n_times = len(t)
        t_jax = jnp.array(t)

        flux_spots = np.asarray(sm.Flux(t))
        flux_transit = np.asarray(self.get_transit_flux(t))
        flux_combined = flux_spots * flux_transit

        # Per-spot flux deficits (set as side effect of Flux)
        dspots = sm.dspots  # (nspot, n_times)

        inc = sm.inc
        nspot = sm.nspot
        spot_longs = np.atleast_1d(sm.long)
        spot_lats = np.atleast_1d(sm.lat)
        spot_tmaxs = sm.tmax

        # -- precompute spot geometry for every frame ---------------------
        spot_alphas = np.zeros((nspot, n_times))
        spot_longs_t = np.zeros((nspot, n_times))
        for k in range(nspot):
            if sm.grow:
                spot_alphas[k] = np.asarray(_alphak(
                    t_jax, spot_tmaxs[k], sm.lspot,
                    sm.tem, sm.tdec, sm.alpha_max))
            else:
                spot_alphas[k] = sm.alpha_max
            if sm.rotate:
                _, longk_t = _betak(
                    t_jax, spot_longs[k], spot_lats[k], spot_tmaxs[k],
                    sm.peq, sm.kappa, sm.inc)
                spot_longs_t[k] = np.asarray(longk_t)
            else:
                spot_longs_t[k] = spot_longs[k]

        # -- precompute planet sky position (in stellar-radii units) ------
        px, py, pz = orbit.get_position(t)
        px = np.asarray(px)
        py = np.asarray(py)
        pz = np.asarray(pz)
        planet_r = orbit.radius_ratio  # in stellar radii

        # ================================================================
        # Figure layout
        # ================================================================
        fig, (ax_star, ax_lc) = plt.subplots(
            1, 2, figsize=figsize,
            gridspec_kw={"width_ratios": [1, 1.6]})

        # -- star panel ---------------------------------------------------
        ax_star.set_aspect("equal")
        ax_star.set_xlim(-1.55, 1.55)
        ax_star.set_ylim(-1.55, 1.55)
        ax_star.set_axis_off()

        if show_dr:
            from matplotlib.colors import Normalize
            import matplotlib.cm as cm

            omega_eq = 2 * np.pi / sm.peq
            cmap = cm.coolwarm

            if sm.kappa == 0:
                stellar_disk = Circle(
                    (0, 0), 1.0, fc="lightyellow", ec="k", lw=1.5,
                    zorder=-1)
                ax_star.add_patch(stellar_disk)

                n_pix = 300
                xp = np.linspace(-1, 1, n_pix)
                yp = np.linspace(-1, 1, n_pix)
                XP, YP = np.meshgrid(xp, yp)
                R2 = XP ** 2 + YP ** 2
                omega_map = np.where(R2 <= 1.0, 0.5, np.nan)
                norm = Normalize(vmin=0.0, vmax=1.0)
                dr_img = ax_star.imshow(
                    omega_map, extent=[-1, 1, -1, 1], origin="lower",
                    interpolation="bilinear", cmap=cmap, norm=norm,
                    alpha=0.3, zorder=0)
                clip_circle = Circle(
                    (0, 0), 1.0, transform=ax_star.transData)
                dr_img.set_clip_path(clip_circle)

                ax_star.text(
                    -1.3, 0.0,
                    rf"$\Omega = {omega_eq:.3f}$ [rad/d]",
                    fontsize=label_size - 2, ha="center", va="center",
                    rotation=90, transform=ax_star.transData)
            else:
                omega_min = omega_eq * (1 - sm.kappa)
                omega_max = omega_eq
                if omega_min > omega_max:
                    omega_min, omega_max = omega_max, omega_min
                norm = Normalize(vmin=omega_min, vmax=omega_max)

                n_pix = 300
                xp = np.linspace(-1, 1, n_pix)
                yp = np.linspace(-1, 1, n_pix)
                XP, YP = np.meshgrid(xp, yp)
                R2 = XP ** 2 + YP ** 2
                CZ = np.sqrt(np.clip(1.0 - R2, 0, None))
                sin_lat = (-np.sin(inc) * YP
                           + np.cos(inc) * CZ)
                sin_lat = np.clip(sin_lat, -1.0, 1.0)
                lat_map = np.arcsin(sin_lat)
                omega_map = omega_eq * (
                    1 - sm.kappa * np.sin(lat_map) ** 2)

                stellar_disk = Circle(
                    (0, 0), 1.0, fc="lightyellow", ec="k", lw=1.5,
                    zorder=-1)
                ax_star.add_patch(stellar_disk)
                dr_img = ax_star.imshow(
                    omega_map, extent=[-1, 1, -1, 1], origin="lower",
                    interpolation="bilinear", cmap=cmap, norm=norm,
                    alpha=0.3, zorder=0)
                clip_circle = Circle(
                    (0, 0), 1.0, transform=ax_star.transData)
                dr_img.set_clip_path(clip_circle)

                cbar = fig.colorbar(
                    dr_img, ax=ax_star, fraction=0.046, pad=0.04,
                    location="left")
                cbar.set_label(r"$\Omega$ [rad/d]",
                               fontsize=label_size)
                cbar.ax.tick_params(labelsize=label_size - 2)
                cbar.ax.text(
                    0.6, 1.02, "faster",
                    transform=cbar.ax.transAxes, ha="center",
                    va="bottom", fontsize=label_size - 2, color="red")
                cbar.ax.text(
                    0.6, -0.02, "slower",
                    transform=cbar.ax.transAxes, ha="center",
                    va="top", fontsize=label_size - 2, color="blue")
        else:
            stellar_disk = Circle(
                (0, 0), 1.0, fc="lightyellow", ec="k", lw=1.5,
                zorder=0)
            ax_star.add_patch(stellar_disk)

        # Grid lines
        if show_grid:
            phi_grid = np.linspace(0, 2 * np.pi, 200)
            for lat_deg in [0, 30, 60, -30, -60]:
                lat_r = np.radians(lat_deg)
                gx = (-np.sin(inc) * np.sin(lat_r)
                      + np.cos(inc) * np.cos(lat_r)
                      * np.cos(phi_grid))
                gy = np.cos(lat_r) * np.sin(phi_grid)
                gz = (np.cos(inc) * np.sin(lat_r)
                      + np.sin(inc) * np.cos(lat_r)
                      * np.cos(phi_grid))
                mask = gz > 0
                style = (("k--", 0.6, 0.3) if lat_deg == 0
                         else ("k-", 0.3, 0.2))
                ax_star.plot(
                    np.where(mask, gy, np.nan),
                    np.where(mask, gx, np.nan),
                    style[0], lw=style[1], alpha=style[2])

        # Rotation axis arrow
        ax_star.annotate(
            "", xy=(0, 1.4), xytext=(0, -0.3),
            arrowprops=dict(arrowstyle="->, head_width=0.08",
                            color="0.5", lw=1.2))

        # Spot patches
        spot_colors = plt.cm.Set1(np.linspace(0, 1, max(nspot, 1)))
        spot_patches_art = []
        ghost_patches_art = []
        for k in range(nspot):
            c = spot_colors[k % len(spot_colors)]
            patch, = ax_star.fill([], [], color=c, alpha=0.85,
                                  zorder=2)
            ghost, = ax_star.fill([], [], color=c, alpha=0.15,
                                  zorder=1, linestyle="--",
                                  edgecolor=c, linewidth=0.8)
            spot_patches_art.append(patch)
            ghost_patches_art.append(ghost)

        # Planet circle (dark disk, high zorder so it draws over spots)
        planet_patch = Circle(
            (0, 0), planet_r, fc="0.15", ec="k", lw=0.6,
            zorder=10, visible=False)
        ax_star.add_patch(planet_patch)

        time_text = ax_star.text(
            0, -1.45, "", fontsize=label_size, ha="center", va="top")

        # -- light curve panel --------------------------------------------
        dip_combined = (1 - flux_combined) * 100
        dip_transit = (1 - flux_transit) * 100
        dip_spots = (1 - flux_spots) * 100
        dip_per_spot = dspots * 100

        dip_max = max(np.max(dip_combined), 1e-3)
        dip_range = dip_max if dip_max > 0 else 1.0

        ax_lc.set_xlim(t[0], t[-1])
        ax_lc.set_ylim(-0.05 * dip_range,
                        dip_max + 0.15 * dip_range)
        ax_lc.invert_yaxis()
        ax_lc.set_xlabel("Time [days]", fontsize=label_size)
        ax_lc.set_ylabel(r"Flux dip [\%]", fontsize=label_size)
        ax_lc.tick_params(labelsize=label_size - 2)
        ax_lc.minorticks_on()

        # Full light curves as faint background
        ax_lc.plot(t, dip_combined, "k-", lw=0.3, alpha=0.12,
                   zorder=0)
        ax_lc.plot(t, dip_transit, "-", color="C0", lw=0.3,
                   alpha=0.12, zorder=0)
        ax_lc.plot(t, dip_spots, "-", color="C1", lw=0.3,
                   alpha=0.12, zorder=0)

        # Traced lines (build up over time)
        lc_line, = ax_lc.plot([], [], "k-", lw=1.4, zorder=4,
                              label="Combined")
        transit_line, = ax_lc.plot([], [], "-", color="C0", lw=1.0,
                                   alpha=0.7, zorder=3,
                                   label="Transit")
        spots_line, = ax_lc.plot([], [], "-", color="C1", lw=1.0,
                                  alpha=0.7, zorder=3,
                                  label="Spots")

        # Per-spot traces
        spot_lc_lines = []
        if show_spots:
            for k in range(nspot):
                c = spot_colors[k % len(spot_colors)]
                ln, = ax_lc.plot([], [], "-", color=c, lw=0.6,
                                 alpha=0.4, zorder=1)
                spot_lc_lines.append(ln)

        # Vertical time marker
        vline = ax_lc.axvline(
            0, color="C3", lw=1.0, alpha=0.7, ls="--", zorder=5)

        ax_lc.legend(fontsize=label_size - 4, loc="upper right")

        fig.tight_layout()

        # Parameter annotation
        if show_params:
            spot_text = (
                rf"$P_{{\rm eq}}={sm.peq:.1f}$ d,  "
                rf"$\kappa={sm.kappa:.2f}$,  "
                rf"$I={sm.inc_deg:.0f}^\circ$,  "
                rf"$N_{{\rm spot}}={sm.nspot}$"
            )
            transit_text = (
                rf"$P_{{\rm orb}}={orbit.period:.2f}$ d,  "
                rf"$R_p/R_*={orbit.radius_ratio:.3f}$,  "
                rf"$b={orbit.impact_param:.2f}$,  "
                rf"$e={orbit.ecc:.2f}$"
            )
            fig.text(0.5, 0.99,
                     spot_text + r"  $|$  " + transit_text,
                     fontsize=label_size - 2, ha="center", va="top")
            fig.subplots_adjust(top=0.90)

        # ================================================================
        # Animation loop
        # ================================================================
        n_frames = int(fps * duration)
        frame_indices = np.linspace(
            0, n_times - 1, n_frames).astype(int)
        empty_xy = np.empty((0, 2))

        def update(frame_num):
            idx = frame_indices[frame_num]
            t_now = t[idx]

            # -- update spots on the star ---------------------------------
            for k in range(nspot):
                alpha_k = spot_alphas[k, idx]
                if alpha_k < 1e-6:
                    spot_patches_art[k].set_xy(empty_xy)
                    ghost_patches_art[k].set_xy(empty_xy)
                    continue

                lon_k = spot_longs_t[k, idx]
                lat_k = spot_lats[k]

                fx, fy, bx, by = _projected_spot_patch(
                    lon_k, lat_k, alpha_k, inc)

                if fx is not None and len(fx) >= 3:
                    spot_patches_art[k].set_xy(
                        np.column_stack([fx, fy]))
                else:
                    spot_patches_art[k].set_xy(empty_xy)

                if bx is not None and len(bx) >= 3:
                    ghost_patches_art[k].set_xy(
                        np.column_stack([bx, by]))
                else:
                    ghost_patches_art[k].set_xy(empty_xy)

            # -- update planet on the star --------------------------------
            # Planet sky coords: x_sky -> horizontal, y_sky -> vertical
            # on the star panel.  Show only when in front (pz > 0).
            if pz[idx] > 0:
                planet_patch.set_visible(True)
                planet_patch.center = (px[idx], py[idx])
            else:
                planet_patch.set_visible(False)

            time_text.set_text(rf"$t = {t_now:.2f}$ d")

            # -- update light curve traces --------------------------------
            lc_line.set_data(t[:idx + 1], dip_combined[:idx + 1])
            transit_line.set_data(t[:idx + 1], dip_transit[:idx + 1])
            spots_line.set_data(t[:idx + 1], dip_spots[:idx + 1])

            if show_spots:
                for k in range(nspot):
                    spot_lc_lines[k].set_data(
                        t[:idx + 1], dip_per_spot[k, :idx + 1])

            vline.set_xdata([t_now])

            return (spot_patches_art + ghost_patches_art
                    + [planet_patch, time_text, lc_line,
                       transit_line, spots_line, vline]
                    + spot_lc_lines)

        anim = animation.FuncAnimation(
            fig, update, frames=n_frames,
            interval=1000 / fps, blit=False)

        if outfile is not None:
            import os
            outdir = os.path.dirname(outfile)
            if outdir:
                os.makedirs(outdir, exist_ok=True)

            if outfile.endswith(".gif"):
                writer = animation.PillowWriter(fps=fps)
            else:
                writer = animation.FFMpegWriter(fps=fps, bitrate=2000)

            print(f"Rendering {n_frames} frames to {outfile}...")
            anim.save(outfile, writer=writer, dpi=dpi)
            print("Done.")

        if save_last_frame is not None:
            update(n_frames - 1)
            fig.savefig(save_last_frame, dpi=dpi,
                        bbox_inches="tight")
            print(f"Last frame saved to {save_last_frame}")

        plt.close(fig)

        return anim
