import jax
import jax.numpy as jnp
import numpy as np
import warnings

import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use('classic')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('figure', facecolor='w')
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)

try:
    from .params import resolve_hparam
    from .spot_model import SpotEvolutionModel
except ImportError:
    from params import resolve_hparam
    from spot_model import SpotEvolutionModel

__all__ = ["LightcurveModel", "compute_sigmak"]


def compute_sigmak(nspot_rate, alpha_max, fspot=0.0):
    """Compute the kernel amplitude prefactor sigma_k.

    Thin wrapper around params.resolve_hparam for the physical_rate mode.

    Parameters
    ----------
    nspot_rate : float
        Spot emergence rate [spots/day].
    alpha_max : float
        Peak spot angular radius [rad].
    fspot : float, optional
        Spot contrast fraction (default 0).

    Returns
    -------
    sigma_k : float
        sigma_k = sqrt(nspot_rate) * (1 - fspot) * alpha_max**2
    """
    return np.sqrt(nspot_rate) * (1 - fspot) * alpha_max**2


# =====================================================================
# Spot projection helpers for animation
# =====================================================================

def _projected_spot_patch(lon, lat, alpha, inc, n_pts=60):
    """
    Compute the 2D projected outline of a circular spot on a sphere.

    Parameters
    ----------
    lon : float
        Spot longitude (radians).
    lat : float
        Spot latitude (radians).
    alpha : float
        Spot angular radius (radians).
    inc : float
        Stellar inclination (radians).
    n_pts : int
        Number of points in the outline polygon.

    Returns
    -------
    front_x, front_y : ndarray or None
        Visible portion outline.
    back_x, back_y : ndarray or None
        Hidden (far-side) portion outline.
    """
    # Spot center direction in observer frame
    cx = -np.sin(inc) * np.sin(lat) + np.cos(inc) * np.cos(lat) * np.cos(lon)
    cy = np.cos(lat) * np.sin(lon)
    cz = np.cos(inc) * np.sin(lat) + np.sin(inc) * np.cos(lat) * np.cos(lon)

    c_vec = np.array([cx, cy, cz])

    # Build orthonormal basis on the tangent plane at spot center
    up = np.array([0, 0, 1.0]) if abs(cz) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(c_vec, up)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(c_vec, e1)
    e2 /= np.linalg.norm(e2)

    # Trace the spot boundary on the unit sphere
    phi = np.linspace(0, 2 * np.pi, n_pts)
    pts = (np.cos(alpha) * c_vec[:, None]
           + np.sin(alpha) * (np.cos(phi) * e1[:, None]
                              + np.sin(phi) * e2[:, None]))

    proj_x = pts[1]  # right on sky
    proj_y = pts[0]  # up on sky
    visible = pts[2] > 0

    if np.all(visible):
        return proj_x, proj_y, None, None
    elif not np.any(visible):
        return None, None, proj_x, proj_y
    else:
        fx, fy = _extract_visible(proj_x, proj_y, pts, visible, n_pts)
        bx, by = _extract_hidden(proj_x, proj_y, pts, visible, n_pts)
        return fx, fy, bx, by


def _extract_visible(proj_x, proj_y, pts, visible, n_pts):
    """Extract visible portion of spot outline with limb interpolation."""
    xs, ys = [], []
    for i in range(n_pts):
        if visible[i]:
            xs.append(proj_x[i])
            ys.append(proj_y[i])
        else:
            if i > 0 and visible[i - 1]:
                t = pts[2, i - 1] / (pts[2, i - 1] - pts[2, i])
                xs.append(proj_x[i - 1] + t * (proj_x[i] - proj_x[i - 1]))
                ys.append(proj_y[i - 1] + t * (proj_y[i] - proj_y[i - 1]))
            if i < n_pts - 1 and visible[i + 1]:
                t = pts[2, i] / (pts[2, i] - pts[2, i + 1])
                xs.append(proj_x[i] + t * (proj_x[i + 1] - proj_x[i]))
                ys.append(proj_y[i] + t * (proj_y[i + 1] - proj_y[i]))
    if len(xs) < 3:
        return None, None
    return np.array(xs), np.array(ys)


def _extract_hidden(proj_x, proj_y, pts, visible, n_pts):
    """Extract hidden portion of spot outline with limb interpolation."""
    xs, ys = [], []
    for i in range(n_pts):
        if not visible[i]:
            xs.append(proj_x[i])
            ys.append(proj_y[i])
        else:
            if i > 0 and not visible[i - 1]:
                t = pts[2, i - 1] / (pts[2, i - 1] - pts[2, i])
                xs.append(proj_x[i - 1] + t * (proj_x[i] - proj_x[i - 1]))
                ys.append(proj_y[i - 1] + t * (proj_y[i] - proj_y[i - 1]))
            if i < n_pts - 1 and not visible[i + 1]:
                t = pts[2, i] / (pts[2, i] - pts[2, i + 1])
                xs.append(proj_x[i] + t * (proj_x[i + 1] - proj_x[i]))
                ys.append(proj_y[i] + t * (proj_y[i + 1] - proj_y[i]))
    if len(xs) < 3:
        return None, None
    return np.array(xs), np.array(ys)


@jax.jit
def _zeta(x):
    """Calculate zeta(x) for spot limb darkening."""
    return (jnp.cos(x) * jnp.heaviside(x, 1.0) * jnp.heaviside(jnp.pi/2 - x, 1.0)
            + jnp.heaviside(-x, 1.0))


@jax.jit
def _alphak(teval, tmaxk, lspot, tem, tdec, alpha_max):
    """Compute spot angular size evolution (vectorized over time)."""
    dt1 = teval - tmaxk + lspot/2 + tem
    dt2 = teval - tmaxk + lspot/2
    dt3 = teval - tmaxk - lspot/2
    dt4 = teval - tmaxk - lspot/2 - tdec

    alphak  = (dt1 * jnp.heaviside(dt1, 1.0) - dt2 * jnp.heaviside(dt2, 1.0)) / tem
    alphak += -(dt3 * jnp.heaviside(dt3, 1.0) - dt4 * jnp.heaviside(dt4, 1.0)) / tdec
    alphak *= alpha_max

    return alphak


@jax.jit
def _betak(teval, longk, latk, tmaxk, peq, kappa, inc):
    """Compute spot angle from disk center (vectorized over time)."""
    longk_t = longk + 2*jnp.pi/peq * (1 - kappa * jnp.sin(latk)**2) * (teval - tmaxk)

    cosb  = jnp.cos(inc) * jnp.sin(latk)
    cosb += jnp.sin(inc) * jnp.cos(latk) * jnp.cos(longk_t)
    betak_t = jnp.arccos(jnp.clip(cosb, -1.0, 1.0))

    return betak_t, longk_t


@jax.jit
def _dflux_single_spot(teval, longk, latk, tmaxk,
                       peq, kappa, inc, lspot, tem, tdec, alpha_max, fspot):
    """
    Compute flux deficit for a single spot over all time steps.
    Fully vectorized over time using JAX.
    """
    betak_t, _ = _betak(teval, longk, latk, tmaxk, peq, kappa, inc)
    alphak_t = _alphak(teval, tmaxk, lspot, tem, tdec, alpha_max)

    cosa = jnp.cos(alphak_t)
    sina = jnp.sin(alphak_t)
    cosb = jnp.cos(betak_t)
    sinb = jnp.sin(betak_t)

    # Avoid division by zero with small epsilon
    eps = 1e-30
    cota = cosa / (sina + eps)
    cscb = 1.0 / (sinb + eps)
    cotb = cosb / (sinb + eps)

    # Clamp argument for arccos to [-1, 1]
    arg1 = jnp.clip(cosa * cscb, -1.0, 1.0)
    arg2 = jnp.clip(-cota * cotb, -1.0, 1.0)
    sqrt_arg = jnp.clip(1 - cosa**2 * cscb**2, 0.0, None)

    Ak  = jnp.arccos(arg1)
    Ak += cosb * sina**2 * jnp.arccos(arg2)
    Ak -= cosa * sinb * jnp.sqrt(sqrt_arg)

    # Simple spot limb darkening factor (no limb darkening case)
    factor = 1.0 - fspot

    dspot = Ak / jnp.pi * factor

    # Zero out contributions where spot has zero size
    dspot = jnp.where(alphak_t > 1e-15, dspot, 0.0)

    return dspot


# Vectorize over spots (batch the single-spot function over spot index)
_dflux_all_spots = jax.vmap(
    _dflux_single_spot,
    in_axes=(None, 0, 0, 0,    # teval shared; longk, latk, tmaxk per-spot
             None, None, None, None, None, None, None, None)  # scalar params shared
)


@jax.jit
def _dflux_single_spot_fixed(teval, tmaxk, lspot, tem, tdec, alpha_max, fspot):
    """
    Flux deficit for a spot fixed at disk center (no stellar rotation).

    Equivalent to _dflux_single_spot with beta=0 at all times: only the
    spot size envelope drives flux changes.  With beta=0 the projected
    area simplifies to A_k = pi * sin^2(alpha).
    """
    alphak_t = _alphak(teval, tmaxk, lspot, tem, tdec, alpha_max)

    sina = jnp.sin(alphak_t)
    dspot = sina**2 * (1.0 - fspot)
    dspot = jnp.where(alphak_t > 1e-15, dspot, 0.0)

    return dspot


# Vectorize fixed-spot function over tmaxk only (no per-spot geometry)
_dflux_all_spots_fixed = jax.vmap(
    _dflux_single_spot_fixed,
    in_axes=(None, 0, None, None, None, None, None)  # teval shared; tmaxk per-spot
)


@jax.jit
def _dflux_single_spot_constant(teval, longk, latk, tmaxk,
                                peq, kappa, inc, alpha_max, fspot):
    """
    Flux deficit for a spot with constant angular size (no envelope evolution).

    The spot is always at full size alpha_max; only stellar rotation via
    _betak modulates the projected area.
    """
    betak_t, _ = _betak(teval, longk, latk, tmaxk, peq, kappa, inc)

    cosa = jnp.cos(alpha_max)
    sina = jnp.sin(alpha_max)
    cosb = jnp.cos(betak_t)
    sinb = jnp.sin(betak_t)

    eps = 1e-30
    cota = cosa / (sina + eps)
    cscb = 1.0 / (sinb + eps)
    cotb = cosb / (sinb + eps)

    arg1 = jnp.clip(cosa * cscb, -1.0, 1.0)
    arg2 = jnp.clip(-cota * cotb, -1.0, 1.0)
    sqrt_arg = jnp.clip(1 - cosa**2 * cscb**2, 0.0, None)

    Ak  = jnp.arccos(arg1)
    Ak += cosb * sina**2 * jnp.arccos(arg2)
    Ak -= cosa * sinb * jnp.sqrt(sqrt_arg)

    return Ak / jnp.pi * (1.0 - fspot)


# Vectorize constant-size function over per-spot geometry
_dflux_all_spots_constant = jax.vmap(
    _dflux_single_spot_constant,
    in_axes=(None, 0, 0, 0, None, None, None, None, None)
)


class LightcurveModel(object):
    """
    JAX-accelerated star with spots and its lightcurve.

    Same interface as the numpy version but uses JAX for vectorized
    computation across all spots simultaneously.

    Args:
        peq (float): Equatorial period of the star.
        kappa (float): Differential rotation shear.
        inc (float): Inclination of the star.
        nspot (int): Number of spots.
        tau_spot (float, optional): Timescale for both emergence and decay of the spots. Defaults to None.
        tem (float, optional): Emergence timescale of the spots. Defaults to 2.
        tdec (float, optional): Decay timescale of the spots. Defaults to 2.
        alpha_max (float, optional): Maximum angular area of the spots. Defaults to 0.1.
        fspot (float, optional): Spot contrast fraction. Defaults to 0.
        lspot (float, optional): Spot lifetime. Defaults to 5.
        long (list, optional): Range of spot longitudes. Defaults to [0, 2*pi].
        lat (list, optional): Range of spot latitudes. Defaults to [0, pi].
        tsim (float, optional): End simulation time. Defaults to 28.
        tsamp (float, optional): Sampling cadence. Defaults to 0.02.
        limb_darkening (bool, optional): Flag to enable limb darkening. Defaults to False.
    """
    def __init__(self, peq=4.0, kappa=0.0, inc=np.pi/2, nspot=None,
                 tau_spot=None, tem=2, tdec=2, alpha_max=0.1, fspot=0, lspot=5,
                 long=[0, 2*np.pi], lat=[-np.pi/2, np.pi/2],
                 tsim=28, tsamp=0.02, limb_darkening=False, tmax=None,
                 rotate=True, grow=True, nspot_rate=None):

        # simulation parameters
        self.tsim = tsim
        self.tsamp = tsamp
        self.t = np.arange(0, self.tsim, self.tsamp)

        # star properties
        self.peq = peq
        self.kappa = kappa
        self.inc = inc
        self.inc_deg = inc * 180/np.pi

        # resolve nspot from nspot_rate if needed
        if nspot_rate is not None:
            self.nspot_rate = float(nspot_rate)
            self.nspot = max(1, int(nspot_rate * tsim))
        elif nspot is not None:
            self.nspot_rate = None
            self.nspot = int(nspot)
        else:
            self.nspot_rate = None
            self.nspot = 10

        # spot properties (scalars)
        if tau_spot is not None:
            self.tem = tau_spot
            self.tdec = tau_spot
        else:
            self.tem = tem
            self.tdec = tdec
        self.alpha_max = alpha_max
        self.fspot = fspot
        self.lspot = lspot
        self.tlifetime = self.lspot + self.tem + self.tdec

        self.long = self._assign_property(long)
        self.lat = self._assign_property(lat)
        if tmax is None:
            self.tmax = np.random.uniform(-(self.lspot/2 + self.tdec),
                                          self.tsim + self.lspot/2 + self.tem,
                                          self.nspot)
        elif isinstance(tmax, float):
            self.tmax = np.full(self.nspot, tmax)
        else:
            self.tmax = np.asarray(tmax)

        self.rotate = bool(rotate)
        self.grow   = bool(grow)

        # limb darkening
        self.limb_darkening = limb_darkening
        self.limbc = np.array([0.3999, 0.4269, -0.0227, -0.0839])
        self.limbd = self.limbc

        # compute lightcurve using JAX
        self.flux = self.Flux(self.t)

    @classmethod
    def from_spot_model(cls, spot_model: "SpotEvolutionModel",
                        nspot: int = None, *, nspot_rate: float = None, **kwargs):
        """Construct a LightcurveModel from a SpotEvolutionModel.

        Parameters
        ----------
        spot_model : SpotEvolutionModel
            Fully configured spot evolution model.
        nspot : int, optional
            Total number of spots to simulate.
        nspot_rate : float, optional
            Spot emergence rate [spots/day]. The actual number of spots is
            ``max(1, int(nspot_rate * tsim))``. Exactly one of ``nspot`` or
            ``nspot_rate`` must be provided.
        **kwargs
            Forwarded to LightcurveModel.__init__ (e.g. tsim, tsamp, lat, long).

        Returns
        -------
        LightcurveModel
        """
        if nspot is None and nspot_rate is None:
            raise ValueError("Provide either nspot or nspot_rate.")
        if nspot is not None and nspot_rate is not None:
            raise ValueError("Provide either nspot or nspot_rate, not both.")
        from .envelope import TrapezoidAsymmetricEnvelope
        env = spot_model.envelope
        if env is not None:
            if isinstance(env, TrapezoidAsymmetricEnvelope):
                tau_em  = env.tau_em
                tau_dec = env.tau_dec
            else:
                tau_em  = env.tau_spot
                tau_dec = env.tau_spot
            lspot = spot_model.lspot
        else:
            tau_em  = kwargs.pop("tem",  kwargs.pop("tau_spot", 2.0))
            tau_dec = kwargs.pop("tdec", tau_em)
            lspot   = kwargs.pop("lspot", 5.0)
        alpha_max = spot_model.alpha_max if spot_model.alpha_max is not None \
                    else kwargs.pop("alpha_max", 0.1)
        fspot     = spot_model.fspot if spot_model.fspot else kwargs.pop("fspot", 0.0)
        if "lat" not in kwargs:
            kwargs["lat"] = list(spot_model.latitude_distribution.lat_range)
        vis = spot_model.visibility
        return cls(
            peq=vis.peq if vis is not None else kwargs.pop("peq", 4.0),
            kappa=vis.kappa if vis is not None else kwargs.pop("kappa", 0.0),
            inc=vis.inc if vis is not None else kwargs.pop("inc", np.pi / 2),
            nspot=nspot,
            nspot_rate=nspot_rate,
            tem=tau_em,
            tdec=tau_dec,
            alpha_max=alpha_max,
            fspot=fspot,
            lspot=lspot,
            rotate=(vis is not None),
            grow=(spot_model.envelope is not None),
            **kwargs,
        )

    @classmethod
    def from_hparam(cls, hparam: dict, nspot: int = None, *,
                    nspot_rate: float = None, **kwargs):
        """Construct a LightcurveModel from a GPSolver-compatible hparam dict.

        Accepts the same raw hparam dict that GPSolver/AnalyticKernel take,
        including all amplitude modes (sigma_k, nspot_rate, or nspot), and
        both symmetric (tau) and asymmetric (tau_em + tau_dec) envelopes.
        This removes the need to manually decompose the dict in scripts.

        Parameters
        ----------
        hparam : dict
            Raw hyperparameter dict.  Must contain peq, kappa, inc, lspot,
            tau_spot (or tau_em/tau_dec), and an amplitude specification.
        nspot : int, optional
            Total number of spots to simulate.
        nspot_rate : float, optional
            Spot emergence rate [spots/day]. Exactly one of ``nspot`` or
            ``nspot_rate`` must be provided.
        **kwargs
            Forwarded to LightcurveModel.__init__ (e.g. tsim, tsamp, lat, long).

        Returns
        -------
        LightcurveModel
        """
        if nspot is None and nspot_rate is None:
            raise ValueError("Provide either nspot or nspot_rate.")
        if nspot is not None and nspot_rate is not None:
            raise ValueError("Provide either nspot or nspot_rate, not both.")
        p = resolve_hparam(hparam)
        tau_em  = p.get("tau_em",  p["tau_spot"])
        tau_dec = p.get("tau_dec", p["tau_spot"])
        alpha_max = p.get("alpha_max", kwargs.pop("alpha_max", 0.1))
        fspot     = p.get("fspot",     kwargs.pop("fspot", 0.0))
        return cls(
            peq=p["peq"], kappa=p["kappa"], inc=p["inc"],
            nspot=nspot, nspot_rate=nspot_rate,
            tem=tau_em, tdec=tau_dec,
            alpha_max=alpha_max, fspot=fspot, lspot=p["lspot"],
            **kwargs,
        )

    def _assign_property(self, var):
        if isinstance(var, float):
            return np.full(self.nspot, var)
        elif isinstance(var, (int, list, np.ndarray)):
            return np.random.uniform(var[0], var[1], self.nspot)
        else:
            raise TypeError("Invalid datatype for model parameter. "
                            "Valid types: int, float, list, np.ndarray")

    def Flux(self, teval):
        """
        Compute the full lightcurve using JAX vmap over all spots.

        Instead of a Python loop over nspot, all spots are computed
        in parallel via JAX's vmap.
        """
        teval_jax = jnp.array(teval)
        long_jax = jnp.array(np.atleast_1d(self.long))
        lat_jax = jnp.array(np.atleast_1d(self.lat))
        tmax_jax = jnp.array(self.tmax)

        # Compute all spots in parallel via vmap
        if self.rotate and self.grow:
            dspots = _dflux_all_spots(
                teval_jax, long_jax, lat_jax, tmax_jax,
                self.peq, self.kappa, self.inc,
                self.lspot, self.tem, self.tdec, self.alpha_max, self.fspot
            )
        elif self.rotate and not self.grow:
            dspots = _dflux_all_spots_constant(
                teval_jax, long_jax, lat_jax, tmax_jax,
                self.peq, self.kappa, self.inc, self.alpha_max, self.fspot
            )
        elif not self.rotate and self.grow:
            dspots = _dflux_all_spots_fixed(
                teval_jax, tmax_jax,
                self.lspot, self.tem, self.tdec, self.alpha_max, self.fspot
            )
        else:  # not rotate, not grow
            dspots = _dflux_all_spots_constant(
                teval_jax, long_jax, lat_jax, tmax_jax,
                self.peq, self.kappa, self.inc, self.alpha_max, self.fspot
            )

        # Convert back to numpy for storage
        self.dspots = np.asarray(dspots)

        # Stellar limb darkening
        self.dlimb = self._stellar_limb()

        # Total remaining flux
        flux = 1 - self.dlimb - np.sum(self.dspots, axis=0)
        return flux

    def _stellar_limb(self):
        if self.limb_darkening:
            ncoeff = len(self.limbc)
            return np.sum([n*self.limbc[n] / (n + ncoeff) for n in range(ncoeff)])
        return 0.0

    def plot_lightcurve(self, show_spots=True, show_title=True):
        """Plot the lightcurve."""
        flux = self.flux + self.dlimb
        dflux_pct = (flux - 1) * 100
        fig = plt.figure(figsize=[16, 6])
        if show_spots:
            for ii in range(self.nspot):
                plt.plot(self.t, -self.dspots[ii] * 100, alpha=0.5)
        plt.plot(self.t, dflux_pct, color="k")

        if show_title:
            title = r"$P_{{\rm eq}}$={:.1f} d, ".format(self.peq)
            title += r"$\kappa$={:.2f}, ".format(self.kappa)
            title += r"$i$={:.0f} deg, ".format(self.inc_deg)
            title += r"nspot={:.0f}, ".format(self.nspot)
            title += r"$\alpha_{{\rm max}}$={:.1f}, ".format(self.alpha_max)
            title += r"$l_{{\rm spot}}$={:.2f}, ".format(self.lspot)
            title += r"$\tau_{{\rm em}}$={:.2f}, ".format(self.tem)
            title += r"$\tau_{{\rm dec}}$={:.2f}".format(self.tdec)
            plt.title(title, fontsize=25)
        plt.xlabel("Time [days]", fontsize=24)
        plt.ylabel(r"$\Delta$ Flux [\%]", fontsize=24)
        plt.ylim(min(dflux_pct) - 0.2, max(dflux_pct) + 0.2)
        plt.xlim(self.t[0], self.t[-1])
        plt.minorticks_on()
        plt.ticklabel_format(axis='both', style='', useOffset=False)
        plt.close()

        return fig

    def animate_lightcurve(self, fps=30, duration=10.0, outfile=None,
                           dpi=150, show_spots=True, show_grid=True,
                           show_params=True, figsize=(14, 5.5),
                           save_last_frame=None, show_dr=True,
                           label_size=18):
        """
        Animate the starspot evolution with two panels: a 2D projection
        of the rotating star (left) and the lightcurve (right).

        Parameters
        ----------
        fps : int
            Frames per second (default 30).
        duration : float
            Animation duration in seconds (default 10).
        outfile : str or None
            Output file path (.mp4 or .gif). If None, returns the
            animation object without saving.
        dpi : int
            Resolution (default 150).
        show_spots : bool
            If True, show individual spot contributions on the
            lightcurve panel (default True).
        show_grid : bool
            If True, draw latitude/longitude grid on the star
            (default True).
        show_params : bool
            If True, show parameter annotation on the lightcurve
            panel (default True).
        figsize : tuple
            Figure size (default (14, 5.5)).
        save_last_frame : str or None
            If provided, save the last frame of the animation as a
            static image to this file path (e.g. "frame.png").
        show_dr : bool
            If True, color the stellar disk by latitude-dependent
            rotation frequency and display a colorbar (default True).
        label_size : int or float
            Font size for all labels, tick marks, and text in the
            plot (default 18).

        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            The animation object.
        """
        import matplotlib.animation as animation
        from matplotlib.patches import Circle

        t = self.t
        flux = self.flux + self.dlimb
        inc = self.inc
        nspot = self.nspot
        n_times = len(t)

        spot_longs = np.atleast_1d(self.long)
        spot_lats = np.atleast_1d(self.lat)
        spot_tmaxs = self.tmax

        # Precompute spot alphas and longitudes for all times
        t_jax = jnp.array(t)
        spot_alphas = np.zeros((nspot, n_times))
        spot_longs_t = np.zeros((nspot, n_times))
        for k in range(nspot):
            if self.grow:
                spot_alphas[k] = np.asarray(_alphak(
                    t_jax, spot_tmaxs[k], self.lspot,
                    self.tem, self.tdec, self.alpha_max))
            else:
                spot_alphas[k] = self.alpha_max
            if self.rotate:
                _, longk_t = _betak(
                    t_jax, spot_longs[k], spot_lats[k], spot_tmaxs[k],
                    self.peq, self.kappa, self.inc)
                spot_longs_t[k] = np.asarray(longk_t)
            else:
                spot_longs_t[k] = spot_longs[k]

        # --- Set up figure ---
        fig, (ax_star, ax_lc) = plt.subplots(
            1, 2, figsize=figsize,
            gridspec_kw={"width_ratios": [1, 1.6]})

        # Star panel
        ax_star.set_aspect("equal")
        ax_star.set_xlim(-1.35, 1.35)
        ax_star.set_ylim(-1.35, 1.35)
        ax_star.set_axis_off()

        if show_dr:
            # Color the stellar disk by differential rotation rate
            from matplotlib.colors import Normalize
            import matplotlib.cm as cm

            omega_eq = 2 * np.pi / self.peq
            cmap = cm.coolwarm

            if self.kappa == 0:
                # Solid-body rotation: uniform shading at middle of colormap
                mid_color = cmap(0.5)
                stellar_disk = Circle((0, 0), 1.0, fc="lightyellow",
                                      ec="k", lw=1.5, zorder=-1)
                ax_star.add_patch(stellar_disk)

                n_pix = 300
                xp = np.linspace(-1, 1, n_pix)
                yp = np.linspace(-1, 1, n_pix)
                XP, YP = np.meshgrid(xp, yp)
                R2 = XP**2 + YP**2
                omega_map = np.where(R2 <= 1.0, 0.5, np.nan)

                norm = Normalize(vmin=0.0, vmax=1.0)
                dr_img = ax_star.imshow(omega_map, extent=[-1, 1, -1, 1],
                                        origin="lower", interpolation="bilinear",
                                        cmap=cmap, norm=norm, alpha=0.3, zorder=0)
                clip_circle = Circle((0, 0), 1.0, transform=ax_star.transData)
                dr_img.set_clip_path(clip_circle)

                ax_star.text(-1.3, 0.0,
                             rf"$\Omega = {omega_eq:.3f}$ [rad/d]",
                             fontsize=label_size - 2, ha="center", va="center",
                             rotation=90, transform=ax_star.transData)
            else:
                omega_min = omega_eq * (1 - self.kappa)
                omega_max = omega_eq
                if omega_min > omega_max:
                    omega_min, omega_max = omega_max, omega_min
                norm = Normalize(vmin=omega_min, vmax=omega_max)

                # Build an image of Omega(lat) on the projected disk
                n_pix = 300
                xp = np.linspace(-1, 1, n_pix)
                yp = np.linspace(-1, 1, n_pix)
                XP, YP = np.meshgrid(xp, yp)
                R2 = XP**2 + YP**2

                CZ = np.sqrt(np.clip(1.0 - R2, 0, None))
                sin_lat = -np.sin(inc) * YP + np.cos(inc) * CZ
                sin_lat = np.clip(sin_lat, -1.0, 1.0)
                lat_map = np.arcsin(sin_lat)
                omega_map = omega_eq * (1 - self.kappa * np.sin(lat_map)**2)

                stellar_disk = Circle((0, 0), 1.0, fc="lightyellow",
                                      ec="k", lw=1.5, zorder=-1)
                ax_star.add_patch(stellar_disk)

                dr_img = ax_star.imshow(omega_map, extent=[-1, 1, -1, 1],
                                        origin="lower", interpolation="bilinear",
                                        cmap=cmap, norm=norm, alpha=0.3, zorder=0)
                clip_circle = Circle((0, 0), 1.0, transform=ax_star.transData)
                dr_img.set_clip_path(clip_circle)

                cbar = fig.colorbar(dr_img, ax=ax_star, fraction=0.046, pad=0.04,
                                    location="left")
                cbar.set_label(r"$\Omega$ [rad/d]", fontsize=label_size)
                cbar.ax.tick_params(labelsize=label_size - 2)
                cbar.ax.text(0.6, 1.02, "faster", transform=cbar.ax.transAxes,
                             ha="center", va="bottom", fontsize=label_size - 2,
                             color="red")
                cbar.ax.text(0.6, -0.02, "slower", transform=cbar.ax.transAxes,
                             ha="center", va="top", fontsize=label_size - 2,
                             color="blue")
        else:
            stellar_disk = Circle((0, 0), 1.0, fc="lightyellow",
                                  ec="k", lw=1.5, zorder=0)
            ax_star.add_patch(stellar_disk)

        # Grid lines on the star
        if show_grid:
            phi_grid = np.linspace(0, 2 * np.pi, 200)
            for lat_deg in [0, 30, 60, -30, -60]:
                lat_r = np.radians(lat_deg)
                gx = (-np.sin(inc) * np.sin(lat_r)
                      + np.cos(inc) * np.cos(lat_r) * np.cos(phi_grid))
                gy = np.cos(lat_r) * np.sin(phi_grid)
                gz = (np.cos(inc) * np.sin(lat_r)
                      + np.sin(inc) * np.cos(lat_r) * np.cos(phi_grid))
                mask = gz > 0
                style = ("k--", 0.6, 0.3) if lat_deg == 0 else ("k-", 0.3, 0.2)
                ax_star.plot(np.where(mask, gy, np.nan),
                             np.where(mask, gx, np.nan),
                             style[0], lw=style[1], alpha=style[2])

        # Rotation axis arrow
        ax_star.annotate(
            "", xy=(0, 1.2), xytext=(0, -0.3),
            arrowprops=dict(arrowstyle="->, head_width=0.08",
                            color="0.5", lw=1.2))

        # Spot patches (updated each frame)
        spot_colors = plt.cm.Set1(np.linspace(0, 1, max(nspot, 1)))
        spot_patches = []
        ghost_patches = []
        for k in range(nspot):
            c = spot_colors[k % len(spot_colors)]
            patch, = ax_star.fill([], [], color=c, alpha=0.85, zorder=2)
            ghost, = ax_star.fill([], [], color=c, alpha=0.15, zorder=1,
                                  linestyle="--", edgecolor=c,
                                  linewidth=0.8)
            spot_patches.append(patch)
            ghost_patches.append(ghost)

        time_text = ax_star.text(0, -1.25, "", fontsize=label_size,
                                 ha="center", va="top")

        # Lightcurve panel (percent dip: 0 = no dip, positive = dimmer)
        dip = (1 - flux) * 100  # percent
        dip_spots = self.dspots * 100  # per-spot percent dip
        dip_max = np.max(dip)
        dip_range = dip_max if dip_max > 0 else 1.0

        ax_lc.set_xlim(t[0], t[-1])
        ax_lc.set_ylim(-0.05 * dip_range,
                        dip_max + 0.1 * dip_range)
        ax_lc.invert_yaxis()
        ax_lc.set_xlabel("Time [days]", fontsize=label_size)
        ax_lc.set_ylabel(r"Flux dip [\%]", fontsize=label_size)
        ax_lc.tick_params(labelsize=label_size - 2)
        ax_lc.minorticks_on()

        # Full lightcurve as faint background
        ax_lc.plot(t, dip, "k-", lw=0.3, alpha=0.15, zorder=0)

        # Traced lightcurve (builds up)
        lc_line, = ax_lc.plot([], [], "k-", lw=1.2, zorder=2)

        # Individual spot contributions
        spot_lc_lines = []
        if show_spots:
            for k in range(nspot):
                c = spot_colors[k % len(spot_colors)]
                ln, = ax_lc.plot([], [], "-", color=c, lw=0.8,
                                 alpha=0.5, zorder=1)
                spot_lc_lines.append(ln)

        # Vertical time marker
        vline = ax_lc.axvline(0, color="C3", lw=1.0, alpha=0.7,
                              ls="--", zorder=3)

        fig.tight_layout()

        # Parameter annotation above the figure
        if show_params:
            param_text = (
                rf"$P_{{\rm eq}}={self.peq:.1f}$ d,  "
                rf"$\kappa={self.kappa:.2f}$,  "
                rf"$I={self.inc_deg:.0f}^\circ$,  "
                rf"$N_{{\rm spot}}={self.nspot}$,  "
                rf"$\alpha_{{\rm max}}={self.alpha_max:.2f}$ rad,  "
                rf"$\ell_{{\rm spot}}={self.lspot:.0f}$ d,  "
                rf"$\tau_{{\rm em}}={self.tem:.1f}$ d,  "
                rf"$\tau_{{\rm dec}}={self.tdec:.1f}$ d"
            )
            fig.text(0.5, 0.99, param_text, fontsize=label_size,
                     ha="center", va="top")
            fig.subplots_adjust(top=0.90)

        # --- Animation ---
        n_frames = int(fps * duration)
        frame_indices = np.linspace(0, n_times - 1,
                                    n_frames).astype(int)
        empty_xy = np.empty((0, 2))

        def update(frame_num):
            idx = frame_indices[frame_num]
            t_now = t[idx]

            # Update spots on the star
            for k in range(nspot):
                alpha_k = spot_alphas[k, idx]
                if alpha_k < 1e-6:
                    spot_patches[k].set_xy(empty_xy)
                    ghost_patches[k].set_xy(empty_xy)
                    continue

                lon_k = spot_longs_t[k, idx]
                lat_k = spot_lats[k]

                fx, fy, bx, by = _projected_spot_patch(
                    lon_k, lat_k, alpha_k, inc)

                if fx is not None and len(fx) >= 3:
                    spot_patches[k].set_xy(
                        np.column_stack([fx, fy]))
                else:
                    spot_patches[k].set_xy(empty_xy)

                if bx is not None and len(bx) >= 3:
                    ghost_patches[k].set_xy(
                        np.column_stack([bx, by]))
                else:
                    ghost_patches[k].set_xy(empty_xy)

            time_text.set_text(rf"$t = {t_now:.1f}$ d")

            # Update lightcurve trace
            lc_line.set_data(t[:idx + 1], dip[:idx + 1])

            # Update individual spot traces
            if show_spots:
                for k in range(nspot):
                    spot_lc_lines[k].set_data(
                        t[:idx + 1], dip_spots[k, :idx + 1])

            vline.set_xdata([t_now])

            return (spot_patches + ghost_patches
                    + [time_text, lc_line, vline]
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

        # Save the last frame as a static image
        if save_last_frame is not None:
            update(n_frames - 1)
            fig.savefig(save_last_frame, dpi=dpi, bbox_inches="tight")
            print(f"Last frame saved to {save_last_frame}")

        plt.close(fig)

        return anim

    def animate_butterfly(self, fps=30, duration=10.0, outfile=None,
                          dpi=150, show_spots=True, show_grid=True,
                          show_params=True, figsize=(18, 5.5),
                          save_last_frame=None, show_dr=True,
                          label_size=18):
        """
        Animate the starspot evolution with three panels: a 2D projection
        of the rotating star (left), the lightcurve (center), and a
        butterfly diagram of spot latitude vs. time (right).

        Parameters
        ----------
        fps : int
            Frames per second (default 30).
        duration : float
            Animation duration in seconds (default 10).
        outfile : str or None
            Output file path (.mp4 or .gif). If None, returns the
            animation object without saving.
        dpi : int
            Resolution (default 150).
        show_spots : bool
            If True, show individual spot contributions on the
            lightcurve panel (default True).
        show_grid : bool
            If True, draw latitude/longitude grid on the star
            (default True).
        show_params : bool
            If True, show parameter annotation above the figure
            (default True).
        figsize : tuple
            Figure size (default (18, 5.5)).
        save_last_frame : str or None
            If provided, save the last frame of the animation as a
            static image to this file path (e.g. "frame.png").
        show_dr : bool
            If True, color the stellar disk by latitude-dependent
            rotation frequency and display a colorbar (default True).
        label_size : int or float
            Font size for all labels, tick marks, and text in the
            plot (default 18).

        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            The animation object.
        """
        import matplotlib.animation as animation
        from matplotlib.patches import Circle

        t = self.t
        flux = self.flux + self.dlimb
        inc = self.inc
        nspot = self.nspot
        n_times = len(t)

        spot_longs = np.atleast_1d(self.long)
        spot_lats = np.atleast_1d(self.lat)
        spot_tmaxs = self.tmax

        # Precompute spot alphas and longitudes for all times
        t_jax = jnp.array(t)
        spot_alphas = np.zeros((nspot, n_times))
        spot_longs_t = np.zeros((nspot, n_times))
        for k in range(nspot):
            if self.grow:
                spot_alphas[k] = np.asarray(_alphak(
                    t_jax, spot_tmaxs[k], self.lspot,
                    self.tem, self.tdec, self.alpha_max))
            else:
                spot_alphas[k] = self.alpha_max
            if self.rotate:
                _, longk_t = _betak(
                    t_jax, spot_longs[k], spot_lats[k], spot_tmaxs[k],
                    self.peq, self.kappa, self.inc)
                spot_longs_t[k] = np.asarray(longk_t)
            else:
                spot_longs_t[k] = spot_longs[k]

        # --- Set up figure ---
        import matplotlib.gridspec as mgs
        fig = plt.figure(figsize=figsize)
        gs = mgs.GridSpec(1, 4, figure=fig,
                          width_ratios=[1, 1.6, 1.6, 0.4],
                          wspace=0.05)
        ax_star = fig.add_subplot(gs[0, 0])
        ax_lc = fig.add_subplot(gs[0, 1])
        ax_bf = fig.add_subplot(gs[0, 2])
        ax_hist = fig.add_subplot(gs[0, 3], sharey=ax_bf)

        # =====================================================================
        # Star panel (left) -- identical to animate_lightcurve
        # =====================================================================
        ax_star.set_aspect("equal")
        ax_star.set_xlim(-1.35, 1.35)
        ax_star.set_ylim(-1.35, 1.35)
        ax_star.set_axis_off()

        if show_dr:
            from matplotlib.colors import Normalize
            import matplotlib.cm as cm

            omega_eq = 2 * np.pi / self.peq
            cmap = cm.coolwarm

            if self.kappa == 0:
                mid_color = cmap(0.5)
                stellar_disk = Circle((0, 0), 1.0, fc="lightyellow",
                                      ec="k", lw=1.5, zorder=-1)
                ax_star.add_patch(stellar_disk)

                n_pix = 300
                xp = np.linspace(-1, 1, n_pix)
                yp = np.linspace(-1, 1, n_pix)
                XP, YP = np.meshgrid(xp, yp)
                R2 = XP**2 + YP**2
                omega_map = np.where(R2 <= 1.0, 0.5, np.nan)

                norm = Normalize(vmin=0.0, vmax=1.0)
                dr_img = ax_star.imshow(
                    omega_map, extent=[-1, 1, -1, 1],
                    origin="lower", interpolation="bilinear",
                    cmap=cmap, norm=norm, alpha=0.3, zorder=0)
                clip_circle = Circle((0, 0), 1.0,
                                     transform=ax_star.transData)
                dr_img.set_clip_path(clip_circle)

                ax_star.text(-1.3, 0.0,
                             rf"$\Omega = {omega_eq:.3f}$ [rad/d]",
                             fontsize=label_size - 2, ha="center",
                             va="center", rotation=90,
                             transform=ax_star.transData)
            else:
                omega_min = omega_eq * (1 - self.kappa)
                omega_max = omega_eq
                if omega_min > omega_max:
                    omega_min, omega_max = omega_max, omega_min
                norm = Normalize(vmin=omega_min, vmax=omega_max)

                n_pix = 300
                xp = np.linspace(-1, 1, n_pix)
                yp = np.linspace(-1, 1, n_pix)
                XP, YP = np.meshgrid(xp, yp)
                R2 = XP**2 + YP**2

                CZ = np.sqrt(np.clip(1.0 - R2, 0, None))
                sin_lat = (-np.sin(inc) * YP
                           + np.cos(inc) * CZ)
                sin_lat = np.clip(sin_lat, -1.0, 1.0)
                lat_map = np.arcsin(sin_lat)
                omega_map = omega_eq * (
                    1 - self.kappa * np.sin(lat_map)**2)

                stellar_disk = Circle((0, 0), 1.0, fc="lightyellow",
                                      ec="k", lw=1.5, zorder=-1)
                ax_star.add_patch(stellar_disk)

                dr_img = ax_star.imshow(
                    omega_map, extent=[-1, 1, -1, 1],
                    origin="lower", interpolation="bilinear",
                    cmap=cmap, norm=norm, alpha=0.3, zorder=0)
                clip_circle = Circle((0, 0), 1.0,
                                     transform=ax_star.transData)
                dr_img.set_clip_path(clip_circle)

                cbar = fig.colorbar(dr_img, ax=ax_star,
                                    fraction=0.046, pad=0.04,
                                    location="left")
                cbar.set_label(r"$\Omega$ [rad/d]",
                               fontsize=label_size)
                cbar.ax.tick_params(labelsize=label_size - 2)
                cbar.ax.text(0.6, 1.02, "faster",
                             transform=cbar.ax.transAxes,
                             ha="center", va="bottom",
                             fontsize=label_size - 2, color="red")
                cbar.ax.text(0.6, -0.02, "slower",
                             transform=cbar.ax.transAxes,
                             ha="center", va="top",
                             fontsize=label_size - 2, color="blue")
        else:
            stellar_disk = Circle((0, 0), 1.0, fc="lightyellow",
                                  ec="k", lw=1.5, zorder=0)
            ax_star.add_patch(stellar_disk)

        # Grid lines on the star
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
                ax_star.plot(np.where(mask, gy, np.nan),
                             np.where(mask, gx, np.nan),
                             style[0], lw=style[1], alpha=style[2])

        # Rotation axis arrow
        ax_star.annotate(
            "", xy=(0, 1.2), xytext=(0, -0.3),
            arrowprops=dict(arrowstyle="->, head_width=0.08",
                            color="0.5", lw=1.2))

        # Spot patches (updated each frame)
        spot_colors = plt.cm.Set1(np.linspace(0, 1, max(nspot, 1)))
        spot_patches = []
        ghost_patches = []
        for k in range(nspot):
            c = spot_colors[k % len(spot_colors)]
            patch, = ax_star.fill([], [], color=c, alpha=0.85,
                                  zorder=2)
            ghost, = ax_star.fill([], [], color=c, alpha=0.15,
                                  zorder=1, linestyle="--",
                                  edgecolor=c, linewidth=0.8)
            spot_patches.append(patch)
            ghost_patches.append(ghost)

        time_text = ax_star.text(0, -1.25, "", fontsize=label_size,
                                 ha="center", va="top")

        # =====================================================================
        # Lightcurve panel (center) -- identical to animate_lightcurve
        # =====================================================================
        dip = (1 - flux) * 100
        dip_spots = self.dspots * 100
        dip_max = np.max(dip)
        dip_range = dip_max if dip_max > 0 else 1.0

        ax_lc.set_xlim(t[0], t[-1])
        ax_lc.set_ylim(-0.05 * dip_range,
                        dip_max + 0.1 * dip_range)
        ax_lc.invert_yaxis()
        ax_lc.set_xlabel("Time [days]", fontsize=label_size)
        ax_lc.set_ylabel(r"Flux dip [\%]", fontsize=label_size)
        ax_lc.tick_params(labelsize=label_size - 2)
        ax_lc.minorticks_on()

        # Full lightcurve as faint background
        ax_lc.plot(t, dip, "k-", lw=0.3, alpha=0.15, zorder=0)

        # Traced lightcurve (builds up)
        lc_line, = ax_lc.plot([], [], "k-", lw=1.2, zorder=2)

        # Individual spot contributions
        spot_lc_lines = []
        if show_spots:
            for k in range(nspot):
                c = spot_colors[k % len(spot_colors)]
                ln, = ax_lc.plot([], [], "-", color=c, lw=0.8,
                                 alpha=0.5, zorder=1)
                spot_lc_lines.append(ln)

        # Vertical time marker
        vline_lc = ax_lc.axvline(0, color="C3", lw=1.0, alpha=0.7,
                                 ls="--", zorder=3)

        # =====================================================================
        # Butterfly diagram panel (3rd)
        # =====================================================================
        spot_lats_deg = np.degrees(spot_lats)

        ax_bf.set_xlim(t[0], t[-1])
        ax_bf.set_ylim(-90, 90)
        ax_bf.set_xlabel("Time [days]", fontsize=label_size)
        ax_bf.set_ylabel(r"Latitude [$^\circ$]", fontsize=label_size)
        ax_bf.set_title("Butterfly Diagram", fontsize=label_size)
        ax_bf.tick_params(labelsize=label_size - 2)
        ax_bf.minorticks_on()
        ax_bf.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax_bf.set_yticks([-90, -60, -30, 0, 30, 60, 90])

        # Faint background: full lifetime extents
        for k in range(nspot):
            c = spot_colors[k % len(spot_colors)]
            active_mask = spot_alphas[k] > 1e-6
            if np.any(active_mask):
                active_times = t[active_mask]
                ax_bf.plot(active_times,
                           np.full_like(active_times, spot_lats_deg[k]),
                           "-", color=c, lw=1.0, alpha=0.1, zorder=0)

        # Precompute marker sizes for butterfly trace: scale with alpha
        # Size in points^2 for scatter; map alpha/alpha_max -> area
        min_size = 4
        max_size = 80
        bf_sizes = min_size + (max_size - min_size) * (
            spot_alphas / self.alpha_max)

        # Use one scatter per spot so colors match the star panel
        bf_scatters = []
        for k in range(nspot):
            c = spot_colors[k % len(spot_colors)]
            sc = ax_bf.scatter([], [], s=[], color=c, alpha=0.7,
                               zorder=1, edgecolors="none")
            bf_scatters.append(sc)

        # Current-time marker (ring highlight)
        bf_now_scatters = []
        for k in range(nspot):
            c = spot_colors[k % len(spot_colors)]
            sc, = ax_bf.plot([], [], "o", color=c, markersize=8,
                             alpha=0.9, zorder=2, markeredgecolor="k",
                             markeredgewidth=0.5)
            bf_now_scatters.append(sc)

        vline_bf = ax_bf.axvline(0, color="C3", lw=1.0, alpha=0.7,
                                 ls="--", zorder=3)

        # =====================================================================
        # Active latitudes histogram panel (4th, rightmost)
        # =====================================================================
        ax_hist.set_title("Active\nLatitudes", fontsize=label_size - 2)
        ax_hist.tick_params(labelsize=label_size - 2)
        ax_hist.set_xlim(0, 1.05)
        ax_hist.set_xlabel(r"$\alpha / \alpha_{\rm max}$",
                           fontsize=label_size - 2)
        plt.setp(ax_hist.get_yticklabels(), visible=False)
        ax_hist.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax_hist.minorticks_on()

        # One horizontal bar per spot, updated each frame
        hist_bars = []
        for k in range(nspot):
            c = spot_colors[k % len(spot_colors)]
            bar = ax_hist.barh(spot_lats_deg[k], 0, height=6,
                               color=c, alpha=0.8, edgecolor="k",
                               linewidth=0.3, zorder=1)
            hist_bars.append(bar[0])

        fig.subplots_adjust(wspace=0.35, left=0.05, right=0.97)

        # Parameter annotation above the figure
        if show_params:
            param_text = (
                rf"$P_{{\rm eq}}={self.peq:.1f}$ d,  "
                rf"$\kappa={self.kappa:.2f}$,  "
                rf"$I={self.inc_deg:.0f}^\circ$,  "
                rf"$N_{{\rm spot}}={self.nspot}$,  "
                rf"$\alpha_{{\rm max}}={self.alpha_max:.2f}$ rad,  "
                rf"$\ell_{{\rm spot}}={self.lspot:.0f}$ d,  "
                rf"$\tau_{{\rm em}}={self.tem:.1f}$ d,  "
                rf"$\tau_{{\rm dec}}={self.tdec:.1f}$ d"
            )
            fig.text(0.5, 0.99, param_text, fontsize=label_size,
                     ha="center", va="top")
            fig.subplots_adjust(top=0.90)

        # --- Animation ---
        n_frames = int(fps * duration)
        frame_indices = np.linspace(0, n_times - 1,
                                    n_frames).astype(int)
        empty_xy = np.empty((0, 2))

        def update(frame_num):
            idx = frame_indices[frame_num]
            t_now = t[idx]

            # Update spots on the star
            for k in range(nspot):
                alpha_k = spot_alphas[k, idx]
                if alpha_k < 1e-6:
                    spot_patches[k].set_xy(empty_xy)
                    ghost_patches[k].set_xy(empty_xy)
                    continue

                lon_k = spot_longs_t[k, idx]
                lat_k = spot_lats[k]

                fx, fy, bx, by = _projected_spot_patch(
                    lon_k, lat_k, alpha_k, inc)

                if fx is not None and len(fx) >= 3:
                    spot_patches[k].set_xy(
                        np.column_stack([fx, fy]))
                else:
                    spot_patches[k].set_xy(empty_xy)

                if bx is not None and len(bx) >= 3:
                    ghost_patches[k].set_xy(
                        np.column_stack([bx, by]))
                else:
                    ghost_patches[k].set_xy(empty_xy)

            time_text.set_text(rf"$t = {t_now:.1f}$ d")

            # Update lightcurve trace
            lc_line.set_data(t[:idx + 1], dip[:idx + 1])

            # Update individual spot traces
            if show_spots:
                for k in range(nspot):
                    spot_lc_lines[k].set_data(
                        t[:idx + 1], dip_spots[k, :idx + 1])

            vline_lc.set_xdata([t_now])

            # Update butterfly diagram
            for k in range(nspot):
                # Scatter trace with sizes matching spot size
                active_mask = spot_alphas[k, :idx + 1] > 1e-6
                if np.any(active_mask):
                    t_active = t[:idx + 1][active_mask]
                    lat_active = np.full(np.sum(active_mask),
                                         spot_lats_deg[k])
                    s_active = bf_sizes[k, :idx + 1][active_mask]
                    bf_scatters[k].set_offsets(
                        np.column_stack([t_active, lat_active]))
                    bf_scatters[k].set_sizes(s_active)
                else:
                    bf_scatters[k].set_offsets(np.empty((0, 2)))
                    bf_scatters[k].set_sizes([])

                # Current-time ring marker
                alpha_k = spot_alphas[k, idx]
                if alpha_k > 1e-6:
                    bf_now_scatters[k].set_data([t_now],
                                                [spot_lats_deg[k]])
                    ms = 4 + 12 * (alpha_k / self.alpha_max)
                    bf_now_scatters[k].set_markersize(ms)
                else:
                    bf_now_scatters[k].set_data([], [])

            vline_bf.set_xdata([t_now])

            # Update active latitudes histogram
            for k in range(nspot):
                alpha_k = spot_alphas[k, idx]
                hist_bars[k].set_width(alpha_k / self.alpha_max)

            return (spot_patches + ghost_patches
                    + [time_text, lc_line, vline_lc, vline_bf]
                    + spot_lc_lines + bf_scatters
                    + bf_now_scatters + hist_bars)

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

        # Save the last frame as a static image
        if save_last_frame is not None:
            update(n_frames - 1)
            fig.savefig(save_last_frame, dpi=dpi, bbox_inches="tight")
            print(f"Last frame saved to {save_last_frame}")

        plt.close(fig)

        return anim
