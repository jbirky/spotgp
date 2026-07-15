import logging
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from .params import resolve_hparam
from .spot_model import SpotEvolutionModel
from .animations import AnimationMixin

__all__ = ["LightcurveModel", "compute_sigmak"]

logger = logging.getLogger("spotgp")


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


class LightcurveModel(AnimationMixin):
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
        import matplotlib.pyplot as plt
        from matplotlib import rc
        rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        rc('text', usetex=True)
        rc('figure', facecolor='w')
        rc('xtick', labelsize=20)
        rc('ytick', labelsize=20)

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

