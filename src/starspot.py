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

__all__ = ["LightcurveModel"]


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
        tem (float, optional): Emergence timescale of the spots. Defaults to 1.
        tdec (float, optional): Decay timescale of the spots. Defaults to 2.
        alpha_max (float, optional): Maximum angular area of the spots. Defaults to 0.1.
        fspot (float, optional): Spot contrast fraction. Defaults to 0.
        lspot (float, optional): Spot lifetime. Defaults to 2.
        long (list, optional): Range of spot longitudes. Defaults to [0, 2*pi].
        lat (list, optional): Range of spot latitudes. Defaults to [0, pi].
        tsim (float, optional): End simulation time. Defaults to 28.
        tsamp (float, optional): Sampling cadence. Defaults to 0.02.
        limb_darkening (bool, optional): Flag to enable limb darkening. Defaults to False.
    """
    def __init__(self, peq, kappa, inc, nspot,
                 tem=1, tdec=2, alpha_max=0.1, fspot=0, lspot=2,
                 long=[0, 2*np.pi], lat=[0, np.pi],
                 tsim=28, tsamp=0.02, limb_darkening=False):

        # simulation parameters
        self.tsim = tsim
        self.tsamp = tsamp
        self.t = np.arange(0, self.tsim, self.tsamp)

        # star properties
        self.peq = peq
        self.kappa = kappa
        self.inc = inc
        self.inc_deg = inc * 180/np.pi
        self.nspot = int(nspot)

        # spot properties (scalars)
        self.tem = tem
        self.tdec = tdec
        self.alpha_max = alpha_max
        self.fspot = fspot
        self.lspot = lspot

        self.long = self._assign_property(long)
        self.lat = self._assign_property(lat)
        self.tmax = np.random.uniform(0, self.tsim, self.nspot)

        # limb darkening
        self.limb_darkening = limb_darkening
        self.limbc = np.array([0.3999, 0.4269, -0.0227, -0.0839])
        self.limbd = self.limbc

        # compute lightcurve using JAX
        self.flux = self.Flux(self.t)

    def _assign_property(self, var):
        if isinstance(var, (int, float)):
            return var
        elif isinstance(var, (list, np.ndarray)):
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
        dspots = _dflux_all_spots(
            teval_jax, long_jax, lat_jax, tmax_jax,
            self.peq, self.kappa, self.inc,
            self.lspot, self.tem, self.tdec, self.alpha_max, self.fspot
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
        fig = plt.figure(figsize=[16, 6])
        if show_spots:
            for ii in range(self.nspot):
                plt.plot(self.t, 1-self.dspots[ii], alpha=0.5)
        plt.plot(self.t, flux, color="k")

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
        plt.ylabel("Flux", fontsize=24)
        plt.ylim(min(flux)-2e-3, 1+1e-3)
        plt.xlim(self.t[0], self.t[-1])
        plt.minorticks_on()
        plt.ticklabel_format(axis='both', style='', useOffset=False)
        plt.close()

        return fig
