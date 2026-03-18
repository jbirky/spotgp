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
except ImportError:
    from params import resolve_hparam

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


def _zeta(x):
    """Calculate zeta(x) for spot limb darkening."""
    return (np.cos(x) * np.heaviside(x, 1.0) * np.heaviside(np.pi/2 - x, 1.0)
            + np.heaviside(-x, 1.0))


def _alphak(teval, tmaxk, lspot, tem, tdec, alpha_max):
    """Compute spot angular size evolution (vectorized over time)."""
    dt1 = teval - tmaxk + lspot/2 + tem
    dt2 = teval - tmaxk + lspot/2
    dt3 = teval - tmaxk - lspot/2
    dt4 = teval - tmaxk - lspot/2 - tdec

    alphak  = (dt1 * np.heaviside(dt1, 1.0) - dt2 * np.heaviside(dt2, 1.0)) / tem
    alphak += -(dt3 * np.heaviside(dt3, 1.0) - dt4 * np.heaviside(dt4, 1.0)) / tdec
    alphak *= alpha_max

    return alphak


def _betak(teval, longk, latk, tmaxk, peq, kappa, inc):
    """Compute spot angle from disk center (vectorized over time)."""
    longk_t = longk + 2*np.pi/peq * (1 - kappa * np.sin(latk)**2) * (teval - tmaxk)

    cosb  = np.cos(inc) * np.sin(latk)
    cosb += np.sin(inc) * np.cos(latk) * np.cos(longk_t)
    betak_t = np.arccos(np.clip(cosb, -1.0, 1.0))

    return betak_t, longk_t


def _dflux_single_spot(teval, longk, latk, tmaxk,
                       peq, kappa, inc, lspot, tem, tdec, alpha_max, fspot):
    """
    Compute flux deficit for a single spot over all time steps.
    Fully vectorized over time using NumPy.
    """
    betak_t, _ = _betak(teval, longk, latk, tmaxk, peq, kappa, inc)
    alphak_t = _alphak(teval, tmaxk, lspot, tem, tdec, alpha_max)

    cosa = np.cos(alphak_t)
    sina = np.sin(alphak_t)
    cosb = np.cos(betak_t)
    sinb = np.sin(betak_t)

    # Avoid division by zero with small epsilon
    eps = 1e-30
    cota = cosa / (sina + eps)
    cscb = 1.0 / (sinb + eps)
    cotb = cosb / (sinb + eps)

    # Clamp argument for arccos to [-1, 1]
    arg1 = np.clip(cosa * cscb, -1.0, 1.0)
    arg2 = np.clip(-cota * cotb, -1.0, 1.0)
    sqrt_arg = np.clip(1 - cosa**2 * cscb**2, 0.0, None)

    Ak  = np.arccos(arg1)
    Ak += cosb * sina**2 * np.arccos(arg2)
    Ak -= cosa * sinb * np.sqrt(sqrt_arg)

    # Simple spot limb darkening factor (no limb darkening case)
    factor = 1.0 - fspot

    dspot = Ak / np.pi * factor

    # Zero out contributions where spot has zero size
    dspot = np.where(alphak_t > 1e-15, dspot, 0.0)

    return dspot


class LightcurveModel(object):
    """
    NumPy star with spots and its lightcurve.

    Same interface as the JAX version but uses NumPy loops for spot
    computation instead of vmap.

    Args:
        peq (float): Equatorial period of the star.
        kappa (float): Differential rotation shear.
        inc (float): Inclination of the star.
        nspot (int): Number of spots.
        tau (float, optional): Timescale for both emergence and decay of the spots. Defaults to None.
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
    def __init__(self, peq=4.0, kappa=0.0, inc=np.pi/2, nspot=10,
                 tau=None, tem=2, tdec=2, alpha_max=0.1, fspot=0, lspot=5,
                 long=[0, 2*np.pi], lat=[-np.pi/2, np.pi/2],
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
        if tau is not None:
            self.tem = tau
            self.tdec = tau
        else:
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

        # compute lightcurve
        self.flux = self.Flux(self.t)

    @classmethod
    def from_hparam(cls, hparam: dict, nspot: int, **kwargs):
        """Construct a LightcurveModel from a GPSolver-compatible hparam dict.

        Accepts the same raw hparam dict that GPSolver/AnalyticKernel take,
        including all amplitude modes (sigma_k, nspot_rate, or nspot), and
        both symmetric (tau) and asymmetric (tau_em + tau_dec) envelopes.
        This removes the need to manually decompose the dict in scripts.

        Parameters
        ----------
        hparam : dict
            Raw hyperparameter dict.  Must contain peq, kappa, inc, lspot,
            an envelope timescale, and an amplitude specification.
        nspot : int
            Total number of spots to simulate (distinct from nspot_rate).
        **kwargs
            Forwarded to LightcurveModel.__init__ (e.g. tsim, tsamp, lat, long).

        Returns
        -------
        LightcurveModel
        """
        p = resolve_hparam(hparam)
        tau_em  = p.get("tau_em",  p["tau"])
        tau_dec = p.get("tau_dec", p["tau"])
        alpha_max = p.get("alpha_max", kwargs.pop("alpha_max", 0.1))
        fspot     = p.get("fspot",     kwargs.pop("fspot", 0.0))
        return cls(
            peq=p["peq"], kappa=p["kappa"], inc=p["inc"], nspot=nspot,
            tem=tau_em, tdec=tau_dec,
            alpha_max=alpha_max, fspot=fspot, lspot=p["lspot"],
            **kwargs,
        )

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
        Compute the full lightcurve by looping over all spots.
        """
        teval = np.asarray(teval)
        long_arr = np.atleast_1d(self.long)
        lat_arr = np.atleast_1d(self.lat)
        tmax_arr = self.tmax

        dspots = np.zeros((self.nspot, len(teval)))
        for k in range(self.nspot):
            dspots[k] = _dflux_single_spot(
                teval, long_arr[k], lat_arr[k], tmax_arr[k],
                self.peq, self.kappa, self.inc,
                self.lspot, self.tem, self.tdec, self.alpha_max, self.fspot
            )

        self.dspots = dspots

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
        spot_alphas = np.zeros((nspot, n_times))
        spot_longs_t = np.zeros((nspot, n_times))
        for k in range(nspot):
            spot_alphas[k] = _alphak(
                t, spot_tmaxs[k], self.lspot,
                self.tem, self.tdec, self.alpha_max)
            _, longk_t = _betak(
                t, spot_longs[k], spot_lats[k], spot_tmaxs[k],
                self.peq, self.kappa, self.inc)
            spot_longs_t[k] = longk_t

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
            omega_min = omega_eq * (1 - self.kappa)
            omega_max = omega_eq
            if omega_min > omega_max:
                omega_min, omega_max = omega_max, omega_min
            # Small padding so uniform case still renders
            if omega_max - omega_min < 1e-12:
                omega_max = omega_min + 1e-12
            norm = Normalize(vmin=omega_min, vmax=omega_max)
            cmap = cm.coolwarm

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

            # Disk background
            stellar_disk = Circle((0, 0), 1.0, fc="lightyellow",
                                  ec="k", lw=1.5, zorder=-1)
            ax_star.add_patch(stellar_disk)

            # Render DR shading clipped to the stellar disk
            dr_img = ax_star.imshow(omega_map, extent=[-1, 1, -1, 1],
                                    origin="lower", interpolation="bilinear",
                                    cmap=cmap, norm=norm, alpha=0.3, zorder=0)
            clip_circle = Circle((0, 0), 1.0, transform=ax_star.transData)
            dr_img.set_clip_path(clip_circle)

            # Colorbar on the left side
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
