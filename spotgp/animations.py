"""Animation methods for LightcurveModel, mixed in as AnimationMixin."""

import logging

import jax.numpy as jnp
import numpy as np

logger = logging.getLogger("spotgp")


def _get_lightcurve_helpers():
    from .lightcurve import _alphak, _betak, _projected_spot_patch
    return _alphak, _betak, _projected_spot_patch


class AnimationMixin:
    """Lightcurve and butterfly animation methods."""

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
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.patches import Circle
        _alphak, _betak, _projected_spot_patch = _get_lightcurve_helpers()

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

            logger.info("Rendering %d frames to %s...", n_frames, outfile)
            anim.save(outfile, writer=writer, dpi=dpi)
            logger.info("Done.")

        # Save the last frame as a static image
        if save_last_frame is not None:
            update(n_frames - 1)
            fig.savefig(save_last_frame, dpi=dpi, bbox_inches="tight")
            logger.debug("Last frame saved to %s", save_last_frame)

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
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.patches import Circle
        _alphak, _betak, _projected_spot_patch = _get_lightcurve_helpers()

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

            logger.info("Rendering %d frames to %s...", n_frames, outfile)
            anim.save(outfile, writer=writer, dpi=dpi)
            logger.info("Done.")

        # Save the last frame as a static image
        if save_last_frame is not None:
            update(n_frames - 1)
            fig.savefig(save_last_frame, dpi=dpi, bbox_inches="tight")
            logger.debug("Last frame saved to %s", save_last_frame)

        plt.close(fig)

        return anim
