"""Probabilistic Graphical Model visualization for spotgp solvers.

Uses the ``daft`` package to render plate diagrams showing free parameters,
observed data, deterministic intermediate quantities, and their dependencies.
Install with ``pip install daft``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["PGModelVis"]

# ---------------------------------------------------------------------------
# Parameter categorization
# ---------------------------------------------------------------------------

ROTATION_KEYS = {"peq", "kappa", "inc"}
ENVELOPE_KEYS = {"lspot", "tau_spot", "tau_em", "tau_dec", "sigma_sn", "n_sn"}
LATITUDE_KEYS = {"lat_min", "lat_max"}
AMPLITUDE_KEYS = {"sigma_k", "nspot_rate", "fspot", "alpha_max", "nspot"}
MULTIBAND_KEYS = {"T_spot"}
NOISE_KEYS = {"sigma_n"}

# ---------------------------------------------------------------------------
# LaTeX labels
# ---------------------------------------------------------------------------

PARAM_LABELS = {
    "peq":        r"$P_{\rm eq}$",
    "kappa":      r"$\kappa$",
    "inc":        r"$i$",
    "lspot":      r"$\ell_{\rm spot}$",
    "tau_spot":   r"$\tau_{\rm spot}$",
    "tau_em":     r"$\tau_{\rm em}$",
    "tau_dec":    r"$\tau_{\rm dec}$",
    "sigma_sn":   r"$\sigma_{\rm sn}$",
    "n_sn":       r"$n_{\rm sn}$",
    "sigma_k":    r"$\sigma_k$",
    "sigma_n":    r"$\sigma_n$",
    "lat_min":    r"$\phi_{\rm min}$",
    "lat_max":    r"$\phi_{\rm max}$",
    "T_spot":     r"$T_{\rm spot}$",
    "T_phot":     r"$T_{\rm phot}$",
    "nspot_rate": r"$\dot{n}_{\rm spot}$",
    "fspot":      r"$f_{\rm spot}$",
    "alpha_max":  r"$\alpha_{\rm max}$",
    "nspot":      r"$n_{\rm spot}$",
}


PARAM_DESCRIPTIONS = {
    "peq":        "equatorial rotation period [days]",
    "kappa":      "differential rotation shear",
    "inc":        "stellar inclination [rad]",
    "lspot":      "spot angular extent [deg]",
    "tau_spot":   "spot rise/decay timescale [days]",
    "tau_em":     "spot emergence timescale [days]",
    "tau_dec":    "spot decay timescale [days]",
    "sigma_sn":   "skew-normal width [days]",
    "n_sn":       "skew-normal skewness",
    "sigma_k":    "kernel amplitude",
    "sigma_n":    "white noise std. dev.",
    "lat_min":    "minimum active latitude [rad]",
    "lat_max":    "maximum active latitude [rad]",
    "T_spot":     "spot temperature [K]",
    "T_phot":     "photosphere temperature [K] (fixed)",
    "nspot_rate": "spot emergence rate [1/day]",
    "fspot":      "spot filling factor",
    "alpha_max":  "max. spot contrast",
    "nspot":      "number of spots",
}


def _strip_log_prefix(key: str) -> str:
    """Remove ``log_`` prefix so categorization works for log-space keys."""
    return key[4:] if key.startswith("log_") else key


# ---------------------------------------------------------------------------
# PGModelVis
# ---------------------------------------------------------------------------

class PGModelVis:
    """Probabilistic Graphical Model diagram for a GPSolver or MultiBandGPSolver.

    Introspects the solver's free parameters, model components, and data to
    build a plate diagram with:

    * **Open circles** — latent (free) parameters
    * **Shaded circles** — observed data (*y*)
    * **Small dots** — fixed / known inputs (*x*, measurement uncertainties)
    * **Plates** — repeated structure (observations, bands)

    Parameters
    ----------
    solver : GPSolver or MultiBandGPSolver
        A configured solver whose ``param_keys``, ``spot_model``, and data
        attributes will be inspected.
    """

    def __init__(self, solver):
        self.solver = solver
        self.is_multiband = hasattr(solver, "T_phot")
        self._physical_keys = [_strip_log_prefix(k) for k in solver.param_keys]
        self._categorize()

    # ----- parameter grouping ------------------------------------------------

    def _categorize(self):
        keys = self._physical_keys
        self.rotation_params = [k for k in keys if k in ROTATION_KEYS]
        self.envelope_params = [k for k in keys if k in ENVELOPE_KEYS]
        self.latitude_params = [k for k in keys if k in LATITUDE_KEYS]
        self.amplitude_params = [k for k in keys if k in AMPLITUDE_KEYS]
        self.multiband_params = [k for k in keys if k in MULTIBAND_KEYS]
        self.noise_params = [k for k in keys if k in NOISE_KEYS]

    def _build_groups(self):
        """Return an ordered list of parameter-group dicts."""
        groups = []
        if self.rotation_params:
            groups.append(dict(
                name="rotation", params=self.rotation_params,
                intermediate="V", inter_label=r"$V(\phi)$",
                target="K",
            ))
        if self.envelope_params:
            groups.append(dict(
                name="envelope", params=self.envelope_params,
                intermediate="R_Gamma", inter_label=r"$R_\Gamma(\tau)$",
                target="K",
            ))
        if self.latitude_params:
            groups.append(dict(
                name="latitude", params=self.latitude_params,
                intermediate="p_phi", inter_label=r"$p(\phi)$",
                target="K",
            ))
        if self.amplitude_params:
            groups.append(dict(
                name="amplitude", params=self.amplitude_params,
                intermediate=None, target="K",
            ))
        if self.multiband_params:
            groups.append(dict(
                name="multiband", params=self.multiband_params,
                intermediate="c_lambda", inter_label=r"$c(\lambda)$",
                target="K",
            ))
        if self.noise_params:
            groups.append(dict(
                name="noise", params=self.noise_params,
                intermediate=None, target="y_i",
            ))
        return groups

    # ----- layout ------------------------------------------------------------

    def _compute_layout(self, groups):
        """Compute ``{node_name: (x, y)}`` positions for every element."""
        spacing = 1.5
        gap = 2.5

        param_y = 4
        inter_y = 3
        kernel_y = 2
        data_y = 1

        positions = {}
        group_meta = []

        x_cursor = 1.0
        for g in groups:
            params = g["params"]
            xs = [x_cursor + i * spacing for i in range(len(params))]
            for p, xp in zip(params, xs):
                positions[p] = (xp, param_y)
            center = sum(xs) / len(xs)
            group_meta.append({**g, "center_x": center})
            x_cursor = xs[-1] + gap

        # intermediate deterministic nodes
        for gm in group_meta:
            inter = gm.get("intermediate")
            if inter:
                positions[inter] = (gm["center_x"], inter_y)

        # T_phot (fixed, only for multi-band) placed next to T_spot
        if self.is_multiband and "c_lambda" in positions:
            cx = positions["c_lambda"][0]
            positions["T_phot"] = (cx + spacing, param_y)

        # center of the full layout
        all_x = [p[0] for p in positions.values()]
        center_x = (min(all_x) + max(all_x)) / 2.0

        # kernel and data layer
        positions["K"] = (center_x, kernel_y)
        positions["x_i"] = (center_x - 1.5, data_y)
        positions["y_i"] = (center_x, data_y)
        positions["sigma_obs"] = (center_x + 1.5, data_y)

        return positions, group_meta

    # ----- rendering ---------------------------------------------------------

    def render(self, dpi=150, node_scale=1.2, font_size=11,
               show_legend=False):
        """Render the PGM and return the matplotlib Figure.

        Parameters
        ----------
        dpi : int
            Resolution of the rendered figure.
        node_scale : float
            Scale factor for node radius.
        font_size : int
            Font size for node labels.
        show_legend : bool
            If True, draw a parameter legend below the diagram mapping
            each symbol to a short description.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        try:
            import daft
        except ImportError:
            raise ImportError(
                "The 'daft' package is required for PGM visualization. "
                "Install it with: pip install daft"
            )

        groups = self._build_groups()
        positions, group_meta = self._compute_layout(groups)

        # figure extent
        all_x = [p[0] for p in positions.values()]
        all_y = [p[1] for p in positions.values()]
        margin = 1.5
        x_lo, x_hi = min(all_x) - margin, max(all_x) + margin
        y_lo, y_hi = min(all_y) - 1.0, max(all_y) + 0.75

        pgm = daft.PGM(
            shape=(x_hi - x_lo, y_hi - y_lo),
            origin=(x_lo, y_lo),
            node_unit=1.4,
            grid_unit=2.5,
            observed_style="shaded",
        )

        sc = node_scale
        fs = font_size

        # --- free-parameter nodes (latent, open circles) ---
        for param in self._physical_keys:
            if param in positions:
                x, y = positions[param]
                label = PARAM_LABELS.get(param, param)
                pgm.add_node(param, label, x, y, scale=sc, fontsize=fs)

        # --- fixed T_phot node (multi-band only) ---
        if "T_phot" in positions:
            x, y = positions["T_phot"]
            pgm.add_node(
                "T_phot", PARAM_LABELS["T_phot"], x, y,
                scale=sc, fixed=True, fontsize=fs,
            )

        # --- intermediate deterministic nodes ---
        for gm in group_meta:
            inter = gm.get("intermediate")
            if inter and inter in positions:
                x, y = positions[inter]
                pgm.add_node(
                    inter, gm["inter_label"], x, y,
                    scale=sc, aspect=1.6, fontsize=fs,
                )

        # --- kernel node ---
        kx, ky = positions["K"]
        k_label = r"$K(\tau;\,\lambda)$" if self.is_multiband else r"$K(\tau)$"
        pgm.add_node("K", k_label, kx, ky, scale=sc, aspect=1.6,
                      fontsize=fs)

        # --- data nodes ---
        pgm.add_node(
            "x_i", r"$x_i$", *positions["x_i"],
            scale=sc, fixed=True, fontsize=fs,
        )
        pgm.add_node(
            "y_i", r"$y_i$", *positions["y_i"],
            scale=sc, observed=True, fontsize=fs,
        )
        pgm.add_node(
            "sigma_obs", r"$\sigma_{{\rm obs},i}$", *positions["sigma_obs"],
            scale=sc, fixed=True, fontsize=fs,
        )

        # --- edges ---
        for gm in group_meta:
            inter = gm.get("intermediate")
            if inter:
                for p in gm["params"]:
                    pgm.add_edge(p, inter)
                pgm.add_edge(inter, gm["target"])
            else:
                for p in gm["params"]:
                    pgm.add_edge(p, gm["target"])

        if "T_phot" in positions:
            pgm.add_edge("T_phot", "c_lambda")

        pgm.add_edge("K", "y_i")
        pgm.add_edge("x_i", "y_i")
        pgm.add_edge("sigma_obs", "y_i")

        # --- observation plate ---
        dx = [positions["x_i"][0], positions["y_i"][0],
              positions["sigma_obs"][0]]
        plate_l = min(dx) - 0.75
        plate_r = max(dx) + 0.75
        plate_b = positions["y_i"][1] - 0.65
        n_obs = getattr(self.solver, "N", "N")
        pgm.add_plate(
            [plate_l, plate_b, plate_r - plate_l, 1.3],
            label=rf"$i = 1, \ldots, {n_obs}$",
            shift=-0.1,
        )

        # --- band plate (multi-band only) ---
        if self.is_multiband and "c_lambda" in positions:
            cx, cy = positions["c_lambda"]
            n_bands = getattr(getattr(self.solver, "data", None),
                              "n_bands", "B")
            pgm.add_plate(
                [cx - 0.9, cy - 0.65, 1.8, 1.3],
                label=rf"$b = 1, \ldots, {n_bands}$",
                shift=-0.1,
            )

        pgm.render()
        fig = pgm.figure
        fig.set_dpi(dpi)

        if show_legend:
            self._draw_legend(fig, fs)

        return fig

    # ----- legend -------------------------------------------------------------

    def _draw_legend(self, fig, font_size):
        """Add a parameter legend below the PGM diagram."""
        entries = []
        for param in self._physical_keys:
            sym = PARAM_LABELS.get(param, f"${param}$")
            desc = PARAM_DESCRIPTIONS.get(param, param)
            entries.append((sym, desc))
        if self.is_multiband:
            entries.append((PARAM_LABELS["T_phot"],
                            PARAM_DESCRIPTIONS["T_phot"]))

        # Lay out in two columns
        n = len(entries)
        mid = (n + 1) // 2
        col_L = entries[:mid]
        col_R = entries[mid:]

        lines = []
        for i in range(mid):
            left = f"{col_L[i][0]}  {col_L[i][1]}"
            if i < len(col_R):
                right = f"{col_R[i][0]}  {col_R[i][1]}"
            else:
                right = ""
            lines.append((left, right))

        # Make room at the bottom of the figure
        w, h = fig.get_size_inches()
        n_rows = len(lines)
        extra = 0.28 * n_rows + 0.35
        new_h = h + extra
        fig.set_size_inches(w, new_h)

        for ax in fig.axes:
            pos = ax.get_position()
            scale = h / new_h
            offset = extra / new_h
            ax.set_position([pos.x0,
                             pos.y0 * scale + offset,
                             pos.width,
                             pos.height * scale])

        # Draw each row as two side-by-side text elements
        legend_fs = max(font_size - 2, 7)
        row_h = 0.28 / new_h
        top_y = (extra - 0.2) / new_h

        for i, (left, right) in enumerate(lines):
            y = top_y - i * row_h
            fig.text(0.10, y, left,  ha="left",  va="center",
                     fontsize=legend_fs)
            if right:
                fig.text(0.55, y, right, ha="left", va="center",
                         fontsize=legend_fs)
