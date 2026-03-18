"""
Generate the spotgp logo in dark and light theme variants.
Outputs:
  docs/_static/spotgp_logo_dark.png   — colours for dark backgrounds
  docs/_static/spotgp_logo_light.png  — colours for light backgrounds
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, uniform_filter1d, maximum_filter1d, minimum_filter1d, gaussian_filter1d
from PIL import Image
import io

# ── Star / curve colours (shared) ─────────────────────────────────────────────
STAR_INNER = "#FFD0A0"
STAR_MID   = "#FF6A28"
STAR_OUTER = "#8A1018"

THEMES = {
    "dark": dict(
        TEXT_MAIN   = "#F0F4F8",
        TEXT_SUB    = "#A8C0D6",
        CURVE_COLOR = "#58A4D6",
        DOT_FILL    = "#0D1B2A",
    ),
    "light": dict(
        TEXT_MAIN   = "#1B2A3A",
        TEXT_SUB    = "#3A6080",
        CURVE_COLOR = "#2070B0",
        DOT_FILL    = "#FFFFFF",
    ),
}

FIG_W, FIG_H = 8.0, 3.2
DPI      = 200
FONTSIZE = 82
FONTFAM  = "DejaVu Sans Mono"
SUB_FONT = "DejaVu Sans"
X_CENTER = 0.50
SUB_TEXT = "GP kernels for stellar variability"


def make_logo(theme: str, out: str) -> None:
    colors = THEMES[theme]
    TEXT_MAIN   = colors["TEXT_MAIN"]
    TEXT_SUB    = colors["TEXT_SUB"]
    CURVE_COLOR = colors["CURVE_COLOR"]
    DOT_FILL    = colors["DOT_FILL"]

    rng = np.random.default_rng(42)

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_alpha(0.0)

    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)

    fig_w_px = FIG_W * DPI
    fig_h_px = FIG_H * DPI
    ax_inv   = ax.transAxes.inverted()

    # ── Layout measurement ────────────────────────────────────────────────────
    def probe(txt, fs, x, y, fw="bold", ff=FONTFAM, **kw):
        t = ax.text(x, y, txt, fontsize=fs, fontweight=fw, fontfamily=ff,
                    color=(0, 0, 0, 0), transform=ax.transAxes, **kw)
        fig.canvas.draw()
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        t.remove()
        return bb

    bb_o     = probe("o",   FONTSIZE, 0.5, 0.5, ha="center", va="center", fontstyle="italic")
    o_w_ax   = bb_o.width  / fig_w_px
    o_h_ax   = bb_o.height / fig_h_px

    bb_sp_m  = probe("sp",  FONTSIZE, 0.5, 0.5, ha="right", va="center", fontstyle="italic")
    bb_tgp_m = probe("tgp", FONTSIZE, 0.5, 0.5, ha="left",  va="center", fontstyle="italic")
    sp_w_ax  = bb_sp_m.width  / fig_w_px
    tgp_w_ax = bb_tgp_m.width / fig_w_px

    gap_extra = o_w_ax * 0.35        # extra padding so the star clears "p" and "t"
    o_gap_ax  = o_w_ax + gap_extra
    total_w = sp_w_ax + o_gap_ax + tgp_w_ax
    x_sp    = X_CENTER - total_w / 2 + sp_w_ax
    x_tgp   = X_CENTER - total_w / 2 + sp_w_ax + o_gap_ax
    X_STAR  = X_CENTER - total_w / 2 + sp_w_ax + o_gap_ax / 2

    bb_sp  = probe("sp",  FONTSIZE, x_sp,  0.5, ha="right", va="center", fontstyle="italic")
    bb_tgp = probe("tgp", FONTSIZE, x_tgp, 0.5, ha="left",  va="center", fontstyle="italic")

    left_text_ax  = ax_inv.transform((bb_sp.x0,  0))[0]
    right_text_ax = ax_inv.transform((bb_tgp.x1, 0))[0]

    # Re-derive X_STAR as the exact midpoint between the right edge of "sp"
    # and the left edge of "tgp" so the star is evenly spaced between p and t.
    right_sp_ax  = ax_inv.transform((bb_sp.x1,  0))[0]
    left_tgp_ax  = ax_inv.transform((bb_tgp.x0, 0))[0]
    X_STAR = (right_sp_ax + left_tgp_ax) / 2
    text_width_ax = right_text_ax - left_text_ax

    bb_sub_trial = probe(SUB_TEXT, 13, X_CENTER, 0.5,
                         fw="normal", ff=SUB_FONT,
                         ha="center", va="center", fontstyle="italic")
    sub_scale = (text_width_ax * fig_w_px) / bb_sub_trial.width
    sub_fs    = 13 * sub_scale
    sub_h_ax  = (bb_sub_trial.height / fig_h_px) * sub_scale

    gap_ax  = 0.025
    total_h = o_h_ax + gap_ax + sub_h_ax
    Y_TEXT  = 0.5 + total_h / 2 - o_h_ax / 2
    y_sub   = Y_TEXT - o_h_ax / 2 - gap_ax - sub_h_ax / 2

    # ── Wordmark text ─────────────────────────────────────────────────────────
    ax.text(x_sp, Y_TEXT, "sp",
            fontsize=FONTSIZE, fontweight="bold", fontfamily=FONTFAM,
            fontstyle="italic", ha="right", va="center", color=TEXT_MAIN,
            transform=ax.transAxes, zorder=5)
    ax.text(x_tgp, Y_TEXT, "tgp",
            fontsize=FONTSIZE, fontweight="bold", fontfamily=FONTFAM,
            fontstyle="italic", ha="left", va="center", color=TEXT_MAIN,
            transform=ax.transAxes, zorder=5)
    # ax.text(X_CENTER, y_sub, SUB_TEXT,
    #         fontsize=sub_fs, fontfamily=SUB_FONT, fontweight="normal",
    #         color=TEXT_SUB, fontstyle="italic",
    #         ha="center", va="center", transform=ax.transAxes, zorder=5)

    # ── Star disk inset ───────────────────────────────────────────────────────
    ax_pos  = ax.get_position()
    fig_cx  = ax_pos.x0 + X_STAR  * ax_pos.width
    fig_cy  = ax_pos.y0 + Y_TEXT  * ax_pos.height
    pad     = 1.08
    inset_h = o_h_ax * ax_pos.height * pad
    inset_w = inset_h * (FIG_H / FIG_W)

    ax_star = fig.add_axes(
        [fig_cx - inset_w / 2, fig_cy - inset_h / 2, inset_w, inset_h]
    )
    ax_star.set_aspect("equal")
    ax_star.set_axis_off()
    ax_star.patch.set_alpha(0.0)

    N  = 900
    yp, xp = np.mgrid[-N//2:N//2, -N//2:N//2]
    xn = xp / (N // 2)
    yn = yp / (N // 2)
    r  = np.sqrt(xn**2 + yn**2)

    u1, u2 = 0.55, 0.35
    with np.errstate(invalid="ignore"):
        mu = np.sqrt(np.maximum(1 - r**2, 0))
    limb = np.where(r <= 1.0, 1 - u1*(1-mu) - u2*(1-mu)**2, 0.0)

    raw_noise   = rng.normal(0, 1, (N, N))
    gran_large  = gaussian_filter(raw_noise, sigma=20)
    gran_small  = gaussian_filter(raw_noise, sigma=5)
    granulation = gran_large - gran_small
    granulation /= granulation[r <= 0.9].std()
    gran_texture = 0.05 * granulation * np.where(r <= 1.0, mu**0.3, 0.0)

    spots = [
        ( 0.22,  0.32, 0.13, 0.09,  12, 0.55),
        (-0.30, -0.12, 0.15, 0.12,  -6, 0.60),
        ( 0.05, -0.44, 0.10, 0.07,  28, 0.50),
        (-0.14,  0.50, 0.08, 0.05,   5, 0.45),
        ( 0.54, -0.20, 0.09, 0.06, -18, 0.45),
        (-0.52,  0.25, 0.07, 0.05,  10, 0.40),
        ( 0.40,  0.48, 0.07, 0.05, -10, 0.40),
        (-0.18, -0.56, 0.08, 0.05,  20, 0.45),
    ]
    spot_layer = np.zeros((N, N), dtype=np.float64)
    for (cx, cy, rx, ry, ang, depth) in spots:
        ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        dx = xn - cx;  dy = yn - cy
        dx_r =  dx * ca + dy * sa
        dy_r = -dx * sa + dy * ca
        spot_layer = np.maximum(spot_layer,
                                depth * np.exp(-(dx_r**2/(2*rx**2) + dy_r**2/(2*ry**2))))

    intensity = np.clip(limb + gran_texture - spot_layer * limb, 0.0, 1.0)

    corona_close = np.exp(-((r - 1.0) / 0.045)**2) * 0.38
    corona_wide  = np.exp(-((r - 1.0) / 0.10) **2) * 0.16
    corona = np.where(r > 1.0,
                      gaussian_filter(corona_close, sigma=2)
                      + gaussian_filter(corona_wide, sigma=5), 0.0)

    star_cmap = LinearSegmentedColormap.from_list(
        "star", [STAR_OUTER, STAR_MID, STAR_INNER], N=512)

    def hex_to_rgb01(h):
        return [int(h[i:i+2], 16)/255 for i in (1, 3, 5)]

    corona_rgba = np.zeros((*corona.shape, 4), dtype=np.float64)
    corona_rgba[..., :3] = hex_to_rgb01(STAR_MID)
    corona_rgba[..., 3]  = np.clip(corona * 0.55, 0, 1)

    star_rgb  = star_cmap(intensity)[..., :3]
    disk_mask = (r <= 1.0).astype(np.float64)
    star_rgba = np.concatenate([star_rgb, disk_mask[..., None]], axis=-1)

    EXT = 1.50
    ax_star.imshow(corona_rgba, origin="lower",
                   extent=[-EXT, EXT, -EXT, EXT], interpolation="bilinear")
    ax_star.imshow(star_rgba, origin="lower",
                   extent=[-1.08, 1.08, -1.08, 1.08], interpolation="bilinear")
    ax_star.set_xlim(-EXT, EXT)
    ax_star.set_ylim(-EXT, EXT)

    # ── CME loop (commented out) ──────────────────────────────────────────────
    scale_x = inset_w / (2 * EXT)
    scale_y = inset_h / (2 * EXT)

    # rgb_i = np.array(hex_to_rgb01(STAR_INNER))
    # rgb_o = np.array(hex_to_rgb01(STAR_MID))
    # star_top   = Y_TEXT + scale_y
    # avail_y    = 1.0 - star_top
    # loop_hw    = scale_x * 1.8
    # foot_y     = Y_TEXT + scale_y * 0.72
    # fp1_ax  = np.array([X_STAR - loop_hw, foot_y])
    # fp2_ax  = np.array([X_STAR + loop_hw, foot_y])
    # apex_ax = np.array([X_STAR, star_top + avail_y * 0.76])
    # n_cl   = 600
    # t_cl   = np.linspace(0, 1, n_cl)
    # cl_ax  = ((1-t_cl)**2)[:,None]*fp1_ax + (2*t_cl*(1-t_cl))[:,None]*apex_ax + (t_cl**2)[:,None]*fp2_ax
    # dcl    = np.gradient(cl_ax, axis=0)
    # tang   = dcl / (np.linalg.norm(dcl, axis=1, keepdims=True) + 1e-10)
    # norm_v = np.column_stack([-tang[:,1], tang[:,0]])
    # n_part = 3000
    # tube_w = loop_hw * 0.20
    # idx    = rng.integers(0, n_cl, n_part)
    # t_param = t_cl[idx]
    # perp    = rng.normal(0, tube_w, n_part)
    # along   = rng.normal(0, 0.003, n_part)
    # pts_ax  = cl_ax[idx] + perp[:,None]*norm_v[idx] + along[:,None]*tang[idx]
    # ell_r = np.sqrt(((pts_ax[:,0] - X_STAR)/scale_x)**2 +
    #                 ((pts_ax[:,1] - Y_TEXT) /scale_y)**2)
    # keep  = ((pts_ax[:,0] > 0.01) & (pts_ax[:,0] < 0.99) &
    #          (pts_ax[:,1] > 0.01) & (pts_ax[:,1] < 0.99) &
    #          (ell_r > 1.0))
    # pts_ax, t_f, perp_f = pts_ax[keep], t_param[keep], perp[keep]
    # x_l, y_l = pts_ax[:,0], pts_ax[:,1]
    # foot_prox = 1.0 - 1.55 * (t_f - 0.5)**2
    # tube_fade = np.exp(-np.abs(perp_f) / (tube_w * 0.85))
    # alpha_l   = np.clip(0.78 * foot_prox * tube_fade, 0.01, 0.82)
    # size_l    = np.clip(4.5 * tube_fade * (0.5 + 0.5*foot_prox), 0.3, 6.0)
    # fade_l  = np.clip(1.0 - foot_prox * 0.9, 0, 1)
    # rgb_mix = rgb_i[None,:] * (1-fade_l[:,None]) + rgb_o[None,:] * fade_l[:,None]
    # rgba_l  = np.column_stack([rgb_mix, alpha_l])
    # ax.scatter(x_l, y_l, s=size_l, c=rgba_l, edgecolors='none', zorder=4)
    # ell_cl  = np.sqrt(((cl_ax[:,0]-X_STAR)/scale_x)**2 + ((cl_ax[:,1]-Y_TEXT)/scale_y)**2)
    # show_cl = (cl_ax[:,1] > 0.01) & (cl_ax[:,1] < 0.99) & (ell_cl > 1.02)
    # segs    = np.split(np.where(show_cl)[0],
    #                    np.where(np.diff(np.where(show_cl)[0]) > 1)[0] + 1)
    # for seg in segs:
    #     if len(seg) >= 2:
    #         ax.plot(cl_ax[seg,0], cl_ax[seg,1], color=STAR_INNER,
    #                 alpha=0.15, linewidth=4.5, solid_capstyle='round', zorder=3)

    # ── GP lightcurve ─────────────────────────────────────────────────────────
    t = np.linspace(0, 6 * np.pi, 900)
    y  = 0.60 * np.sin(t + 0.3)
    y += 0.30 * np.sin(2.03*t - 0.6)
    y += 0.25 * np.sin(3.07*t + 1.1)
    y += 0.22 * np.sin(4.01*t - 0.9)
    y += 0.20 * np.sin(5.2 *t + 0.5)
    y += 0.18 * np.sin(7.3 *t - 1.2)
    y += 0.16 * np.sin(9.1 *t + 0.8)
    y += rng.normal(0, 0.018, size=len(t))
    y_sm   = uniform_filter1d(y, size=2)
    y_norm = (y_sm - y_sm.mean()) / (y_sm.max() - y_sm.min())

    gp_amp = 0.10
    y_plot = Y_TEXT + gp_amp * y_norm
    t_norm = (t - t.min()) / (t.max() - t.min())

    def smooth_step(x, lo, hi):
        c = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
        return c * c * (3 - 2 * c)

    trans   = 0.025
    in_text = (smooth_step(t_norm, left_text_ax - trans, left_text_ax + trans)
               * (1 - smooth_step(t_norm, right_text_ax - trans, right_text_ax + trans)))

    # Capture the canvas (text + star already drawn) and read per-column
    # pixel extents to make the uncertainty band follow the text contour.
    fig.canvas.draw()
    buf_rgba = np.array(fig.canvas.buffer_rgba())   # (H, W, 4) uint8
    fig_H, fig_W = buf_rgba.shape[:2]
    alpha_ch = buf_rgba[..., 3].astype(float)

    NARROW     = 0.055  # band half-width (axes units) outside the text
    PAD        = 0.018  # padding above/below detected text pixels (wide band)
    PAD_CURVY  = 0.055  # wider offset for the curvy band
    WIDE_BIG   = 80     # large window — fills inter-letter gaps, boxy
    WIDE_SMALL = 20     # small window — follows letter shapes closely, curvy

    # Raw pixel extents with standard padding (for wide band)
    band_hi_raw = y_plot + NARROW
    band_lo_raw = y_plot - NARROW
    # Raw pixel extents with wider padding (for curvy band)
    band_hi_raw_curvy = y_plot + NARROW
    band_lo_raw_curvy = y_plot - NARROW
    for i, tx in enumerate(t_norm):
        col_px = int(np.clip(tx * fig_W, 0, fig_W - 1))
        rows = np.where(alpha_ch[:, col_px] > 20)[0]
        if len(rows) > 0:
            band_hi_raw[i]       = 1.0 - rows.min() / fig_H + PAD
            band_lo_raw[i]       = 1.0 - rows.max() / fig_H - PAD
            band_hi_raw_curvy[i] = 1.0 - rows.min() / fig_H + PAD_CURVY
            band_lo_raw_curvy[i] = 1.0 - rows.max() / fig_H - PAD_CURVY

    # Wide band: fills all inter-letter gaps but boxy
    band_hi_wide = maximum_filter1d(band_hi_raw, size=WIDE_BIG)
    band_lo_wide = minimum_filter1d(band_lo_raw, size=WIDE_BIG)

    # Narrow band: follows letter contours closely, curvy but has gaps
    band_hi_curvy = maximum_filter1d(band_hi_raw_curvy, size=WIDE_SMALL)
    band_lo_curvy = minimum_filter1d(band_lo_raw_curvy, size=WIDE_SMALL)

    # Take the maximum width between the two so gaps are filled AND edges are curvy
    band_hi = np.maximum(band_hi_wide, band_hi_curvy)
    band_lo = np.minimum(band_lo_wide, band_lo_curvy)

    # Outside the text region revert to narrow band, then smooth transitions
    band_hi = np.where(in_text > 0.01, band_hi, y_plot + NARROW)
    band_lo = np.where(in_text > 0.01, band_lo, y_plot - NARROW)
    band_hi = gaussian_filter1d(band_hi, sigma=12)
    band_lo = gaussian_filter1d(band_lo, sigma=12)

    ax.fill_between(t_norm, band_lo, band_hi,
                    color="#89CFF0", alpha=0.45, linewidth=0, zorder=2)

    # Mean line with gradient alpha: dark outside text, faint inside
    r, g, b = matplotlib.colors.to_rgb(CURVE_COLOR)
    alpha_line = 1.0 - 0.8 * in_text          # 1.0 outside → 0.2 inside
    seg_alpha  = (alpha_line[:-1] + alpha_line[1:]) / 2  # per-segment average
    seg_colors = np.column_stack([
        np.full(len(seg_alpha), r),
        np.full(len(seg_alpha), g),
        np.full(len(seg_alpha), b),
        seg_alpha,
    ])
    pts  = np.array([t_norm, y_plot]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, colors=seg_colors, linewidth=1.8, zorder=4)
    ax.add_collection(lc)

    candidate_idx = np.where((t_norm < left_text_ax - trans) |
                             (t_norm > right_text_ax + trans))[0]
    obs_idx = np.sort(rng.choice(candidate_idx,
                                 size=min(38, len(candidate_idx)), replace=False))
    obs_t   = t_norm[obs_idx]
    obs_y   = y_plot[obs_idx] + rng.normal(0, 0.009, size=len(obs_idx))
    obs_err = rng.uniform(0.007, 0.018, size=len(obs_idx))
    ax.errorbar(obs_t, obs_y, yerr=obs_err,
                fmt="o", color=DOT_FILL, markersize=2.8,
                ecolor=CURVE_COLOR, elinewidth=0.9, capsize=0,
                markeredgecolor=CURVE_COLOR, markeredgewidth=0.9, zorder=4)

    # ── Save (with 15° counter-clockwise rotation) ────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, dpi=DPI, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    img = img.rotate(10, expand=True, resample=Image.BICUBIC)
    img.save(out)
    print(f"Saved → {out}")


# ── Generate both variants ─────────────────────────────────────────────────────
make_logo("dark",  "docs/_static/spotgp_logo_dark.png")
make_logo("light", "docs/_static/spotgp_logo_light.png")
