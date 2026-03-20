import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


__all__ = ["crb_corner_plot"]


def _plot_confidence_ellipse(ax, cov2, center, n_sigma=1, **kwargs):
    """Draw a confidence ellipse for a 2x2 covariance submatrix."""
    vals, vecs = np.linalg.eigh(cov2)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle  = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_sigma * np.sqrt(np.abs(vals))
    ellipse = Ellipse(xy=center, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)


def crb_corner_plot(cov_crb, param_keys, true_vals, corner_keys, plot_color="steelblue",
                    param_labels=None, panel_size=3.0, base_fontsize=16,
                    gridspec_kw={"wspace": 0.02, "hspace": 0.02}, title=None):
    """
    Corner plot of Cramer-Rao Bound confidence ellipses.

    Diagonal panels show 1D Gaussians with mean +/- std annotations.
    Lower-triangle panels show 2D confidence ellipses (1-sigma and 2-sigma)
    with the correlation coefficient printed in the upper right.

    Parameters
    ----------
    cov_crb : ndarray, shape (P, P)
        Full CRB covariance matrix over all parameters in param_keys.
    param_keys : list of str
        Names of all parameters corresponding to rows/columns of cov_crb.
    true_vals : list or ndarray
        True (or reference) values for all parameters in param_keys.
    corner_keys : list of str
        Subset of param_keys to include in the corner plot.
    param_labels : dict, optional
        Mapping from key to display label (supports LaTeX). Defaults to key name.
    panel_size : float, optional
        Size in inches per panel. Scales the overall figure. Default 3.0.
    base_fontsize : float, optional
        Base font size at panel_size=3. Scales with panel_size. Default 16.

    Returns
    -------
    fig, axes : Figure and ndarray of Axes
    """
    if param_labels is None:
        param_labels = {}

    corner_keys   = list(corner_keys)
    corner_idx    = [param_keys.index(k) for k in corner_keys]
    corner_vals   = [true_vals[i] for i in corner_idx]
    corner_labels = [param_labels.get(k, k) for k in corner_keys]

    n = len(corner_keys)

    # Scale figure size and fonts with number of panels
    figsize   = (panel_size * n, panel_size * n)
    fontscale = panel_size / 3.0
    fs_label  = base_fontsize * fontscale        # axis labels / diagonal text
    fs_annot  = (base_fontsize - 1) * fontscale  # rho annotation
    fs_title  = fs_label                         # suptitle
    lw        = 1.5 * fontscale
    dot_size  = 30 * fontscale

    fig, axes = plt.subplots(n, n, figsize=figsize, gridspec_kw=gridspec_kw)

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]

            if col > row:
                ax.set_visible(False)
                continue

            ix = corner_idx[col]
            cx = corner_vals[col]
            sx = np.sqrt(cov_crb[ix, ix])

            if row == col:
                # Diagonal: 1D Gaussian from CRB
                x = np.linspace(cx - 4*sx, cx + 4*sx, 300)
                y = np.exp(-0.5*((x - cx)/sx)**2) / (sx * np.sqrt(2*np.pi))
                ax.plot(x, y, color=plot_color, linewidth=lw)
                ax.fill_between(x, y, alpha=0.3, color=plot_color)
                ax.set_xlim(cx - 3.5*sx, cx + 3.5*sx)
                ax.set_yticks([])
                ax.set_title(rf"{corner_labels[col]} = {cx:.3g} $\pm$ {sx:.2g}",
                             fontsize=fs_label, pad=5)
            else:
                # Lower triangle: 2D confidence ellipses
                iy = corner_idx[row]
                cy = corner_vals[row]
                cov2 = cov_crb[np.ix_([ix, iy], [ix, iy])]
                sy   = np.sqrt(cov_crb[iy, iy])

                for ns, alpha in [(2, 0.20), (1, 0.40)]:
                    _plot_confidence_ellipse(
                        ax, cov2, (cx, cy), n_sigma=ns,
                        facecolor=plot_color, edgecolor=plot_color,
                        alpha=alpha, linewidth=lw
                    )

                ax.scatter([cx], [cy], color="white", s=dot_size, zorder=5)
                ax.set_xlim(cx - 3.5*sx, cx + 3.5*sx)
                ax.set_ylim(cy - 3.5*sy, cy + 3.5*sy)

                rho = cov_crb[ix, iy] / (sx * sy)
                ax.text(0.92, 0.92, rf"$\rho = {rho:.2f}$",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=fs_annot, color="k")

                if col == 0:
                    ax.set_ylabel(corner_labels[row], fontsize=fs_label)
                else:
                    ax.set_yticklabels([])

            if row == n - 1:
                ax.set_xlabel(corner_labels[col], fontsize=fs_label)
                ax.tick_params(axis='x', labelrotation=45,
                               labelsize=fs_label * 0.8)
            else:
                ax.set_xticklabels([])

            ax.tick_params(axis='y', labelsize=fs_label * 0.8)

    if title is not None:
        fig.suptitle(title, y=1.01, fontsize=fs_title)
        
    return fig, axes
