"""Base MCMC sampler with shared diagnostics, summary, and plotting."""

import logging

import numpy as np

from ..gp_solver import GPSolver

logger = logging.getLogger("spotgp")


class MCMCSampler:
    """
    Base MCMC sampler for GP hyperparameters.

    Wraps a GPSolver object and provides shared storage, diagnostics,
    summary statistics, corner plots, and dict conversion.  Subclasses
    implement specific sampling algorithms (e.g. NUTS).

    Parameters
    ----------
    gp : GPSolver
        A configured GPSolver instance.
    """

    def __init__(self, gp):
        if not isinstance(gp, GPSolver):
            raise TypeError("gp must be a GPSolver instance")
        self.gp = gp

        # Storage for MCMC results
        self.samples = None
        self._info = None
        self._last_state = None
        self._adapted_step_size = None
        self._adapted_inv_mass = None
        self._last_rng_key = None
        self._checkpoint_file = None
        
        self._map_completed = False
        self._warmup_completed = False

    @property
    def param_keys(self):
        return self.gp.param_keys

    @property
    def n_params(self):
        return self.gp.n_params

    # =================================================================
    # Diagnostics
    # =================================================================

    def summary(self):
        """
        Print summary statistics of the posterior samples.

        Returns
        -------
        stats : dict
            Parameter names mapped to (mean, std, 16%, 50%, 84%).
        """
        if self.samples is None:
            raise RuntimeError("No samples available. Run a sampler first.")

        samples = np.asarray(self.samples)
        stats = {}

        print(f"{'param':>12s}  {'mean':>10s}  {'std':>10s}  "
              f"{'16%':>10s}  {'50%':>10s}  {'84%':>10s}")
        print("-" * 68)
        for i, key in enumerate(self.param_keys):
            col = samples[:, i]
            q16, q50, q84 = np.percentile(col, [16, 50, 84])
            m, s = np.mean(col), np.std(col)
            stats[key] = {"mean": m, "std": s,
                          "q16": q16, "q50": q50, "q84": q84}
            print(f"{key:>12s}  {m:10.5f}  {s:10.5f}  "
                  f"{q16:10.5f}  {q50:10.5f}  {q84:10.5f}")

        if self._info is not None and "n_divergent" in self._info:
            print(f"\nDivergences: {self._info['n_divergent']}")
            print(f"Mean acceptance: "
                  f"{np.mean(self._info['acceptance_rate']):.3f}")

        return stats

    def plot_covariance(self, method="fisher", theta_map=None,
                        n_sigma=2, n_grid=200, samples=None,
                        figsize=None, color="C0", alpha=0.3,
                        true_params=None, savefig=None,
                        **corner_kwargs):
        """
        Corner plot of 2D covariance ellipses from the Hessian or Fisher
        matrix, with 1D marginal Gaussians on the diagonal.

        Uses ``corner.corner`` to lay out the figure when MCMC samples
        are provided, and overlays the Laplace/Fisher Gaussian
        approximation (ellipses + 1D marginals).

        Parameters
        ----------
        method : {"fisher", "hessian_map", "laplace"}
            Which matrix to use for the Gaussian approximation.
        theta_map : array_like, optional
            Center of the ellipses. If None, uses MAP estimate.
        n_sigma : float
            Number of sigma for the ellipse contours (default 2).
        n_grid : int
            Grid resolution for the ellipse curves (default 200).
        samples : array_like, optional
            If provided, plotted as the corner histogram/contours.
            If None, the figure is created with empty axes and only
            the Gaussian approximation is drawn.
        figsize : tuple, optional
            Figure size.
        color : str
            Color for Gaussian ellipses and marginals (default "C0").
        alpha : float
            Fill alpha for the ellipse interiors (default 0.3).
        true_params : dict or array_like, optional
            True parameter values to mark with crosshairs.
        savefig : str, optional
            If provided, save figure to this path.
        **corner_kwargs
            Extra keyword arguments forwarded to ``corner.corner``
            (e.g. ``quantiles``, ``show_titles``, ``hist_kwargs``).

        Returns
        -------
        fig, axes : matplotlib Figure and 2D array of Axes.
        """
        import corner
        import matplotlib.pyplot as plt

        gp = self.gp
        n = self.n_params
        keys = list(self.param_keys)

        # Get MAP center
        if theta_map is None:
            if gp.map_estimate is None:
                gp.fit_map()
            theta_map = gp.map_estimate
        mu = np.asarray(theta_map, dtype=np.float64)

        # Get covariance matrix
        if method == "fisher":
            if gp._fisher_matrix is None:
                gp.mass_matrix_fisher(theta_map)
            cov = np.asarray(gp.inverse_mass_matrix)
        elif method in ("hessian_map", "laplace"):
            if method == "hessian_map":
                if gp._hessian is None:
                    gp.mass_matrix_hessian_map(theta_map)
            else:
                if gp._laplace_hessian is None:
                    gp.mass_matrix_laplace(theta_map)
            cov = np.asarray(gp.inverse_mass_matrix)
        else:
            raise ValueError(f"Unknown method: {method!r}")

        # Parse true_params
        if true_params is not None:
            if isinstance(true_params, dict):
                true_arr = np.array([true_params.get(k, np.nan)
                                     for k in self.param_keys])
            else:
                true_arr = np.asarray(true_params, dtype=np.float64)
        else:
            true_arr = None

        # --- Build figure with corner ----------------------------------
        if samples is not None:
            samples = np.asarray(samples)
            corner_defaults = dict(
                labels=keys,
                show_titles=True,
                plot_density=True,
                plot_contours=True,
                hist_kwargs={"density": True},
            )
            corner_defaults.update(corner_kwargs)
            if true_arr is not None:
                corner_defaults.setdefault(
                    "truths", list(true_arr))
            if figsize is not None:
                corner_defaults["fig"] = plt.figure(figsize=figsize)
            fig = corner.corner(samples, **corner_defaults)
        else:
            # No samples — create an empty corner-style grid
            if figsize is None:
                figsize = (2.5 * n, 2.5 * n)
            fig, axes_grid = plt.subplots(n, n, figsize=figsize)
            if n == 1:
                axes_grid = np.array([[axes_grid]])
            # Hide upper triangle
            for i in range(n):
                for j in range(n):
                    if j > i:
                        axes_grid[i, j].set_visible(False)
            # Label edges
            for i in range(n):
                for j in range(i + 1):
                    if i == n - 1:
                        axes_grid[i, j].set_xlabel(keys[j])
                    if j == 0 and i > 0:
                        axes_grid[i, j].set_ylabel(keys[i])
            fig.subplots_adjust(hspace=0.05, wspace=0.05)

        axes = np.array(fig.axes).reshape(n, n)

        # --- Overlay Gaussian approximation ----------------------------
        t_ellipse = np.linspace(0, 2 * np.pi, n_grid)

        for i in range(n):
            for j in range(n):
                if j > i:
                    continue
                ax = axes[i, j]

                if i == j:
                    # 1D Gaussian marginal (properly normalized)
                    sigma_i = np.sqrt(cov[i, i])
                    x_range = np.linspace(mu[i] - 4 * sigma_i,
                                          mu[i] + 4 * sigma_i, 300)
                    pdf = (np.exp(-0.5 * ((x_range - mu[i]) / sigma_i) ** 2)
                           / (sigma_i * np.sqrt(2 * np.pi)))
                    ax.plot(x_range, pdf, color=color, lw=1.5)
                    ax.fill_between(x_range, pdf, alpha=alpha,
                                    color=color)
                    ax.axvline(mu[i], color=color, ls="--", lw=0.8)
                    if true_arr is not None and np.isfinite(true_arr[i]):
                        ax.axvline(true_arr[i], color="k", ls=":", lw=1)
                else:
                    # 2D covariance ellipses
                    sub_cov = np.array([[cov[j, j], cov[j, i]],
                                        [cov[i, j], cov[i, i]]])
                    eigvals, eigvecs = np.linalg.eigh(sub_cov)
                    eigvals = np.maximum(eigvals, 0)

                    for ns in [1, n_sigma]:
                        xy = (eigvecs
                              @ np.diag(np.sqrt(eigvals) * ns)
                              @ np.array([np.cos(t_ellipse),
                                          np.sin(t_ellipse)]))
                        ax.plot(mu[j] + xy[0], mu[i] + xy[1],
                                color=color, lw=1.2)
                    xy1 = (eigvecs
                           @ np.diag(np.sqrt(eigvals))
                           @ np.array([np.cos(t_ellipse),
                                       np.sin(t_ellipse)]))
                    ax.fill(mu[j] + xy1[0], mu[i] + xy1[1],
                            color=color, alpha=alpha)

                    ax.plot(mu[j], mu[i], "+", color=color, ms=8,
                            mew=1.5)
                    if true_arr is not None:
                        if (np.isfinite(true_arr[j])
                                and np.isfinite(true_arr[i])):
                            ax.plot(true_arr[j], true_arr[i], "x",
                                    color="k", ms=6, mew=1.2)

        if savefig is not None:
            fig.savefig(savefig, dpi=150, bbox_inches="tight")

        return fig, axes

    def plot_corner_map(self, samples=None, checkpoint_path=None,
                        cmap="viridis", marker_size=40, savefig=None,
                        true_params=None, **corner_kwargs):
        """
        Corner plot of MCMC samples with MAP solutions overlaid as
        scatter points colored by their log-likelihood.

        Parameters
        ----------
        samples : array_like, optional
            Shape ``(n_samples, n_params)``.  If None, loads from the
            checkpoint file.
        checkpoint_path : str, optional
            Path to checkpoint ``.npz`` file containing MAP solutions.
            If None, uses the default checkpoint file.
        cmap : str
            Colormap for the MAP scatter points (default "viridis").
        marker_size : float
            Marker size for scatter points (default 40).
        savefig : str, optional
            If provided, save figure to this path.
        true_params : dict or array_like, optional
            True parameter values to mark with crosshairs.
        **corner_kwargs
            Extra keyword arguments forwarded to ``corner.corner``.

        Returns
        -------
        fig, axes : matplotlib Figure and 2D array of Axes.
        """
        import corner
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        gp = self.gp
        keys = list(self.param_keys)
        n = len(keys)

        # Load samples
        if samples is None:
            path = checkpoint_path or self._checkpoint_file
            if path is None:
                raise ValueError("No samples provided and no checkpoint file set.")
            samples = self.load_samples(path)
        samples = np.asarray(samples)

        # Flatten multi-chain samples: (n_chains, n_samples, n_params) -> (N, n_params)
        if samples.ndim == 3:
            samples = samples.reshape(-1, samples.shape[-1])

        # Load MAP solutions and log-likelihoods
        path = checkpoint_path or self._checkpoint_file
        if path is not None:
            _path = path if path.endswith(".npz") else path + ".npz"
            data = np.load(_path, allow_pickle=True)
            all_theta_maps = list(data["all_theta_maps"]) if "all_theta_maps" in data else None
            map_loglikes = np.asarray(data["map_loglikes"]) if "map_loglikes" in data else None
            data.close()
        else:
            all_theta_maps = getattr(self, 'all_theta_maps', None)
            map_loglikes = getattr(self, 'map_loglikes', None)

        if all_theta_maps is None:
            raise ValueError("No MAP solutions found. Run find_map / run_map first.")

        # Convert MAP dicts to arrays
        map_arrays = []
        for tm in all_theta_maps:
            if isinstance(tm, dict):
                arr = np.array([float(tm[k]) for k in keys])
            else:
                arr = np.asarray(tm, dtype=np.float64)
            map_arrays.append(arr)
        map_arr = np.array(map_arrays)  # (n_maps, n_params)

        # Compute log-likelihoods if not available
        if map_loglikes is None:
            map_loglikes = np.array([
                float(gp.log_likelihood_fn(m)) for m in map_arr])

        # Parse true_params
        if true_params is not None:
            if isinstance(true_params, dict):
                true_arr = [true_params.get(k, np.nan) for k in keys]
            else:
                true_arr = list(np.asarray(true_params, dtype=np.float64))
        else:
            true_arr = None

        # Build corner plot
        old_usetex = matplotlib.rcParams.get("text.usetex", False)
        matplotlib.rcParams["text.usetex"] = False
        try:
            corner_defaults = dict(
                labels=keys,
                show_titles=True,
                title_fmt=".3f",
                plot_density=True,
                plot_contours=True,
                hist_kwargs={"density": True},
            )
            if true_arr is not None:
                corner_defaults["truths"] = true_arr
            corner_defaults.update(corner_kwargs)
            fig = corner.corner(samples, **corner_defaults)
            axes = np.array(fig.axes).reshape(n, n)

            # Color normalization for log-likelihoods
            norm = Normalize(vmin=map_loglikes.min(), vmax=map_loglikes.max())
            cm = plt.get_cmap(cmap)

            # Overlay MAP scatter on off-diagonal panels
            for i in range(n):
                for j in range(i):
                    ax = axes[i, j]
                    sc = ax.scatter(map_arr[:, j], map_arr[:, i],
                                    c=map_loglikes, cmap=cmap, norm=norm,
                                    s=marker_size, edgecolors="k",
                                    linewidths=0.5, zorder=10)

            # Overlay MAP values on diagonal histograms
            for i in range(n):
                ax = axes[i, i]
                colors = cm(norm(map_loglikes))
                for k, (val, c) in enumerate(zip(map_arr[:, i], colors)):
                    ax.axvline(val, color=c, lw=1.2, alpha=0.7, zorder=10)

            # Add colorbar
            cbar_ax = fig.add_axes([1.02, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cbar_ax, label="log-likelihood")

            if savefig is not None:
                fig.savefig(savefig, dpi=150, bbox_inches="tight")
                logger.debug(f"Corner + MAP plot saved to {savefig}")
        finally:
            matplotlib.rcParams["text.usetex"] = old_usetex

        return fig, axes

    def to_dict(self, samples=None):
        """
        Convert samples array to a dict keyed by parameter name.

        Parameters
        ----------
        samples : jnp.ndarray, optional
            Shape (n_samples, n_params). If None, uses self.samples.

        Returns
        -------
        d : dict
            {param_name: array of shape (n_samples,)}
        """
        if samples is None:
            samples = self.samples
        if samples is None:
            raise RuntimeError("No samples available.")
        samples = np.asarray(samples)
        return {k: samples[:, i]
                for i, k in enumerate(self.param_keys)}


# =====================================================================
# BlackJAX NUTS sampler
# =====================================================================

