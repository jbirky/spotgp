"""
MCMC sampling for GP hyperparameters.

MCMCSampler is the base class providing shared diagnostics, summary,
and plotting utilities.  MetropolisSampler adds gradient-free
Metropolis-Hastings sampling with adaptive step size.

Note: This is the NumPy equivalent of the JAX mcmc.py module.
BlackJAXSampler is replaced by MetropolisSampler (Metropolis-Hastings),
since NUTS requires automatic differentiation unavailable in NumPy.
"""
import numpy as np

try:
    from .gp_solver import GPSolver
except ImportError:
    from gp_solver import GPSolver

__all__ = ["MCMCSampler", "BlackJAXSampler"]


# =====================================================================
# Base class
# =====================================================================

class MCMCSampler:
    """
    Base MCMC sampler for GP hyperparameters.

    Wraps a GPSolver object and provides shared storage, diagnostics,
    summary statistics, corner plots, and dict conversion.  Subclasses
    implement specific sampling algorithms.

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
            Extra keyword arguments forwarded to ``corner.corner``.

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
                gp.find_map()
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

    def to_dict(self, samples=None):
        """
        Convert samples array to a dict keyed by parameter name.

        Parameters
        ----------
        samples : ndarray, optional
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
# Metropolis-Hastings sampler (numpy equivalent of BlackJAXSampler)
# =====================================================================

class BlackJAXSampler(MCMCSampler):
    """
    Metropolis-Hastings sampler (NumPy equivalent of the JAX NUTS sampler).

    Inherits diagnostics, summary, plotting, and dict conversion from
    MCMCSampler.  Implements ``run_nuts`` as an adaptive
    Metropolis-Hastings sampler with diagonal Gaussian proposal,
    since NUTS requires automatic differentiation unavailable in NumPy.

    Parameters
    ----------
    gp : GPSolver
        A configured GPSolver instance.
    """

    def run_nuts(self, n_samples=1000, n_warmup=500, theta_init=None,
                 mass_matrix_method="hessian_map", step_size=None,
                 rng_key=None, target_accept=0.8, progress_bar=False):
        """
        Run Metropolis-Hastings sampler with adaptive step size.

        Uses a Gaussian proposal with covariance proportional to the
        inverse mass matrix.  Step size is adapted during warmup to
        achieve ``target_accept``.

        Parameters
        ----------
        n_samples : int
            Number of post-warmup samples (default 1000).
        n_warmup : int
            Number of warmup steps for step-size adaptation (default 500).
        theta_init : dict or array_like, optional
            Initial position. If None, uses GPSolver's MAP estimate.
        mass_matrix_method : {"hessian_map", "fisher", "laplace", "diagonal", None}
            Method to estimate the mass matrix (delegated to GPSolver).
        step_size : float, optional
            Initial proposal step size. If None, estimated from mass matrix.
        rng_key : int or None
            Random seed (used as numpy integer seed).
        target_accept : float
            Target acceptance rate for step-size adaptation (default 0.8).
        progress_bar : bool
            If True, display tqdm progress bars (default False).

        Returns
        -------
        samples : ndarray, shape (n_samples, n_params)
            Posterior samples.
        info : dict
            Sampling diagnostics.
        """
        seed = int(rng_key) if rng_key is not None else 0
        rng = np.random.default_rng(seed)

        gp = self.gp

        # Initial position
        if theta_init is None:
            if gp.map_estimate is None:
                gp.find_map()
            theta_init = np.asarray(gp.map_estimate, dtype=np.float64)
        elif isinstance(theta_init, dict):
            theta_init = np.array(
                [float(theta_init[k]) for k in gp.param_keys],
                dtype=np.float64)
        else:
            theta_init = np.asarray(theta_init, dtype=np.float64)

        # Estimate mass matrix (delegated to GPSolver)
        inv_mass = np.asarray(gp._get_mass_matrix(mass_matrix_method, theta_init))

        # Use diagonal for robustness
        inv_mass_diag = np.diag(inv_mass)
        median_var = np.median(inv_mass_diag)
        inv_mass_diag = np.clip(inv_mass_diag,
                                 median_var * 1e-4, median_var * 1e4)

        # Initial step size
        if step_size is None:
            step_size = float(0.5 * np.min(np.sqrt(inv_mass_diag)))
            step_size = max(step_size, 1e-5)

        # Proposal: diagonal Gaussian with scale = step_size * sqrt(inv_mass_diag)
        proposal_scale = step_size * np.sqrt(inv_mass_diag)

        # -- Warmup: adaptive step size ----------------------------------
        if progress_bar:
            from tqdm.auto import tqdm

        print(f"Warmup: {n_warmup} steps (adaptive, "
              f"init step_size={step_size:.6f})...")

        current = theta_init.copy()
        current_logp = float(gp.log_posterior(current))

        n_accept = 0
        warmup_iter = range(1, n_warmup + 1)
        if progress_bar:
            warmup_iter = tqdm(warmup_iter, desc="Warmup", leave=True)

        for m in warmup_iter:
            proposal = current + proposal_scale * rng.standard_normal(len(current))
            prop_logp = float(gp.log_posterior(proposal))
            log_alpha = prop_logp - current_logp

            if np.log(rng.random()) < log_alpha:
                current = proposal
                current_logp = prop_logp
                n_accept += 1

            # Adapt step size every 50 steps
            if m % 50 == 0:
                accept_rate = n_accept / m
                if accept_rate < target_accept:
                    step_size *= 0.9
                else:
                    step_size *= 1.1
                step_size = max(step_size, 1e-8)
                proposal_scale = step_size * np.sqrt(inv_mass_diag)

            if progress_bar:
                warmup_iter.set_postfix(
                    step_size=f"{step_size:.2e}",
                    accept=f"{n_accept/m:.3f}")

        print(f"  Adapted step size: {step_size:.6f}")

        # -- Sampling ----------------------------------------------------
        print(f"Sampling {n_samples} post-warmup iterations...")

        all_positions = []
        all_accept = []

        n_accept_samp = 0
        sample_iter = range(n_samples)
        if progress_bar:
            sample_iter = tqdm(sample_iter, desc="Sampling", leave=True)

        for i in sample_iter:
            proposal = current + proposal_scale * rng.standard_normal(len(current))
            prop_logp = float(gp.log_posterior(proposal))
            log_alpha = prop_logp - current_logp

            accept_prob = min(1.0, np.exp(log_alpha))
            if np.log(rng.random()) < log_alpha:
                current = proposal
                current_logp = prop_logp
                n_accept_samp += 1

            all_positions.append(current.copy())
            all_accept.append(accept_prob)

            if progress_bar and i % 10 == 0:
                sample_iter.set_postfix(
                    accept=f"{n_accept_samp / (i + 1):.3f}")

        self.samples = np.array(all_positions)
        self._info = {
            "divergences": np.zeros(n_samples, dtype=bool),
            "acceptance_rate": np.array(all_accept),
            "num_steps": np.ones(n_samples, dtype=int),
            "step_size": step_size,
            "n_warmup": n_warmup,
            "n_samples": n_samples,
            "n_divergent": 0,
        }

        mean_accept = float(np.mean(all_accept))
        print(f"MH complete: {n_samples} samples, "
              f"0 divergences, "
              f"mean acceptance rate = {mean_accept:.3f}")

        return self.samples, self._info
