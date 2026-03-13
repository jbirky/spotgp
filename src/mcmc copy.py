"""
MCMC sampling for GP hyperparameters using BlackJAX NUTS.

Provides a sampler that wraps a GPSolver object, using its
differentiable log-posterior for gradient-based NUTS sampling.
"""
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import blackjax

try:
    from .gp_solver import GPSolver
except ImportError:
    from gp_solver import GPSolver

__all__ = ["MCMCSampler"]


class MCMCSampler:
    """
    MCMC sampler for GP hyperparameters using BlackJAX NUTS.

    Wraps a GPSolver object and uses its differentiable log-posterior
    for gradient-based sampling.

    Parameters
    ----------
    gp : GPSolver
        A configured GPSolver instance. The solver's log_posterior,
        bounds, param_keys, and optimization methods are used directly.
    """

    def __init__(self, gp):
        if not isinstance(gp, GPSolver):
            raise TypeError("gp must be a GPSolver instance")
        self.gp = gp

        # Storage for MCMC results
        self.samples = None
        self._nuts_info = None

    @property
    def param_keys(self):
        return self.gp.param_keys

    @property
    def n_params(self):
        return self.gp.n_params

    # =================================================================
    # BlackJAX NUTS sampling
    # =================================================================

    def run_nuts(self, n_samples=1000, n_warmup=500, theta_init=None,
                 mass_matrix_method="hessian_map", step_size=None,
                 rng_key=None, target_accept=0.8, progress_bar=False):
        """
        Run BlackJAX NUTS sampler.

        Uses a manual warmup loop with dual averaging for step-size
        adaptation, then a JIT-compiled sampling loop via lax.scan.

        Parameters
        ----------
        n_samples : int
            Number of post-warmup samples (default 1000).
        n_warmup : int
            Number of warmup steps for step-size adaptation (default 500).
        theta_init : array_like, optional
            Initial position. If None, uses GPSolver's MAP estimate.
        mass_matrix_method : {"hessian_map", "fisher", "laplace", "diagonal", None}
            Method to estimate the mass matrix (delegated to GPSolver).
        step_size : float, optional
            NUTS step size. If None, adapted via dual averaging.
        rng_key : jax.random.PRNGKey, optional
            Random key. Default: PRNGKey(0).
        target_accept : float
            Target acceptance rate for dual averaging (default 0.8).
        progress_bar : bool
            If True, display tqdm progress bars for warmup and sampling
            (default False). When enabled, sampling falls back to a
            Python loop instead of lax.scan.

        Returns
        -------
        samples : jnp.ndarray, shape (n_samples, n_params)
            Posterior samples.
        info : dict
            Sampling diagnostics.
        """
        gp = self.gp

        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        # Initial position
        if theta_init is None:
            if gp.map_estimate is None:
                gp.find_map()
            theta_init = gp.map_estimate
        elif isinstance(theta_init, dict):
            theta_init = jnp.array(
                [float(theta_init[k]) for k in gp.param_keys],
                dtype=jnp.float64)
        else:
            theta_init = jnp.asarray(theta_init, dtype=jnp.float64)

        # Estimate mass matrix (delegated to GPSolver)
        inv_mass = gp._get_mass_matrix(mass_matrix_method, theta_init)

        # For NUTS, use diagonal mass matrix (more robust than full)
        inv_mass_diag = jnp.diag(inv_mass)
        # Clamp extreme values for stability
        median_var = jnp.median(inv_mass_diag)
        inv_mass_diag = jnp.clip(inv_mass_diag,
                                  median_var * 1e-4, median_var * 1e4)

        # Initial step size: heuristic based on mass matrix scale
        if step_size is None:
            step_size = float(0.5 * jnp.min(jnp.sqrt(inv_mass_diag)))
            step_size = max(step_size, 1e-5)

        # -- Warmup: dual averaging for step size --------------------
        if progress_bar:
            from tqdm.auto import tqdm

        print(f"Warmup: {n_warmup} steps (dual averaging, "
              f"init step_size={step_size:.6f})...")
        log_step = jnp.log(step_size)
        log_step_bar = jnp.log(step_size)
        mu = jnp.log(10.0 * step_size)
        gamma = 0.05
        t0 = 10
        kappa = 0.75
        H_bar = 0.0

        state = blackjax.nuts(
            gp.log_posterior,
            step_size=step_size,
            inverse_mass_matrix=inv_mass_diag,
        ).init(theta_init)

        warmup_key, sample_key = jax.random.split(rng_key)

        warmup_iter = range(1, n_warmup + 1)
        if progress_bar:
            warmup_iter = tqdm(warmup_iter, desc="Warmup", leave=True)

        for m in warmup_iter:
            warmup_key, step_key = jax.random.split(warmup_key)
            current_step = max(float(jnp.exp(log_step)), 1e-10)

            kernel = blackjax.nuts(
                gp.log_posterior,
                step_size=current_step,
                inverse_mass_matrix=inv_mass_diag,
            )
            state, step_info = kernel.step(step_key, state)

            accept = float(step_info.acceptance_rate)
            # Dual averaging update (Hoffman & Gelman 2014, Algorithm 5)
            w = 1.0 / (m + t0)
            H_bar = (1 - w) * H_bar + w * (target_accept - accept)
            log_step = mu - jnp.sqrt(m) / gamma * H_bar
            m_w = m ** (-kappa)
            log_step_bar = m_w * log_step + (1 - m_w) * log_step_bar

            if progress_bar:
                warmup_iter.set_postfix(
                    step_size=f"{current_step:.2e}", accept=f"{accept:.3f}")

        final_step_size = max(float(jnp.exp(log_step_bar)), 1e-8)
        print(f"  Adapted step size: {final_step_size:.6f}")

        # -- Sampling ------------------------------------------------
        print(f"Sampling {n_samples} post-warmup iterations...")

        nuts_kernel = blackjax.nuts(
            gp.log_posterior,
            step_size=final_step_size,
            inverse_mass_matrix=inv_mass_diag,
        )

        sample_keys = jax.random.split(sample_key, n_samples)

        if progress_bar:
            all_positions = []
            all_divergent = []
            all_accept = []
            all_num_steps = []

            sample_iter = tqdm(range(n_samples), desc="Sampling",
                               leave=True)
            for i in sample_iter:
                state, step_info = nuts_kernel.step(sample_keys[i], state)
                all_positions.append(state.position)
                all_divergent.append(step_info.is_divergent)
                all_accept.append(step_info.acceptance_rate)
                all_num_steps.append(step_info.num_integration_steps)

                if i % 10 == 0:
                    sample_iter.set_postfix(
                        accept=f"{float(step_info.acceptance_rate):.3f}",
                        div=int(jnp.sum(jnp.array(all_divergent))))

            self.samples = jnp.stack(all_positions)
            self._nuts_info = {
                "divergences": np.asarray(all_divergent),
                "acceptance_rate": np.asarray(all_accept),
                "num_steps": np.asarray(all_num_steps),
                "step_size": final_step_size,
                "n_warmup": n_warmup,
                "n_samples": n_samples,
                "n_divergent": int(jnp.sum(jnp.array(all_divergent))),
            }
        else:
            # JIT-compiled scan (faster, no progress bar)
            def one_step(carry, key):
                state = carry
                state, info = nuts_kernel.step(key, state)
                return state, (state, info)

            final_state, (states, infos) = jax.lax.scan(
                one_step, state, sample_keys
            )

            self.samples = states.position
            self._nuts_info = {
                "divergences": np.asarray(infos.is_divergent),
                "acceptance_rate": np.asarray(infos.acceptance_rate),
                "num_steps": np.asarray(infos.num_integration_steps),
                "step_size": final_step_size,
                "n_warmup": n_warmup,
                "n_samples": n_samples,
                "n_divergent": int(jnp.sum(infos.is_divergent)),
            }

        n_div = self._nuts_info["n_divergent"]
        mean_accept = float(np.mean(self._nuts_info["acceptance_rate"]))
        print(f"NUTS complete: {n_samples} samples, "
              f"{n_div} divergences, "
              f"mean acceptance rate = {mean_accept:.3f}")

        return self.samples, self._nuts_info

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
            raise RuntimeError("No samples available. Run run_nuts() first.")

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

        if self._nuts_info is not None:
            print(f"\nDivergences: {self._nuts_info['n_divergent']}")
            print(f"Mean acceptance: "
                  f"{np.mean(self._nuts_info['acceptance_rate']):.3f}")

        return stats

    def plot_covariance(self, method="fisher", theta_map=None,
                        n_sigma=2, n_grid=200, samples=None,
                        figsize=None, color="C0", alpha=0.3,
                        true_params=None, savefig=None):
        """
        Corner plot of 2D covariance ellipses from the Hessian or Fisher
        matrix, with 1D marginal Gaussians on the diagonal.

        Parameters
        ----------
        method : {"fisher", "hessian_map", "laplace"}
            Which matrix to use.
        theta_map : array_like, optional
            Center of the ellipses. If None, uses MAP estimate.
        n_sigma : float
            Number of sigma for the ellipse contours (default 2).
        n_grid : int
            Grid resolution for the ellipse curves (default 200).
        samples : array_like, optional
            If provided, scatter MCMC samples behind the ellipses.
        figsize : tuple, optional
            Figure size.
        color : str
            Color for ellipses and Gaussians (default "C0").
        alpha : float
            Fill alpha for the ellipse interiors (default 0.3).
        true_params : dict or array_like, optional
            True parameter values to mark with crosshairs.
        savefig : str, optional
            If provided, save figure to this path.

        Returns
        -------
        fig, axes : matplotlib Figure and 2D array of Axes.
        """
        import matplotlib.pyplot as plt

        gp = self.gp

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

        if samples is not None:
            samples = np.asarray(samples)

        n = self.n_params
        keys = self.param_keys
        if figsize is None:
            figsize = (2.5 * n, 2.5 * n)

        fig, axes = plt.subplots(n, n, figsize=figsize)
        if n == 1:
            axes = np.array([[axes]])

        t_ellipse = np.linspace(0, 2 * np.pi, n_grid)

        for i in range(n):
            for j in range(n):
                ax = axes[i, j]

                if j > i:
                    ax.set_visible(False)
                    continue

                if i == j:
                    sigma_i = np.sqrt(cov[i, i])
                    x_range = np.linspace(mu[i] - 4 * sigma_i,
                                          mu[i] + 4 * sigma_i, 300)
                    pdf = np.exp(
                        -0.5 * ((x_range - mu[i]) / sigma_i) ** 2)
                    pdf /= pdf.max()
                    ax.plot(x_range, pdf, color=color, lw=1.5)
                    ax.fill_between(x_range, pdf, alpha=alpha,
                                    color=color)
                    ax.axvline(mu[i], color=color, ls="--", lw=0.8)
                    if true_arr is not None and np.isfinite(true_arr[i]):
                        ax.axvline(true_arr[i], color="k", ls=":", lw=1)
                    if samples is not None:
                        ax.hist(samples[:, i], bins=30, density=True,
                                alpha=0.15, color="gray",
                                histtype="stepfilled")
                    ax.set_yticks([])
                else:
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
                    if samples is not None:
                        ax.scatter(samples[:, j], samples[:, i],
                                   s=1, alpha=0.1, color="gray",
                                   rasterized=True)

                if i == n - 1:
                    ax.set_xlabel(keys[j])
                else:
                    ax.set_xticklabels([])
                if j == 0 and i != 0:
                    ax.set_ylabel(keys[i])
                elif j != 0:
                    ax.set_yticklabels([])

        fig.tight_layout()

        if savefig is not None:
            fig.savefig(savefig, dpi=150, bbox_inches="tight")

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
