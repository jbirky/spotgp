"""
MCMC sampling for GP hyperparameters.

MCMCSampler is the base class providing shared diagnostics, summary,
and plotting utilities.  BlackJAXSampler adds gradient-based NUTS
sampling via the BlackJAX library.
"""
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
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

class BlackJAXSampler(MCMCSampler):
    """
    NUTS sampler using the BlackJAX library.

    Inherits diagnostics, summary, plotting, and dict conversion from
    MCMCSampler.  Adds ``run_nuts`` for gradient-based No-U-Turn
    sampling with dual-averaging step-size adaptation.

    When multiple chains are requested, sampling is parallelized across
    available devices via ``jax.pmap``.  Chains are distributed evenly
    across devices (``n_chains`` must be divisible by
    ``jax.device_count()``).  On a single GPU this behaves identically
    to the previous ``jax.vmap`` implementation.

    Parameters
    ----------
    gp : GPSolver
        A configured GPSolver instance.
    save_dir : str, optional
        Directory for all outputs produced by this sampler (corner
        plots, covariance plots, etc.).  Created automatically if it
        does not exist.  When set, ``save_checkpoint`` will default to
        saving the checkpoint inside this directory.
    """

    def __init__(self, gp, save_dir=None):
        super().__init__(gp)
        if save_dir is not None:
            import os
            os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self._n_devices = jax.device_count()

    def run_nuts(self, n_samples=1000, n_warmup=500, theta_init=None,
                 mass_matrix_method="hessian_map", step_size=None,
                 rng_key=None, target_accept=0.8, progress_bar=False,
                 n_chains=1, checkpoint_file=None):
        """
        Run BlackJAX NUTS sampler.

        Uses ``blackjax.window_adaptation`` for JIT-compiled warmup
        (step-size and mass-matrix adaptation), then ``jax.lax.scan``
        for the sampling loop.  Both paths avoid Python-level loops,
        minimizing retracing overhead and memory from accumulated
        intermediates.

        Warmup always runs a single chain to adapt the step size and
        mass matrix.  When ``n_chains > 1``, the adapted parameters
        are shared across all chains, which are initialized with
        jittered copies of the warmup endpoint and distributed across
        devices via ``jax.pmap`` (with ``jax.vmap`` over chains
        within each device).

        Parameters
        ----------
        n_samples : int
            Number of post-warmup samples per chain (default 1000).
        n_warmup : int
            Number of warmup steps for step-size adaptation (default 500).
        theta_init : dict or array_like, optional
            Initial position. If None, uses GPSolver's MAP estimate.
        mass_matrix_method : {"hessian_map", "fisher", "laplace", "diagonal", None}
            Method to estimate the mass matrix (delegated to GPSolver).
        step_size : float, optional
            Initial NUTS step size before adaptation. If None, a
            heuristic based on the mass matrix scale is used.
        rng_key : jax.random.PRNGKey, optional
            Random key. Default: PRNGKey(0).
        target_accept : float
            Target acceptance rate for dual averaging (default 0.8).
        progress_bar : bool
            If True, print periodic progress updates during the
            lax.scan sampling loop (default False).
        n_chains : int
            Number of independent chains to run in parallel via
            ``jax.pmap`` across devices (default 1).  Must be
            divisible by ``jax.device_count()``.  All chains share
            the same adapted step size and mass matrix from a single
            warmup.
        checkpoint_file : str, optional
            Default file path for ``save_checkpoint``.  When set,
            calling ``save_checkpoint()`` with no arguments will
            use this path.

        Returns
        -------
        samples : jnp.ndarray
            Shape ``(n_samples, n_params)`` when ``n_chains=1``, or
            ``(n_chains, n_samples, n_params)`` when ``n_chains > 1``.
        info : dict
            Sampling diagnostics (arrays have a leading chain
            dimension when ``n_chains > 1``).
        """
        import blackjax

        gp = self.gp
        n_devices = self._n_devices

        if n_chains > 1 and n_chains % n_devices != 0:
            raise ValueError(
                f"n_chains ({n_chains}) must be divisible by the number of "
                f"available devices ({n_devices}). Use n_chains in "
                f"{[n_devices * i for i in range(1, 5)]}.")

        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        # Initial position
        # theta_init can be:
        #   - None          -> use GPSolver MAP estimate
        #   - dict          -> single starting point
        #   - 1-D array     -> single starting point
        #   - list of dicts -> per-chain starting points
        #   - 2-D array (n_chains, n_params) -> per-chain starting points
        _per_chain_inits = None

        if theta_init is None:
            if gp.map_estimate is None:
                gp.fit_map()
            theta_init = gp.map_estimate
        elif isinstance(theta_init, list) and len(theta_init) > 0 and isinstance(theta_init[0], dict):
            _per_chain_inits = jnp.array(
                [[float(d[k]) for k in gp.param_keys] for d in theta_init],
                dtype=jnp.float64)
            theta_init = _per_chain_inits[0]  # best MAP for warmup
        elif isinstance(theta_init, dict):
            theta_init = jnp.array(
                [float(theta_init[k]) for k in gp.param_keys],
                dtype=jnp.float64)
        else:
            theta_init = jnp.asarray(theta_init, dtype=jnp.float64)
            if theta_init.ndim == 2:
                _per_chain_inits = theta_init
                theta_init = _per_chain_inits[0]  # best MAP for warmup

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

        self._n_chains = n_chains
        if checkpoint_file is not None:
            self._checkpoint_file = checkpoint_file

        # -- Warmup (single chain) -----------------------------------
        print(f"Warmup: {n_warmup} steps (window adaptation, "
              f"init step_size={step_size:.6f})...")

        warmup_key, sample_key = jax.random.split(rng_key)

        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            gp.log_posterior,
            is_mass_matrix_diagonal=True,
            initial_step_size=step_size,
            target_acceptance_rate=target_accept,
            progress_bar=progress_bar,
        )
        adapt_results, adapt_info = warmup.run(
            warmup_key, theta_init, num_steps=n_warmup,
        )

        adapted_step_size = adapt_results.parameters["step_size"]
        adapted_inv_mass = adapt_results.parameters["inverse_mass_matrix"]
        warmup_state = adapt_results.state

        print(f"  Adapted step size: {float(adapted_step_size):.6f}")

        # Store adapted parameters and checkpoint between warmup and
        # sampling so that warmup intermediates can be freed.
        self._adapted_step_size = float(adapted_step_size)
        self._adapted_inv_mass = np.asarray(adapted_inv_mass)
        self._last_state = warmup_state
        self._last_rng_key = sample_key
        self._info = {
            "step_size": self._adapted_step_size,
            "n_warmup": n_warmup,
            "n_samples": 0,
            "n_chains": n_chains,
            "n_divergent": 0,
        }

        if self._checkpoint_file is not None:
            self.save_checkpoint(append_samples=False)
            print("  Warmup checkpoint saved; clearing warmup memory...")

        del adapt_results, adapt_info, warmup
        jax.clear_caches()

        # Warm up the log_posterior JIT kernel so CUDA timers are
        # accurate when the sampling scan launches.
        jax.block_until_ready(gp.log_posterior(theta_init))

        # -- Sampling via lax.scan -----------------------------------
        def _run_one_chain(state, chain_key):
            """Sample one chain via lax.scan."""
            kernel = blackjax.nuts(
                gp.log_posterior,
                step_size=adapted_step_size,
                inverse_mass_matrix=adapted_inv_mass,
            )
            chain_keys = jax.random.split(chain_key, n_samples)

            def one_step(carry, key_idx):
                st, n_div = carry
                key, _idx = key_idx
                st, info = kernel.step(key, st)
                n_div = n_div + info.is_divergent.astype(jnp.int32)
                return (st, n_div), (st.position, info)

            indices = jnp.arange(n_samples)
            (final_st, total_div), (positions, infos) = jax.lax.scan(
                one_step, (state, jnp.int32(0)), (chain_keys, indices),
            )
            return final_st, total_div, positions, infos, chain_keys[-1]

        if n_chains > 1:
            chains_per_device = n_chains // n_devices

            print(f"Sampling {n_samples} iterations x {n_chains} chains "
                  f"across {n_devices} device(s)...")

            # Create independent starting positions for each chain.
            # If per-chain init positions were provided, use them directly;
            # otherwise jitter the warmup endpoint.
            if _per_chain_inits is not None and _per_chain_inits.shape[0] >= n_chains:
                init_positions = _per_chain_inits[:n_chains]
                print(f"  Using {n_chains} distinct MAP solutions as chain init positions")
            else:
                jitter_key, sample_key = jax.random.split(sample_key)
                jitter_scale = 0.01 * jnp.sqrt(adapted_inv_mass)
                noise = jax.random.normal(
                    jitter_key, shape=(n_chains, len(theta_init)))
                init_positions = warmup_state.position[None, :] \
                    + jitter_scale[None, :] * noise

            # Initialize NUTS states for each chain from jittered positions
            init_fn = blackjax.nuts(
                gp.log_posterior,
                step_size=adapted_step_size,
                inverse_mass_matrix=adapted_inv_mass,
            ).init
            states = jax.vmap(init_fn)(init_positions)

            sample_keys = jax.random.split(sample_key, n_chains)

            # Reshape for pmap: (n_devices, chains_per_device, ...)
            states = jax.tree.map(
                lambda x: x.reshape(n_devices, chains_per_device, *x.shape[1:]),
                states)
            sample_keys = sample_keys.reshape(n_devices, chains_per_device, -1)

            # pmap over devices, vmap over chains within each device
            all_final, all_div, all_pos, all_infos, all_last_keys = jax.pmap(
                jax.vmap(_run_one_chain)
            )(states, sample_keys)

            # Flatten device dimension: (n_devices, chains_per_device, ...) -> (n_chains, ...)
            all_final = jax.tree.map(
                lambda x: x.reshape(n_chains, *x.shape[2:]), all_final)
            all_div = all_div.reshape(n_chains)
            all_pos = all_pos.reshape(n_chains, n_samples, -1)
            all_infos = jax.tree.map(
                lambda x: x.reshape(n_chains, *x.shape[2:]), all_infos)
            all_last_keys = all_last_keys.reshape(n_chains, -1)

            # Shape: (n_chains, n_samples, n_params)
            self.samples = all_pos
            self._info = {
                "divergences": np.asarray(all_infos.is_divergent),
                "acceptance_rate": np.asarray(all_infos.acceptance_rate),
                "num_steps": np.asarray(
                    all_infos.num_integration_steps),
                "step_size": float(adapted_step_size),
                "n_warmup": n_warmup,
                "n_samples": n_samples,
                "n_chains": n_chains,
                "n_divergent": int(jnp.sum(all_div)),
            }

            self._last_state = all_final
            self._adapted_step_size = float(adapted_step_size)
            self._adapted_inv_mass = np.asarray(adapted_inv_mass)
            self._last_rng_key = all_last_keys

            total_div = int(jnp.sum(all_div))
            mean_accept = float(jnp.mean(
                jnp.array(self._info["acceptance_rate"])))
            print(f"NUTS complete: {n_chains} chains x {n_samples} samples, "
                  f"{total_div} total divergences, "
                  f"mean acceptance rate = {mean_accept:.3f}")
        else:
            print(f"Sampling {n_samples} post-warmup iterations...")

            final_state, total_div, positions, infos, last_key = \
                _run_one_chain(warmup_state, sample_key)

            self.samples = positions
            self._info = {
                "divergences": np.asarray(infos.is_divergent),
                "acceptance_rate": np.asarray(infos.acceptance_rate),
                "num_steps": np.asarray(infos.num_integration_steps),
                "step_size": float(adapted_step_size),
                "n_warmup": n_warmup,
                "n_samples": n_samples,
                "n_chains": 1,
                "n_divergent": int(total_div),
            }

            self._last_state = final_state
            self._adapted_step_size = float(adapted_step_size)
            self._adapted_inv_mass = np.asarray(adapted_inv_mass)
            self._last_rng_key = last_key

            mean_accept = float(np.mean(self._info["acceptance_rate"]))
            print(f"NUTS complete: {n_samples} samples, "
                  f"{int(total_div)} divergences, "
                  f"mean acceptance rate = {mean_accept:.3f}")

        return self.samples, self._info

    def save_checkpoint(self, path=None, append_samples=True,
                        plot_corner=False):
        """
        Save sampler state to disk for later resumption.

        When ``append_samples=True`` (the default), new samples are
        appended to any existing samples already stored in ``path``,
        and ``self.samples`` is cleared from memory.  This enables a
        sample-checkpoint-clear loop that keeps memory usage constant.

        Parameters
        ----------
        path : str, optional
            File path (saved as ``.npz``).  If None, uses the
            ``checkpoint_file`` set in ``run_nuts``, or
            ``save_dir/checkpoint.npz`` if ``save_dir`` was set.
        append_samples : bool
            If True, append current ``self.samples`` to any samples
            already on disk, then clear ``self.samples`` from memory.
            If False, overwrite with only the current in-memory samples.
        plot_corner : bool
            If True, load all samples currently on disk after saving
            and write a corner plot to ``save_dir/corner_plot.png``
            (or alongside the checkpoint file if ``save_dir`` is not
            set).
        """
        import os
        if self._last_state is None:
            raise RuntimeError("No sampler state to save. Run run_nuts first.")
        if path is None:
            path = self._checkpoint_file
        if path is None and self.save_dir is not None:
            path = os.path.join(self.save_dir, "checkpoint.npz")
        if path is None:
            raise ValueError(
                "No path provided, no checkpoint_file set, and no save_dir. "
                "Pass a path, set checkpoint_file in run_nuts, or set save_dir.")

        samples_to_save = np.asarray(self.samples) if self.samples is not None else None

        # Merge with samples already on disk
        if append_samples and samples_to_save is not None:
            import os
            _path = path if path.endswith(".npz") else path + ".npz"
            if os.path.exists(_path):
                existing = np.load(_path)
                if "samples" in existing and existing["samples"].size > 0:
                    # multi-chain: (n_chains, n_samples, n_params) → concat on axis=1
                    # single-chain: (n_samples, n_params) → concat on axis=0
                    cat_axis = 1 if samples_to_save.ndim == 3 else 0
                    samples_to_save = np.concatenate(
                        [existing["samples"], samples_to_save], axis=cat_axis)
                existing.close()

        save_kwargs = {
            # NUTS state (shape has leading chain dim when n_chains > 1)
            "position": np.asarray(self._last_state.position),
            "logdensity": np.asarray(self._last_state.logdensity),
            "logdensity_grad": np.asarray(self._last_state.logdensity_grad),
            # Adapted kernel parameters
            "step_size": np.asarray(self._adapted_step_size),
            "inverse_mass_matrix": np.asarray(self._adapted_inv_mass),
            "rng_key": np.asarray(self._last_rng_key),
            # Diagnostics (scalars)
            "n_warmup": np.asarray(self._info["n_warmup"]),
            "n_chains": np.asarray(getattr(self, "_n_chains", 1)),
        }

        if samples_to_save is not None:
            save_kwargs["samples"] = samples_to_save
            n_on_disk = samples_to_save.shape[0]
        else:
            save_kwargs["samples"] = np.array([])
            n_on_disk = 0

        np.savez(path, **save_kwargs)

        if append_samples:
            # Free in-memory samples and per-sample diagnostics
            self.samples = None
            self._info = {
                "step_size": self._info["step_size"],
                "n_warmup": self._info["n_warmup"],
                "n_samples": n_on_disk,
                "n_divergent": self._info.get("n_divergent", 0),
            }

        print(f"Checkpoint saved to {path} ({n_on_disk} samples on disk)")

        if plot_corner and n_on_disk > 0:
            import corner
            import matplotlib.pyplot as plt
            all_samples = self.load_samples(path)
            fig = corner.corner(
                all_samples,
                labels=list(self.param_keys),
                show_titles=True,
                title_fmt=".3f",
            )
            _chk = path if path.endswith(".npz") else path + ".npz"
            corner_dir = self.save_dir if self.save_dir is not None \
                else os.path.dirname(os.path.abspath(_chk))
            corner_path = os.path.join(corner_dir, "corner_plot.png")
            fig.savefig(corner_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Corner plot saved to {corner_path} "
                  f"({n_on_disk} samples)")

    def load_checkpoint(self, path):
        """
        Restore sampler state from a checkpoint file.

        Loads only the NUTS state and adapted kernel parameters needed
        to resume sampling.  Samples stored in the file are **not**
        loaded into memory — use ``load_samples`` to read them later.

        Parameters
        ----------
        path : str
            Path to a ``.npz`` checkpoint file.
        """
        import blackjax

        data = np.load(path)

        # Reconstruct NUTS state (works for both single and multi-chain:
        # arrays have a leading chain dimension when n_chains > 1)
        self._last_state = blackjax.mcmc.hmc.HMCState(
            position=jnp.asarray(data["position"]),
            logdensity=jnp.asarray(data["logdensity"]),
            logdensity_grad=jnp.asarray(data["logdensity_grad"]),
        )

        n_chains = int(data["n_chains"]) if "n_chains" in data else 1
        self._n_chains = n_chains

        if n_chains > 1:
            self._adapted_step_size = np.asarray(data["step_size"])
        else:
            self._adapted_step_size = float(data["step_size"])
        self._adapted_inv_mass = jnp.asarray(data["inverse_mass_matrix"])
        self._last_rng_key = jnp.asarray(data["rng_key"])

        n_on_disk = data["samples"].shape[0] if data["samples"].size > 0 else 0
        n_warmup = int(data["n_warmup"])
        data.close()

        # Don't load samples into memory — keep it lightweight
        self.samples = None
        self._info = {
            "step_size": self._adapted_step_size,
            "n_warmup": n_warmup,
            "n_samples": n_on_disk,
            "n_chains": n_chains,
            "n_divergent": 0,
        }

        print(f"Checkpoint loaded from {path} "
              f"({n_on_disk} samples on disk, {n_chains} chain(s), "
              f"not loaded into memory)")

    @staticmethod
    def load_samples(path, flatten_chains=True):
        """
        Read all samples from a checkpoint file without loading
        the sampler state.

        Parameters
        ----------
        path : str
            Path to a ``.npz`` checkpoint file.
        flatten_chains : bool
            If True (default), collapse the chain dimension so the
            returned array is always ``(n_total, n_params)``.  Set to
            False to get the raw ``(n_chains, n_samples, n_params)``
            array for per-chain diagnostics (e.g. R-hat).

        Returns
        -------
        samples : np.ndarray
            Shape ``(n_total, n_params)`` when ``flatten_chains=True``,
            or ``(n_chains, n_samples, n_params)`` otherwise.
        """
        data = np.load(path)
        samples = data["samples"].copy()
        data.close()
        if flatten_chains and samples.ndim == 3:
            n_chains, n_samp, n_params = samples.shape
            samples = samples.reshape(n_chains * n_samp, n_params)
        return samples

    def resume_nuts(self, n_samples=1000, n_chains=None, rng_key=None, progress_bar=False):
        """
        Continue NUTS sampling from a previous run or loaded checkpoint.

        Skips warmup entirely and uses the previously adapted step size
        and mass matrix (shared across all chains).  Returns only the
        new batch of samples.  Call ``save_checkpoint`` afterward to
        append the batch to disk and free memory.

        Parameters
        ----------
        n_samples : int
            Number of additional samples per chain (default 1000).
        n_chains : int, optional
            Number of chains to run.  If None (default), uses the value
            stored in the sampler state (from ``run_nuts`` or
            ``load_checkpoint``).
        rng_key : jax.random.PRNGKey, optional
            Random key.  If None, advances from the last key used.
        progress_bar : bool
            If True, print periodic progress updates (default False).
            Only supported for single-chain runs.

        Returns
        -------
        samples : jnp.ndarray
            Shape ``(n_samples, n_params)`` for single chain, or
            ``(n_chains, n_samples, n_params)`` for multiple chains.
        info : dict
            Diagnostics for this batch.
        """
        import blackjax

        if self._last_state is None:
            raise RuntimeError(
                "No previous state. Run run_nuts or load_checkpoint first.")

        if n_chains is None:
            n_chains = getattr(self, "_n_chains", 1)
        else:
            self._n_chains = n_chains

        n_devices = self._n_devices
        if n_chains > 1 and n_chains % n_devices != 0:
            raise ValueError(
                f"n_chains ({n_chains}) must be divisible by the number of "
                f"available devices ({n_devices}). Use n_chains in "
                f"{[n_devices * i for i in range(1, 5)]}.")

        gp = self.gp
        step_size = self._adapted_step_size
        inv_mass = jnp.asarray(self._adapted_inv_mass)

        # Warm up the log_posterior JIT kernel so CUDA timers are
        # accurate when the sampling scan launches.
        warmup_pos = self._last_state.position
        if warmup_pos.ndim > 1:
            warmup_pos = warmup_pos[0]
        jax.block_until_ready(gp.log_posterior(warmup_pos))

        def _run_one_chain(state, chain_key):
            kernel = blackjax.nuts(
                gp.log_posterior,
                step_size=step_size,
                inverse_mass_matrix=inv_mass,
            )
            chain_keys = jax.random.split(chain_key, n_samples)

            def one_step(carry, key_idx):
                st, n_div = carry
                key, _idx = key_idx
                st, info = kernel.step(key, st)
                n_div = n_div + info.is_divergent.astype(jnp.int32)
                return (st, n_div), (st.position, info)

            indices = jnp.arange(n_samples)
            (final_st, total_div), (positions, infos) = jax.lax.scan(
                one_step, (state, jnp.int32(0)), (chain_keys, indices),
            )
            return final_st, total_div, positions, infos, chain_keys[-1]

        if n_chains > 1:
            chains_per_device = n_chains // n_devices

            states = self._last_state

            if rng_key is None:
                last_key = self._last_rng_key
                # If resuming from a single-chain warmup checkpoint, the
                # stored key is a single PRNGKey rather than an array of
                # per-chain keys.  Use it directly instead of indexing.
                if last_key.ndim == 1:
                    rng_key = jax.random.fold_in(last_key, 1)
                else:
                    rng_key = jax.random.fold_in(last_key[0], 1)

            # If the checkpoint is from a single-chain warmup, expand
            # into multi-chain states by jittering the position.
            is_single_chain_state = states.position.ndim == 1
            if is_single_chain_state:
                jitter_key, rng_key = jax.random.split(rng_key)
                jitter_scale = 0.01 * jnp.sqrt(inv_mass)
                noise = jax.random.normal(
                    jitter_key,
                    shape=(n_chains, len(states.position)))
                init_positions = (states.position[None, :]
                                  + jitter_scale[None, :] * noise)
                init_fn = blackjax.nuts(
                    gp.log_posterior,
                    step_size=step_size,
                    inverse_mass_matrix=inv_mass,
                ).init
                states = jax.vmap(init_fn)(init_positions)

            chain_keys = jax.random.split(rng_key, n_chains)

            print(f"Resuming: {n_chains} chains x {n_samples} samples "
                  f"across {n_devices} device(s)...")

            # Reshape for pmap: (n_devices, chains_per_device, ...)
            states = jax.tree.map(
                lambda x: x.reshape(n_devices, chains_per_device, *x.shape[1:]),
                states)
            chain_keys = chain_keys.reshape(n_devices, chains_per_device, -1)

            all_final, all_div, all_pos, all_infos, all_last_keys = \
                jax.pmap(jax.vmap(_run_one_chain))(states, chain_keys)

            # Flatten device dimension back to (n_chains, ...)
            all_final = jax.tree.map(
                lambda x: x.reshape(n_chains, *x.shape[2:]), all_final)
            all_div = all_div.reshape(n_chains)
            all_pos = all_pos.reshape(n_chains, n_samples, -1)
            all_infos = jax.tree.map(
                lambda x: x.reshape(n_chains, *x.shape[2:]), all_infos)
            all_last_keys = all_last_keys.reshape(n_chains, -1)

            self.samples = all_pos
            self._info = {
                "divergences": np.asarray(all_infos.is_divergent),
                "acceptance_rate": np.asarray(all_infos.acceptance_rate),
                "num_steps": np.asarray(
                    all_infos.num_integration_steps),
                "step_size": step_size,
                "n_warmup": self._info.get("n_warmup", 0),
                "n_samples": n_samples,
                "n_chains": n_chains,
                "n_divergent": int(jnp.sum(all_div)),
            }

            self._last_state = all_final
            self._last_rng_key = all_last_keys

            total_div = int(jnp.sum(all_div))
            mean_accept = float(jnp.mean(
                jnp.array(self._info["acceptance_rate"])))
            print(f"Resume complete: {n_chains} chains x {n_samples} "
                  f"new samples, {total_div} divergences, "
                  f"mean acceptance rate = {mean_accept:.3f}")
        else:
            state = self._last_state

            if rng_key is None:
                rng_key = jax.random.split(self._last_rng_key)[0]

            print(f"Resuming: {n_samples} additional samples "
                  f"(step_size={step_size:.6f})...")

            final_state, new_div, new_positions, new_infos, last_key = \
                _run_one_chain(state, rng_key)

            self.samples = new_positions
            self._info = {
                "step_size": step_size,
                "n_warmup": self._info.get("n_warmup", 0),
                "n_samples": n_samples,
                "n_chains": 1,
                "n_divergent": int(new_div),
                "divergences": np.asarray(new_infos.is_divergent),
                "acceptance_rate": np.asarray(new_infos.acceptance_rate),
                "num_steps": np.asarray(new_infos.num_integration_steps),
            }

            self._last_state = final_state
            self._last_rng_key = last_key

            mean_accept = float(np.mean(self._info["acceptance_rate"]))
            print(f"Resume complete: {n_samples} new samples, "
                  f"{int(new_div)} divergences, "
                  f"mean acceptance rate = {mean_accept:.3f}")

        return self.samples, self._info
