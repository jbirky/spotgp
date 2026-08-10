"""Fitting methods (MAP, ACF, ACF+PSD), mixed into GPSolver."""

import logging
import time as _time

import jax
import jax.numpy as jnp
import numpy as np

from .analytic_kernel import _kernel_eval
from .validation import (
    raise_cholesky_error, format_nan_gradient_warning,
)

logger = logging.getLogger("spotgp")


class FittingMixin:
    """MAP and ACF fitting methods for GPSolver."""

    def fit_acf(self, theta0=None, keys=None, tlags=None, n_bins=50,
                method="L-BFGS-B", maxiter=500, ftol=0, gtol=1e-8,
                disp=False, nopt=1, ncore=None, rng=None, _save=True):
        """
        Fit the analytic kernel to the empirical ACF via least-squares.

        Minimizes sum_i (ACF_data(lag_i) - K(lag_i; theta))^2 over the
        kernel hyperparameters, using JAX gradients and scipy.

        Parameters
        ----------
        theta0 : dict or array_like, optional
            Starting point. Can be:
              - None: uses self.theta0 (kernel params only, no sigma_n).
              - dict: values for any subset of kernel keys set the
                starting point. If ``keys`` is not given, the dict
                keys that overlap with ``KERNEL_HPARAM_KEYS`` are
                treated as the free variables; the rest are held fixed.
                Extra keys not in ``KERNEL_HPARAM_KEYS`` are ignored.
              - array_like: full kernel theta vector (6 elements).
        keys : list of str, optional
            Which parameters to vary during optimization. Overrides
            the automatic inference from a dict ``theta0``. Parameters
            not listed are held fixed at their current values. If None
            and theta0 is not a dict, all kernel parameters are varied.
        tlags : array_like, optional
            Bin edges for compute_acf. If None, linearly spaced from 0 to
            half the baseline with n_bins+1 edges.
        n_bins : int
            Number of lag bins (used when tlags is None).
        method : str
            Scipy optimizer method.
        maxiter : int
            Maximum iterations.
        ftol : float
            Function-value convergence tolerance (default 0, disabled).
        gtol : float
            Gradient-norm convergence tolerance (default 1e-8).
        disp : bool
            If True, print optimizer convergence messages (default False).

        nopt : int
            Number of independent optimisation trials (default 1).
            When > 1, ``fit_acf_parallel`` is called and the best
            result across all trials is returned.
        ncore : int or None
            Number of parallel workers. Only used when ``nopt > 1``.
        rng : numpy.random.Generator, optional
            RNG for random starting points. Only used when ``nopt > 1``.

        Returns
        -------
        theta_dict : dict
            Full dictionary of all kernel hyperparameters (fixed + optimized).
        result : scipy.optimize.OptimizeResult
            Full optimizer output.
        """
        if nopt > 1:
            return self.fit_acf_parallel(
                nopt=nopt, ncore=ncore, keys=keys, tlags=tlags,
                n_bins=n_bins, method=method, maxiter=maxiter,
                ftol=ftol, gtol=gtol, disp=disp, rng=rng,
            )

        from scipy.optimize import minimize

        # Build lag bins
        if tlags is None:
            baseline = float(jnp.max(self.x) - jnp.min(self.x))
            tlags = np.linspace(0, baseline / 2, n_bins + 1)

        # Compute empirical ACF (unnormalized so units match the kernel)
        lag_centers, acf_data = self.compute_acf(tlags=tlags, n_bins=n_bins,
                                                  normalize=False)
        lag_centers_jax = jnp.asarray(lag_centers)
        acf_data_jax = jnp.asarray(acf_data)

        # --- Parse theta0 -------------------------------------------------
        # Use envelope-aware param_keys (excludes sigma_n)
        kernel_keys = list(self.spot_model.param_keys)
        n_kernel = len(kernel_keys)
        if theta0 is None:
            theta0_arr = self.theta0[:n_kernel]
        elif isinstance(theta0, dict):
            theta0_arr = self.theta0[:n_kernel].copy()
            dict_keys_in_kernel = []
            for k, v in theta0.items():
                if k in kernel_keys:
                    idx = kernel_keys.index(k)
                    theta0_arr = theta0_arr.at[idx].set(float(v))
                    dict_keys_in_kernel.append(k)
            if keys is None and dict_keys_in_kernel:
                keys = dict_keys_in_kernel
        else:
            theta0_arr = jnp.asarray(theta0, dtype=jnp.float64)

        # Resolve free vs fixed parameters (within kernel keys only)
        if keys is None:
            free_idx = list(range(n_kernel))
            fixed_idx = []
            fixed_vals = jnp.array([])
        else:
            for k in keys:
                if k not in kernel_keys:
                    raise ValueError(
                        f"Unknown key '{k}'. Valid kernel keys: {kernel_keys}")
            free_idx = [i for i, k in enumerate(kernel_keys) if k in keys]
            fixed_idx = [i for i, k in enumerate(kernel_keys) if k not in keys]
            fixed_vals = (theta0_arr[jnp.array(fixed_idx)]
                          if fixed_idx else jnp.array([]))

        free0_theta = theta0_arr[jnp.array(free_idx)]
        bounds_kernel = self.bounds[:n_kernel]
        free_bounds = bounds_kernel[jnp.array(free_idx)]
        blo = free_bounds[:, 0]
        bhi = free_bounds[:, 1]
        brange = bhi - blo

        # Optimize in normalized coordinates u = (theta - lo) / (hi - lo)
        u0 = np.asarray((free0_theta - blo) / brange, dtype=np.float64)

        qn, qw = self._quad_nodes, self._quad_weights
        n_h, n_l, lr = self.harmonics, self.n_lat, self.lat_range
        r_gamma_fn = self.spot_model.get_r_gamma_func()
        lat_wt_fn = self.spot_model.get_lat_weight_func()
        cn_sq_fn = self.spot_model.get_cn_sq_func(n_h)

        @jax.jit
        def loss_u(u_arr):
            free_theta = blo + u_arr * brange
            theta_full = self._theta_from_free(
                free_theta, free_idx, fixed_idx, fixed_vals)
            K_model = _kernel_eval(theta_full, lag_centers_jax,
                                   n_h, n_l, lr,
                                   quad_nodes=qn, quad_weights=qw,
                                   r_gamma_func=r_gamma_fn,
                                   lat_weight_func=lat_wt_fn,
                                   cn_sq_func=cn_sq_fn)
            return jnp.sum((acf_data_jax - K_model) ** 2)

        vg_fn = jax.jit(jax.value_and_grad(loss_u))

        logger.info("Compiling ACF fit objective (one-time cost)...")
        _t0 = _time.time()
        jax.block_until_ready(vg_fn(jnp.array(u0, dtype=jnp.float64)))
        logger.info("ACF fit compiled in %.2fs", _time.time() - _t0)

        n_free = len(free_idx)
        free_keys = [kernel_keys[i] for i in free_idx]
        _gradient_free = method.lower() in ("nelder-mead", "cobyla", "powell")
        _nan_grad_warned = [False]

        if _gradient_free:
            def objective(u_np):
                u_jax = jnp.array(u_np, dtype=jnp.float64)
                val, _ = vg_fn(u_jax)
                v = float(val)
                return 1e30 if not np.isfinite(v) else v
        else:
            def objective(u_np):
                u_jax = jnp.array(u_np, dtype=jnp.float64)
                val, grad = vg_fn(u_jax)
                v = float(val)
                g = np.asarray(grad, dtype=np.float64)
                if not np.isfinite(v):
                    return 1e30, np.zeros_like(g)
                if not np.all(np.isfinite(g)):
                    if not _nan_grad_warned[0]:
                        theta_here = blo + u_jax * brange
                        msg = format_nan_gradient_warning(
                            theta_here, g, free_keys, free_bounds)
                        logger.warning("fit_acf: %s", msg)
                        _nan_grad_warned[0] = True
                    return v, np.zeros_like(g)
                return v, g
        if _gradient_free:
            _minimize_kwargs = dict(
                method=method,
                options={"maxiter": maxiter, "xatol": ftol, "fatol": ftol,
                         "disp": disp},
            )
        else:
            _minimize_kwargs = dict(
                jac=True, method=method,
                bounds=[(0.0, 1.0)] * n_free,
                options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol,
                         "disp": disp},
            )
        result = minimize(objective, u0, **_minimize_kwargs)

        # Transform back to physical coordinates
        free_best = blo + jnp.array(result.x, dtype=jnp.float64) * brange
        theta_full = self._theta_from_free(
            free_best, free_idx, fixed_idx, fixed_vals)

        # Store results
        self.acf_fit_theta = theta_full
        self._acf_fit_result = result
        self._acf_lag_centers = lag_centers
        self._acf_data = acf_data

        theta_dict = {k: float(theta_full[i])
                      for i, k in enumerate(kernel_keys)}
        if _save:
            self._autosave("acf_fit_results.npz", theta_acf=theta_dict)
        return theta_dict, result


    def _fit_acf_batch_jaxopt(self, starts, keys, tlags, n_bins,
                              maxiter, gtol):
        """
        Optimize all ACF-fit ``starts`` with one vmapped ``jaxopt.LBFGSB``
        program (see ``_fit_map_batch_jaxopt``).

        Builds the same least-squares ACF loss as ``fit_acf`` (kernel
        evaluated at the empirical-ACF bin centers) and runs every
        restart inside a single compiled program.

        Returns
        -------
        results : list of (dict, scipy.optimize.OptimizeResult)
            One ``(theta_dict, result)`` pair per start, in input order,
            matching the return format of ``fit_acf``.

        Raises
        ------
        ImportError
            If jaxopt is not installed.
        """
        from scipy.optimize import OptimizeResult

        # Empirical ACF, exactly as in fit_acf
        user_tlags = tlags
        if tlags is None:
            baseline = float(jnp.max(self.x) - jnp.min(self.x))
            tlags = np.linspace(0, baseline / 2, n_bins + 1)
        lag_centers, acf_data = self.compute_acf(tlags=tlags, n_bins=n_bins,
                                                 normalize=False)
        lag_centers_jax = jnp.asarray(lag_centers)
        acf_data_jax = jnp.asarray(acf_data)

        # Free/fixed resolution over kernel keys, as in fit_acf with
        # theta0=None (multi-start draws replace the starting point).
        kernel_keys = list(self.spot_model.param_keys)
        n_kernel = len(kernel_keys)
        theta0_arr = self.theta0[:n_kernel]
        if keys is None:
            free_idx = list(range(n_kernel))
            fixed_idx = []
            fixed_vals = jnp.array([])
        else:
            free_idx = [i for i, k in enumerate(kernel_keys) if k in keys]
            fixed_idx = [i for i, k in enumerate(kernel_keys)
                         if k not in keys]
            fixed_vals = (theta0_arr[jnp.array(fixed_idx)]
                          if fixed_idx else jnp.array([]))
        free_keys = [kernel_keys[i] for i in free_idx]

        bounds_kernel = np.asarray(self.bounds[:n_kernel])
        blo_np = bounds_kernel[np.array(free_idx), 0]
        brange_np = bounds_kernel[np.array(free_idx), 1] - blo_np
        blo = jnp.asarray(blo_np)
        brange = jnp.asarray(brange_np)

        qn, qw = self._quad_nodes, self._quad_weights
        n_h, n_l, lr = self.harmonics, self.n_lat, self.lat_range
        r_gamma_fn = self.spot_model.get_r_gamma_func()
        lat_wt_fn = self.spot_model.get_lat_weight_func()
        cn_sq_fn = self.spot_model.get_cn_sq_func(n_h)

        def loss_u(u_arr):
            free_theta = blo + u_arr * brange
            theta_full = self._theta_from_free(
                free_theta, free_idx, fixed_idx, fixed_vals)
            K_model = _kernel_eval(theta_full, lag_centers_jax,
                                   n_h, n_l, lr,
                                   quad_nodes=qn, quad_weights=qw,
                                   r_gamma_func=r_gamma_fn,
                                   lat_weight_func=lat_wt_fn,
                                   cn_sq_func=cn_sq_fn)
            return jnp.sum((acf_data_jax - K_model) ** 2)

        vg_fn = jax.value_and_grad(loss_u)

        u0_batch = np.array(
            [[(s[k] - blo_np[j]) / brange_np[j]
              for j, k in enumerate(free_keys)] for s in starts])

        # Cache the compiled runner (the empirical ACF is baked into the
        # compiled objective, so only the default-binning configuration
        # is cached; explicit tlags always compile fresh).
        cache = self.__dict__.setdefault("_lbfgsb_runners", {})
        cache_key = (("acf", tuple(free_idx), int(n_bins),
                      int(maxiter), float(gtol))
                     if user_tlags is None else None)
        runner = cache.get(cache_key) if cache_key is not None else None
        if runner is None:
            from .gp_solver import _make_lbfgsb_batch_runner
            runner = _make_lbfgsb_batch_runner(
                vg_fn, len(free_idx), maxiter, gtol)
            if cache_key is not None:
                cache[cache_key] = runner

        u, fun, nit = runner(u0_batch)

        # Stash the ACF used, as fit_acf does
        self._acf_lag_centers = lag_centers
        self._acf_data = acf_data

        results = []
        for i in range(len(starts)):
            free_best = jnp.asarray(blo_np + u[i] * brange_np)
            theta_full = self._theta_from_free(
                free_best, free_idx, fixed_idx, fixed_vals)
            theta_dict = {k: float(theta_full[j])
                          for j, k in enumerate(kernel_keys)}
            result = OptimizeResult(
                x=u[i], fun=float(fun[i]), nit=int(nit[i]),
                success=bool(np.isfinite(fun[i])),
                message="jaxopt.LBFGSB (vmapped multi-start)")
            results.append((theta_dict, result))
        return results


    def fit_acf_parallel(self, nopt=10, ncore=None, keys=None,
                         tlags=None, n_bins=50, method="nelder-mead",
                         maxiter=500, ftol=0, gtol=1e-8, disp=False,
                         return_all=False, rng=None, batch=False):
        """
        Run ``fit_acf`` from multiple random starting points in parallel.

        Starting points are drawn uniformly within the kernel parameter
        bounds.

        Parameters
        ----------
        nopt : int
            Number of independent optimization trials (default 10).
        ncore : int or None
            Number of parallel workers. If None, uses ``nopt`` or the
            number of available CPUs, whichever is smaller.
        keys : list of str, optional
            Free parameters (forwarded to ``fit_acf``).
        tlags, n_bins
            Forwarded to ``fit_acf``.
        method : str
            Optimizer method (default "nelder-mead").
        batch : bool
            If True and ``method="L-BFGS-B"`` (and jaxopt is installed),
            run all restarts as a single vmapped ``jaxopt.LBFGSB``
            program (see ``fit_map_parallel``; the compiled program is
            cached on the solver when ``tlags`` is None).
        maxiter, ftol, gtol, disp
            Forwarded to ``fit_acf``.
        return_all : bool
            If True, return all solutions sorted by objective value.
            If False (default), return only the best solution.
        rng : numpy.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        theta_best : dict  (or list of dict if ``return_all=True``)
            Best-fit kernel hyperparameters.
        result_best : scipy.optimize.OptimizeResult
            (or list of OptimizeResult if ``return_all=True``)
        """
        from concurrent.futures import ThreadPoolExecutor
        import os

        if ncore is None:
            ncore = min(nopt, os.cpu_count() or 1)
        if rng is None:
            rng = np.random.default_rng()

        # Determine which kernel indices are free
        kernel_keys = list(self.spot_model.param_keys)
        n_kernel = len(kernel_keys)
        if keys is None:
            free_keys = list(kernel_keys)
            free_bounds_np = np.asarray(self.bounds[:n_kernel])
        else:
            for k in keys:
                if k not in kernel_keys:
                    raise ValueError(
                        f"Unknown key '{k}'. Valid kernel keys: {kernel_keys}")
            free_keys = [k for k in kernel_keys if k in keys]
            idx = [kernel_keys.index(k) for k in free_keys]
            free_bounds_np = np.asarray(self.bounds[jnp.array(idx)])
        blo = free_bounds_np[:, 0]
        bhi = free_bounds_np[:, 1]

        # Generate random starting points using independent child RNGs
        seeds = rng.integers(0, 2**31, size=nopt)
        starts = []
        for i in range(nopt):
            child_rng = np.random.default_rng(int(seeds[i]))
            theta0_dict = {}
            u = child_rng.uniform(size=len(free_keys))
            for j, k in enumerate(free_keys):
                theta0_dict[k] = float(blo[j] + u[j] * (bhi[j] - blo[j]))
            starts.append(theta0_dict)

        # Opt-in fast path: for L-BFGS-B, run all restarts as one
        # vmapped jaxopt.LBFGSB program (see fit_map_parallel).
        results = None
        if batch and method.lower() in ("l-bfgs-b", "lbfgsb"):
            try:
                results = self._fit_acf_batch_jaxopt(
                    starts, keys=keys, tlags=tlags, n_bins=n_bins,
                    maxiter=maxiter, gtol=gtol)
            except ImportError:
                results = None  # jaxopt not installed — use threads below

        if results is None:
            def _run_one(theta0_dict):
                return self.fit_acf(theta0=theta0_dict, keys=keys,
                                    tlags=tlags, n_bins=n_bins,
                                    method=method, maxiter=maxiter,
                                    ftol=ftol, gtol=gtol, disp=disp,
                                    _save=False)

            with ThreadPoolExecutor(max_workers=ncore) as pool:
                futures = [pool.submit(_run_one, s) for s in starts]
                results = [f.result() for f in futures]

        # Sort by objective value (lower is better)
        results.sort(key=lambda tr: float(tr[1].fun))

        if return_all:
            return ([r[0] for r in results], [r[1] for r in results])

        best_theta, best_result = results[0]
        # Store the best
        self.acf_fit_theta = jnp.array(
            [float(best_theta[k]) for k in kernel_keys],
            dtype=jnp.float64)
        self._acf_fit_result = best_result
        theta_all = np.array(
            [[float(r[0][k]) for k in kernel_keys] for r in results])
        self._autosave("acf_fit_results.npz",
                       theta_acf=best_theta, theta_all=theta_all)
        return best_theta, best_result


    def fit_acf_psd(self, theta0=None, keys=None,
                    tlags=None, n_bins=50,
                    n_freq=200, dt_kernel=None,
                    acf_weight=1.0, psd_weight=1.0,
                    method="L-BFGS-B", maxiter=500, ftol=0, gtol=1e-8,
                    disp=False):
        """
        Fit kernel parameters jointly to the empirical ACF and PSD.

        Minimizes a weighted sum of two normalized mean-squared-error terms:

            loss = acf_weight * acf_loss + psd_weight * psd_loss

        where

            acf_loss = mean((ACF_data - K_model)^2) / mean(ACF_data^2)

        is the relative MSE of the kernel against the empirical ACF
        (unnormalized autocovariance), and

            psd_loss = mean((PSD_data_norm - PSD_model_norm)^2)

        is the MSE between the Lomb-Scargle periodogram and the analytic
        kernel PSD, both normalized to unit integral so the comparison is
        independent of overall amplitude.

        The model PSD is computed via a direct cosine transform of the kernel
        evaluated on a uniform lag grid, making it fully differentiable with
        respect to the kernel parameters.

        Parameters
        ----------
        theta0 : dict or array_like, optional
            Starting point in ``self.param_keys`` space (sampling space,
            with ``log_``-prefixed keys where applicable).  Follows the same
            convention as ``fit_map``: None uses ``self.theta0``, a dict
            overrides named entries and infers free keys, an array is used
            directly.
        keys : list of str, optional
            Parameters to vary during optimization (names from
            ``self.param_keys``).  Defaults to all kernel parameters
            (first 6 entries of ``self.param_keys``, i.e. excluding
            ``sigma_n`` if present).
        tlags : array_like, optional
            Bin edges for the empirical ACF. If None, ``n_bins+1`` edges
            linearly spaced from 0 to half the baseline.
        n_bins : int
            Number of ACF lag bins when ``tlags`` is None (default 50).
        n_freq : int
            Number of frequency points for the Lomb-Scargle periodogram
            (default 200).
        dt_kernel : float, optional
            Uniform lag spacing [days] for evaluating the analytic kernel
            before the direct cosine transform.  Defaults to one-fifth of
            the median data spacing.
        acf_weight : float
            Weight for the ACF loss term (default 1.0).
        psd_weight : float
            Weight for the PSD loss term (default 1.0).
        method : str
            Scipy optimizer method (default ``"L-BFGS-B"``).
        maxiter : int
            Maximum optimizer iterations (default 500).
        ftol, gtol : float
            Convergence tolerances forwarded to scipy.
        disp : bool
            Print optimizer messages if True.

        Returns
        -------
        theta_dict : dict
            Best-fit parameters in ``self.param_keys`` space.
        result : scipy.optimize.OptimizeResult
        """
        from scipy.optimize import minimize
        from scipy.signal import lombscargle

        n_kernel = len(self.spot_model.param_keys)  # envelope-dependent
        to_phys = self._to_physical
        qn, qw = self._quad_nodes, self._quad_weights
        n_h, n_l, lr = self.harmonics, self.n_lat, self.lat_range
        r_gamma_fn = self.spot_model.get_r_gamma_func()
        lat_wt_fn = self.spot_model.get_lat_weight_func()
        cn_sq_fn = self.spot_model.get_cn_sq_func(n_h)
        w_acf = float(acf_weight)
        w_psd = float(psd_weight)

        x     = np.asarray(self.x)
        resid = np.asarray(self.y) - self.mean_val
        dt_med   = float(np.median(np.diff(x)))
        baseline = float(x[-1] - x[0])

        # --- Empirical ACF (skipped when acf_weight == 0) -------------------
        if w_acf != 0.0:
            if tlags is None:
                tlags = np.linspace(0, baseline / 2, n_bins + 1)
            lag_centers, acf_data = self.compute_acf(tlags, normalize=False)
            lag_jax = jnp.asarray(lag_centers, dtype=jnp.float64)
            acf_jax = jnp.asarray(acf_data,    dtype=jnp.float64)
            acf_rms = jnp.sqrt(jnp.mean(acf_jax ** 2)) + 1e-30
        else:
            lag_jax = acf_jax = acf_rms = None

        # --- Empirical PSD via Lomb-Scargle (skipped when psd_weight == 0) --
        if w_psd != 0.0:
            freq_min  = 1.0 / baseline
            freq_max  = 1.0 / (2.0 * dt_med)
            freqs     = np.linspace(freq_min, freq_max, n_freq)
            pgram     = lombscargle(x, resid, 2.0 * np.pi * freqs, normalize=False)
            df        = freqs[1] - freqs[0]
            psd_data_norm = pgram / (np.sum(pgram) * df)
            psd_jax   = jnp.asarray(psd_data_norm, dtype=jnp.float64)
            freqs_jax = jnp.asarray(freqs,         dtype=jnp.float64)
            if dt_kernel is None:
                dt_kernel = dt_med / 5.0
            tau_grid = np.arange(0.0, baseline, dt_kernel)
            tau_jax  = jnp.asarray(tau_grid, dtype=jnp.float64)
            cos_mat  = jnp.cos(2.0 * jnp.pi * tau_jax[:, None] * freqs_jax[None, :])
        else:
            psd_jax = freqs_jax = tau_jax = cos_mat = df = None

        # --- Parse theta0 ---------------------------------------------------
        if theta0 is None:
            theta0_arr = self.theta0[:n_kernel].copy()
        elif isinstance(theta0, dict):
            theta0_arr = self.theta0[:n_kernel].copy()
            dict_keys_in_params = []
            for k, v in theta0.items():
                if k in self.param_keys[:n_kernel]:
                    idx = list(self.param_keys).index(k)
                    theta0_arr = theta0_arr.at[idx].set(float(v))
                    dict_keys_in_params.append(k)
            if keys is None and dict_keys_in_params:
                keys = dict_keys_in_params
        else:
            theta0_arr = jnp.asarray(theta0, dtype=jnp.float64)

        # --- Resolve free vs fixed parameters -------------------------------
        kernel_param_keys = list(self.param_keys[:n_kernel])
        if keys is None:
            free_idx  = list(range(n_kernel))
            fixed_idx = []
            fixed_vals = jnp.array([])
        else:
            for k in keys:
                if k not in kernel_param_keys:
                    raise ValueError(
                        f"Unknown key '{k}'. Valid keys: {kernel_param_keys}")
            free_idx  = [i for i, k in enumerate(kernel_param_keys) if k in keys]
            fixed_idx = [i for i, k in enumerate(kernel_param_keys) if k not in keys]
            fixed_vals = (theta0_arr[jnp.array(fixed_idx)]
                          if fixed_idx else jnp.array([]))

        free0     = theta0_arr[jnp.array(free_idx)]
        bds       = self.bounds[:n_kernel]
        free_bds  = bds[jnp.array(free_idx)]
        blo, bhi  = free_bds[:, 0], free_bds[:, 1]
        brange    = bhi - blo
        u0        = np.asarray((free0 - blo) / brange, dtype=np.float64)

        @jax.jit
        def loss_u(u_arr):
            free_theta  = blo + u_arr * brange
            theta_samp  = self._theta_from_free(
                free_theta, free_idx, fixed_idx, fixed_vals)
            theta_phys  = to_phys(theta_samp)

            loss = 0.0

            if w_acf != 0.0:
                K_acf    = _kernel_eval(theta_phys, lag_jax, n_h, n_l, lr,
                                        quad_nodes=qn, quad_weights=qw,
                                        r_gamma_func=r_gamma_fn,
                                        lat_weight_func=lat_wt_fn,
                                        cn_sq_func=cn_sq_fn)
                acf_loss = jnp.mean(((acf_jax - K_acf) / acf_rms) ** 2)
                loss = loss + w_acf * acf_loss

            if w_psd != 0.0:
                K_tau = _kernel_eval(theta_phys, tau_jax, n_h, n_l, lr,
                                     quad_nodes=qn, quad_weights=qw,
                                     r_gamma_func=r_gamma_fn,
                                     lat_weight_func=lat_wt_fn,
                                     cn_sq_func=cn_sq_fn)
                psd_model = jnp.maximum(
                    dt_kernel * (K_tau[0] + 2.0 * jnp.dot(K_tau[1:], cos_mat[1:])),
                    0.0)
                psd_norm  = jnp.sum(psd_model) * df
                psd_model_norm = psd_model / (psd_norm + 1e-30)
                psd_loss = jnp.mean((psd_jax - psd_model_norm) ** 2)
                loss = loss + w_psd * psd_loss

            return loss

        vg_fn = jax.jit(jax.value_and_grad(loss_u))

        logger.info("Compiling ACF+PSD fit objective (one-time cost)...")
        _t0 = _time.time()
        jax.block_until_ready(vg_fn(jnp.array(u0, dtype=jnp.float64))[0])
        logger.info("ACF+PSD fit compiled in %.2fs", _time.time() - _t0)

        n_free = len(free_idx)
        free_keys = [kernel_param_keys[i] for i in free_idx]
        _gradient_free = method.lower() in ("nelder-mead", "cobyla", "powell")
        _nan_grad_warned = [False]

        if _gradient_free:
            def objective(u_np):
                val, _ = vg_fn(jnp.array(u_np, dtype=jnp.float64))
                v = float(val)
                return 1e30 if not np.isfinite(v) else v
        else:
            def objective(u_np):
                u_jax = jnp.array(u_np, dtype=jnp.float64)
                val, grad = vg_fn(u_jax)
                v = float(val)
                g = np.asarray(grad, dtype=np.float64)
                if not np.isfinite(v):
                    return 1e30, np.zeros_like(g)
                if not np.all(np.isfinite(g)):
                    if not _nan_grad_warned[0]:
                        theta_here = blo + u_jax * brange
                        free_bds_arr = bds[jnp.array(free_idx)]
                        msg = format_nan_gradient_warning(
                            theta_here, g, free_keys, free_bds_arr)
                        logger.warning("fit_acf_psd: %s", msg)
                        _nan_grad_warned[0] = True
                    return v, np.zeros_like(g)
                return v, g

        if _gradient_free:
            _minimize_kwargs = dict(
                method=method,
                options={"maxiter": maxiter, "xatol": ftol, "fatol": ftol,
                         "disp": disp},
            )
        else:
            _minimize_kwargs = dict(
                jac=True, method=method,
                bounds=[(0.0, 1.0)] * n_free,
                options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol,
                         "disp": disp},
            )
        result = minimize(objective, u0, **_minimize_kwargs)

        free_best  = blo + jnp.array(result.x, dtype=jnp.float64) * brange
        theta_full = self._theta_from_free(
            free_best, free_idx, fixed_idx, fixed_vals)

        theta_dict = {k: float(theta_full[i])
                      for i, k in enumerate(kernel_param_keys)}
        return theta_dict, result


    def _theta_dict_to_phys_array(self, theta):
        """Convert a theta dict or array to a physical kernel parameter array.

        Handles sampling-space dicts that may contain ``log_``-prefixed keys
        (e.g. ``log_sigma_k``), converting them to physical values via
        ``10 ** value`` before building the array.  Plain arrays are returned
        as-is (assumed already physical).

        Parameters
        ----------
        theta : dict or array_like
            Kernel parameters, either as a dict (physical or log-space keys)
            or an array in spot_model.param_keys order.

        Returns
        -------
        theta_arr : jnp.ndarray
            Physical kernel parameters in spot_model.param_keys order.
        """
        kernel_keys = list(self.spot_model.param_keys)
        if isinstance(theta, dict):
            phys = {}
            for k, v in theta.items():
                if k.startswith("log_"):
                    phys[k[4:]] = 10.0 ** float(v)
                else:
                    phys[k] = float(v)
            return jnp.array([float(phys[k]) for k in kernel_keys],
                             dtype=jnp.float64)
        return jnp.asarray(theta, dtype=jnp.float64)


    def fit_map(self, theta0=None, keys=None, method="L-BFGS-B",
                 maxiter=500, ftol=0, gtol=1e-8, disp=False, nopt=1,
                 ncore=None, rng=None, batch=False, _save=True):
        """
        Find the maximum a posteriori (MAP) estimate.

        Uses scipy.optimize.minimize with JAX-computed gradients.
        When ``nopt > 1``, delegates to ``fit_map_parallel`` which
        runs ``nopt`` independent trials from random starting points
        (drawn uniformly within the bounds) and returns the best result.

        Parameters
        ----------
        theta0 : dict or array_like, optional
            Starting point. Can be:
              - None: uses self.theta0 (current hyperparameters).
              - dict: values for any subset of param_keys set the
                starting point. If ``keys`` is not given, the dict
                keys that overlap with ``self.param_keys`` are treated
                as the free variables to optimize; the rest are held
                fixed. Extra keys not in ``param_keys`` are ignored.
              - array_like: full theta vector (length ``n_params``).

            Ignored when ``nopt > 1`` (starting points are randomised).
        keys : list of str, optional
            Which parameters to vary during optimization. Overrides
            the automatic inference from a dict ``theta0``. Parameters
            not listed are held fixed at their current values. If None
            and theta0 is not a dict, all parameters are varied.
        method : str
            Scipy optimizer method (default "L-BFGS-B").
        maxiter : int
            Maximum iterations.
        ftol : float
            Function-value convergence tolerance for L-BFGS-B
            (default 0, i.e. disabled so that convergence is
            controlled by ``gtol``).
        gtol : float
            Gradient-norm convergence tolerance (default 1e-8).
        disp : bool
            If True, print optimizer convergence messages (default False).
        nopt : int
            Number of independent optimisation trials (default 1).
            When > 1, ``fit_map_parallel`` is called and the best
            result across all trials is returned.
        ncore : int or None
            Number of parallel workers for multi-start runs. If None,
            uses ``nopt`` or the number of available CPUs, whichever is
            smaller. Only used when ``nopt > 1``.
        rng : numpy.random.Generator, optional
            RNG for random starting points. Only used when ``nopt > 1``.
        batch : bool
            Run multi-start restarts as one vmapped ``jaxopt.LBFGSB``
            program (see ``fit_map_parallel``). Only used when
            ``nopt > 1`` and ``method="L-BFGS-B"``.

        Returns
        -------
        theta_dict : dict
            Full dictionary of all hyperparameters (fixed + optimized).
        result : scipy OptimizeResult
            Full optimizer output.
        """
        if nopt > 1:
            return self.fit_map_parallel(
                nopt=nopt, ncore=ncore, keys=keys, method=method,
                maxiter=maxiter, ftol=ftol, gtol=gtol, disp=disp, rng=rng,
                theta0=theta0, batch=batch,
            )
        from scipy.optimize import minimize

        # --- Parse theta0 -------------------------------------------------
        if theta0 is None:
            theta0_arr = self.theta0.copy()
        elif isinstance(theta0, dict):
            # Build full array from current values, overriding with dict
            theta0_arr = self.theta0.copy()
            dict_keys_in_params = []
            for k, v in theta0.items():
                if k in self.param_keys:
                    idx = self.param_keys.index(k)
                    theta0_arr = theta0_arr.at[idx].set(float(v))
                    dict_keys_in_params.append(k)
            # Infer free keys from dict when keys is not explicitly given
            if keys is None and dict_keys_in_params:
                keys = dict_keys_in_params
        else:
            theta0_arr = jnp.asarray(theta0, dtype=jnp.float64)

        free_idx, fixed_idx, fixed_vals = self._resolve_keys(keys)

        free0_theta = theta0_arr[jnp.array(free_idx)]
        free_bounds = self.bounds[jnp.array(free_idx)]
        blo = free_bounds[:, 0]
        bhi = free_bounds[:, 1]
        brange = bhi - blo

        # Optimize in normalized coordinates u = (theta - lo) / (hi - lo)
        # so all free parameters live in [0, 1] with comparable scale.
        u0 = np.asarray((free0_theta - blo) / brange, dtype=np.float64)

        # Reuse the prebuilt compiled value-and-grad of the log-posterior,
        # so the objective is exactly the posterior that ``log_posterior``
        # and the MCMC samplers use (matrix solver, envelope R_Gamma,
        # edge-on and Toeplitz fast paths included).  The map from the
        # optimizer's u coordinates to theta is affine on the free indices
        # (theta = lo + u * range), so the chain rule is applied outside
        # XLA — repeated fit_map calls trigger no recompilation.
        vag = self.value_and_grad_log_posterior
        free_idx_arr = jnp.array(free_idx)

        def vg_fn(u_arr):
            free_theta = blo + u_arr * brange
            theta_full = self._theta_from_free(
                free_theta, free_idx, fixed_idx, fixed_vals)
            val, grad_theta = vag(theta_full)
            return -val, -(grad_theta[free_idx_arr] * brange)

        logger.info("Compiling MAP objective (one-time cost)...")
        _t0 = _time.time()
        try:
            jax.block_until_ready(
                vg_fn(jnp.array(u0, dtype=jnp.float64))[0])
        except Exception as e:
            theta_at_fail = blo + jnp.array(u0, dtype=jnp.float64) * brange
            theta_full_fail = self._theta_from_free(
                theta_at_fail, free_idx, fixed_idx, fixed_vals)
            raise_cholesky_error(
                e, theta=theta_full_fail,
                param_keys=self.param_keys, bounds=self.bounds,
                context="MAP optimization (initial evaluation)")
        logger.info("MAP objective compiled in %.2fs", _time.time() - _t0)

        n_free = len(free_idx)
        free_keys = [list(self.param_keys)[i] for i in free_idx]
        _gradient_free = method.lower() in ("nelder-mead", "cobyla", "powell")
        _nan_grad_warned = [False]

        if _gradient_free:
            def objective(u_np):
                u_jax = jnp.array(u_np, dtype=jnp.float64)
                val, _ = vg_fn(u_jax)
                v = float(val)
                return 1e30 if not np.isfinite(v) else v
        else:
            def objective(u_np):
                u_jax = jnp.array(u_np, dtype=jnp.float64)
                val, grad = vg_fn(u_jax)
                v = float(val)
                g = np.asarray(grad, dtype=np.float64)
                if not np.isfinite(v):
                    return 1e30, np.zeros_like(g)
                if not np.all(np.isfinite(g)):
                    if not _nan_grad_warned[0]:
                        theta_here = blo + u_jax * brange
                        free_bounds = self.bounds[jnp.array(free_idx)]
                        msg = format_nan_gradient_warning(
                            theta_here, g, free_keys, free_bounds)
                        logger.warning("fit_map: %s", msg)
                        _nan_grad_warned[0] = True
                    return v, np.zeros_like(g)
                return v, g
        if _gradient_free:
            _minimize_kwargs = dict(
                method=method,
                options={"maxiter": maxiter, "xatol": ftol, "fatol": ftol,
                         "disp": disp},
            )
        else:
            _minimize_kwargs = dict(
                jac=True, method=method,
                bounds=[(0.0, 1.0)] * n_free,
                options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol,
                         "disp": disp},
            )
        result = minimize(objective, u0, **_minimize_kwargs)

        # Transform back to physical coordinates
        free_best = blo + jnp.array(result.x, dtype=jnp.float64) * brange
        theta_full = self._theta_from_free(
            free_best, free_idx, fixed_idx, fixed_vals)

        self.map_estimate = theta_full
        self._map_result = result
        theta_dict = self._result_dict(theta_full)
        if _save:
            self._autosave("map_fit_results.npz", theta_map=theta_dict)
        return theta_dict, result


    def _fit_map_batch_jaxopt(self, starts, keys, maxiter, gtol):
        """
        Optimize all ``starts`` with one vmapped ``jaxopt.LBFGSB`` program.

        Reuses the prebuilt compiled ``value_and_grad_log_posterior``, so
        the objective is exactly the posterior that ``fit_map`` and the
        MCMC samplers use (matrix solver, envelope R_Gamma, edge-on and
        Toeplitz fast paths included).

        Parameters
        ----------
        starts : list of dict
            Starting points as ``{free_key: value}`` dicts in physical
            coordinates (the format produced by ``fit_map_parallel``).
        keys : list of str or None
            Free parameters, as in ``fit_map``.
        maxiter, gtol
            Forwarded to the solver.

        Returns
        -------
        results : list of (dict, scipy.optimize.OptimizeResult)
            One ``(theta_dict, result)`` pair per start, in input order,
            matching the return format of ``fit_map``.

        Raises
        ------
        ImportError
            If jaxopt is not installed.
        """
        from scipy.optimize import OptimizeResult

        free_idx, fixed_idx, fixed_vals = self._resolve_keys(keys)
        free_keys = [self.param_keys[i] for i in free_idx]
        free_bounds = np.asarray(self.bounds[jnp.array(free_idx)])
        blo_np = free_bounds[:, 0]
        brange_np = free_bounds[:, 1] - free_bounds[:, 0]
        blo = jnp.asarray(blo_np)
        brange = jnp.asarray(brange_np)

        vag = self.value_and_grad_log_posterior
        free_idx_arr = jnp.array(free_idx)

        def vg_fn(u_arr):
            free_theta = blo + u_arr * brange
            theta_full = self._theta_from_free(
                free_theta, free_idx, fixed_idx, fixed_vals)
            val, grad_theta = vag(theta_full)
            return -val, -(grad_theta[free_idx_arr] * brange)

        u0_batch = np.array(
            [[(s[k] - blo_np[j]) / brange_np[j]
              for j, k in enumerate(free_keys)] for s in starts])

        # Cache the compiled runner: tracing the vmapped optimizer is
        # expensive, so repeated calls with the same free parameters and
        # solver settings reuse the compiled program.
        cache = self.__dict__.setdefault("_lbfgsb_runners", {})
        cache_key = ("map", tuple(free_idx), int(maxiter), float(gtol))
        runner = cache.get(cache_key)
        if runner is None:
            from .gp_solver import _make_lbfgsb_batch_runner
            runner = _make_lbfgsb_batch_runner(
                vg_fn, len(free_idx), maxiter, gtol)
            cache[cache_key] = runner

        u, fun, nit = runner(u0_batch)

        results = []
        for i in range(len(starts)):
            free_best = jnp.asarray(blo_np + u[i] * brange_np)
            theta_full = self._theta_from_free(
                free_best, free_idx, fixed_idx, fixed_vals)
            result = OptimizeResult(
                x=u[i], fun=float(fun[i]), nit=int(nit[i]),
                success=bool(np.isfinite(fun[i])),
                message="jaxopt.LBFGSB (vmapped multi-start)")
            results.append((self._result_dict(theta_full), result))
        return results


    def fit_map_parallel(self, nopt=10, ncore=None, keys=None,
                          method="nelder-mead", maxiter=500, ftol=0,
                          gtol=1e-8, disp=False, return_all=False,
                          rng=None, theta0=None, jitter=0.01,
                          batch=False):
        """
        Run ``fit_map`` from multiple random starting points in parallel.

        Starting points are drawn uniformly within the parameter bounds.

        Parameters
        ----------
        nopt : int
            Number of independent optimization trials (default 10).
        ncore : int or None
            Number of parallel workers. If None, uses ``nopt`` or the
            number of available CPUs, whichever is smaller.
        keys : list of str, optional
            Free parameters (forwarded to ``fit_map``).
        method : str
            Optimizer method (default "nelder-mead").
        maxiter, ftol, gtol, disp
            Forwarded to ``fit_map``.
        batch : bool
            If True and ``method="L-BFGS-B"`` (and jaxopt is installed),
            run all restarts as a single vmapped ``jaxopt.LBFGSB``
            program instead of one scipy optimizer per thread
            (``ftol``, ``disp`` and ``ncore`` are ignored on this path).
            The first call pays a large one-off XLA compilation (~1 min
            for a banded solver on CPU) that is cached on the solver, so
            this wins when the same solver configuration is fit many
            times per session (per-call execution is faster than the
            thread pool and improves with ``nopt``), or on GPU.  For a
            single multi-start fit, the default thread pool is usually
            faster end-to-end.
        return_all : bool
            If True, return all solutions sorted by objective value.
            If False (default), return only the best solution.
        rng : numpy.random.Generator, optional
            Random number generator for reproducibility.
        theta0 : dict, optional
            Initial parameter guess to include as one of the starting
            points.  Replaces one random start so the total number of
            trials stays ``nopt``.

        Returns
        -------
        theta_best : dict  (or list of dict if ``return_all=True``)
            Best-fit hyperparameters.
        result_best : scipy.optimize.OptimizeResult
            (or list of OptimizeResult if ``return_all=True``)
        """
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
        import os

        if ncore is None:
            ncore = min(nopt, os.cpu_count() or 1)
        if rng is None:
            rng = np.random.default_rng()

        # Determine which indices are free
        free_idx, fixed_idx, _ = self._resolve_keys(keys)
        free_keys = [self.param_keys[i] for i in free_idx]
        free_bounds = np.asarray(self.bounds[jnp.array(free_idx)])
        blo = free_bounds[:, 0]
        bhi = free_bounds[:, 1]

        # Generate starting points using independent child RNGs.
        # If theta0 is provided, use it as the first start and fill the
        # rest with random draws so the total count stays nopt.
        if theta0 is not None:
            # For keys in theta0, use the provided value; for the rest,
            # draw a random starting point within bounds.
            child_rng0 = np.random.default_rng(int(rng.integers(0, 2**31)))
            u0 = child_rng0.uniform(size=len(free_keys))
            first_start = {}
            for j, k in enumerate(free_keys):
                if k in theta0:
                    # Add small jitter (1% of bound range) to theta0 values
                    span = bhi[j] - blo[j]
                    jitter_val = child_rng0.uniform(-jitter, jitter) * span
                    first_start[k] = float(np.clip(theta0[k] + jitter_val, blo[j], bhi[j]))
                else:
                    first_start[k] = float(blo[j] + u0[j] * (bhi[j] - blo[j]))
            starts = [first_start]
            n_random = max(nopt - 1, 0)
        else:
            starts = []
            n_random = nopt

        seeds = rng.integers(0, 2**31, size=n_random)
        for i in range(n_random):
            child_rng = np.random.default_rng(int(seeds[i]))
            theta0_dict = {}
            u = child_rng.uniform(size=len(free_keys))
            for j, k in enumerate(free_keys):
                theta0_dict[k] = float(blo[j] + u[j] * (bhi[j] - blo[j]))
            starts.append(theta0_dict)

        # Opt-in fast path: for L-BFGS-B, run all restarts as one
        # vmapped jaxopt.LBFGSB program (compiled once, cached on the
        # solver) instead of one scipy optimizer per thread.
        results = None
        if batch and method.lower() in ("l-bfgs-b", "lbfgsb"):
            try:
                results = self._fit_map_batch_jaxopt(
                    starts, keys=keys, maxiter=maxiter, gtol=gtol)
            except ImportError:
                results = None  # jaxopt not installed — use threads below

        if results is None:
            # Distribute work across available JAX devices (GPUs/TPUs)
            devices = jax.devices()
            n_devices = len(devices)

            def _run_one(device_idx, theta0_dict):
                with jax.default_device(devices[device_idx % n_devices]):
                    return self.fit_map(theta0=theta0_dict, keys=keys,
                                         method=method, maxiter=maxiter,
                                         ftol=ftol, gtol=gtol, disp=disp,
                                         _save=False)

            # Run the first trial sequentially to warm up JIT on device 0
            results = [_run_one(0, starts[0])]

            # Run remaining trials in parallel using threads, distributed
            # across devices.  JAX releases the GIL during compiled
            # computation, so threads give real parallelism.
            if len(starts) > 1:
                with ThreadPoolExecutor(max_workers=ncore) as pool:
                    futures = [pool.submit(_run_one, i + 1, s)
                               for i, s in enumerate(starts[1:])]
                    results.extend(f.result() for f in futures)

        # Sort by objective value (lower is better for neg_log_posterior)
        results.sort(key=lambda tr: float(tr[1].fun))

        if return_all:
            return ([r[0] for r in results], [r[1] for r in results])

        best_theta, best_result = results[0]
        # Store the best as the MAP estimate
        self.map_estimate = jnp.array(
            [float(best_theta[k]) for k in self.param_keys],
            dtype=jnp.float64)
        self._map_result = best_result
        theta_all = np.array(
            [[float(r[0][k]) for k in self.param_keys] for r in results])
        self._autosave("map_fit_results.npz",
                       theta_map=best_theta, theta_all=theta_all)
        return best_theta, best_result

    # =================================================================
    # Mass matrix helpers
    # =================================================================


