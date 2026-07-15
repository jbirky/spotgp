"""Dynesty nested sampler for GP hyperparameters."""

import logging
import os

import jax.numpy as jnp
import numpy as np

from .base import MCMCSampler

logger = logging.getLogger("spotgp")


class DynestySampler(MCMCSampler):
    """
    Nested sampler using the dynesty library.

    Computes the posterior and Bayesian evidence via nested sampling.
    Inherits diagnostics, summary, plotting, and dict conversion from
    MCMCSampler.

    Parameters
    ----------
    gp : GPSolver
        A configured GPSolver instance.
    prior_transform : callable, optional
        Function mapping the unit hypercube ``u`` (array of shape
        ``(n_params,)``) to the physical parameter space.  If None,
        a default uniform prior transform is constructed from the
        GPSolver bounds.
    save_dir : str, optional
        Directory for outputs.  Created automatically if it does not
        exist.
    checkpoint_file : str, optional
        Path to the checkpoint file (relative to ``save_dir``).
    """

    def __init__(self, gp, prior_transform=None, save_dir="results",
                 checkpoint_file="mcmc_checkpoint.npz"):
        super().__init__(gp)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        if checkpoint_file is not None:
            self._checkpoint_file = os.path.join(save_dir, checkpoint_file)
        else:
            self._checkpoint_file = None

        if prior_transform is not None:
            self.prior_transform = prior_transform
        else:
            bounds = np.asarray(gp.bounds)
            lo = bounds[:, 0]
            hi = bounds[:, 1]

            def _default_prior_transform(u):
                return lo + u * (hi - lo)

            self.prior_transform = _default_prior_transform

    def _log_likelihood(self, theta):
        """Evaluate the GP log-likelihood (numpy-compatible wrapper)."""
        return float(self.gp.log_likelihood_fn(jnp.asarray(theta)))

    def run_map(self, nopt=10, keys=None, checkpoint_file=None,
                theta0=None, **kwargs):
        """
        Find MAP solutions via parallel multi-start optimization.

        Same interface as ``BlackJAXSampler.run_map``.

        Parameters
        ----------
        nopt : int
            Number of independent optimization restarts (default 10).
        keys : list of str, optional
            Parameter names to optimize.
        theta0 : dict, optional
            Initial parameter guess.
        checkpoint_file : str, optional
            Path to save/load MAP solutions.
        **kwargs
            Additional keyword arguments passed to
            ``GPSolver.fit_map_parallel``.

        Returns
        -------
        all_theta_maps : list of dict
            All MAP solutions sorted by objective (best first).
        """
        if checkpoint_file is not None:
            self._checkpoint_file = checkpoint_file
        path = self._checkpoint_file

        if path is not None and os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            if "all_theta_maps" in data:
                all_theta_maps = list(data["all_theta_maps"])
                data.close()
                logger.info(f"Loaded {len(all_theta_maps)} MAP solutions from {path}")
                self.all_theta_maps = all_theta_maps
                self.theta_map = all_theta_maps[0]
                return all_theta_maps
            data.close()

        gp = self.gp
        if keys is None:
            keys = list(gp.param_keys)

        logger.info(f"Finding MAP solution ({nopt} restarts, returning all)...")
        all_theta_maps, all_results = gp.fit_map_parallel(
            nopt=nopt, keys=keys, return_all=True, theta0=theta0, **kwargs)
        self.all_theta_maps = all_theta_maps
        self.theta_map = all_theta_maps[0]

        map_loglikes = []
        for tm in all_theta_maps:
            if isinstance(tm, dict):
                arr = np.array([float(tm[k]) for k in gp.param_keys],
                               dtype=np.float64)
            else:
                arr = np.asarray(tm, dtype=np.float64)
            map_loglikes.append(float(gp.log_likelihood_fn(arr)))
        self.map_loglikes = np.array(map_loglikes)

        logger.info(f"MAP solution: {self.theta_map}")

        if path is not None:
            _path = path if path.endswith(".npz") else path + ".npz"
            save_kwargs = {}
            if os.path.exists(_path):
                existing = np.load(_path, allow_pickle=True)
                for k in existing.files:
                    save_kwargs[k] = existing[k]
                existing.close()
            save_kwargs["theta_map"] = self.theta_map
            save_kwargs["all_theta_maps"] = all_theta_maps
            save_kwargs["map_loglikes"] = self.map_loglikes
            np.savez(path, **save_kwargs)
            logger.debug(f"MAP solutions saved to {path}")
        self._map_completed = True

        return all_theta_maps

    def run_sampling(self, nlive=500, dlogz=0.01, bound="multi",
                     sample="auto", rstate=None, checkpoint_file=None,
                     **kwargs):
        """
        Run dynesty nested sampling.

        Parameters
        ----------
        nlive : int
            Number of live points (default 500).
        dlogz : float
            Stopping criterion on the remaining evidence (default 0.01).
        bound : str
            Bounding method (default "multi").
        sample : str
            Sampling method (default "auto").
        rstate : numpy.random.RandomState, optional
            Random state for reproducibility.
        checkpoint_file : str, optional
            Override the default checkpoint file path.
        **kwargs
            Additional keyword arguments passed to
            ``dynesty.NestedSampler`` and ``sampler.run_nested``.

        Returns
        -------
        samples : np.ndarray
            Equally weighted posterior samples, shape
            ``(n_samples, n_params)``.
        info : dict
            Diagnostics including log-evidence and its uncertainty.
        """
        import dynesty

        if checkpoint_file is not None:
            self._checkpoint_file = checkpoint_file

        n_params = self.n_params

        # Separate constructor kwargs from run_nested kwargs
        constructor_keys = {
            "update_interval", "first_update", "npdim", "bootstrap",
            "enlarge", "vol_dec", "vol_check", "walks", "facc",
            "slices", "fmove", "max_move", "maxiter_init",
            "maxcall_init", "ncdim", "pool", "queue_size",
        }
        constructor_kwargs = {k: v for k, v in kwargs.items()
                              if k in constructor_keys}
        run_kwargs = {k: v for k, v in kwargs.items()
                      if k not in constructor_keys}

        logger.info(f"Dynesty nested sampling: nlive={nlive}, "
              f"dlogz={dlogz}, bound={bound}, sample={sample}")

        sampler = dynesty.NestedSampler(
            self._log_likelihood,
            self.prior_transform,
            n_params,
            nlive=nlive,
            bound=bound,
            sample=sample,
            rstate=rstate,
            **constructor_kwargs,
        )
        sampler.run_nested(dlogz=dlogz, **run_kwargs)

        results = sampler.results

        # Extract equally weighted posterior samples
        from dynesty.utils import resample_equal
        weights = np.exp(results.logwt - results.logz[-1])
        samples = resample_equal(results.samples, weights)

        self.samples = samples
        self._dynesty_results = results
        self._info = {
            "logz": results.logz[-1],
            "logzerr": results.logzerr[-1],
            "nlive": nlive,
            "niter": results.niter,
            "ncall": sum(results.ncall),
            "eff": results.eff,
            "n_samples": len(samples),
            "n_chains": 1,
            "n_warmup": 0,
        }

        logger.info(f"Dynesty complete: {results.niter} iterations, "
              f"{len(samples)} posterior samples")
        logger.info(f"  log(Z) = {results.logz[-1]:.2f} "
              f"+/- {results.logzerr[-1]:.2f}")

        if self._checkpoint_file is not None:
            self.save_checkpoint()

        return samples, self._info

    def run_dynamic_sampling(self, nlive_init=500, nlive_batch=250,
                             dlogz_init=0.01, maxbatch=10,
                             wt_kwargs=None, rstate=None,
                             checkpoint_file=None, **kwargs):
        """
        Run dynesty dynamic nested sampling.

        Dynamic nested sampling allocates live points adaptively to
        improve both evidence and posterior estimates.

        Parameters
        ----------
        nlive_init : int
            Initial number of live points (default 500).
        nlive_batch : int
            Number of live points per batch (default 250).
        dlogz_init : float
            Stopping criterion for the initial baseline run
            (default 0.01).
        maxbatch : int
            Maximum number of dynamic batches (default 10).
        wt_kwargs : dict, optional
            Keyword arguments for the importance weight function.
            Default: ``{"pfrac": 0.8}`` (80% posterior, 20% evidence).
        rstate : numpy.random.RandomState, optional
            Random state for reproducibility.
        checkpoint_file : str, optional
            Override the default checkpoint file path.
        **kwargs
            Additional keyword arguments passed to
            ``dynesty.DynamicNestedSampler``.

        Returns
        -------
        samples : np.ndarray
            Equally weighted posterior samples, shape
            ``(n_samples, n_params)``.
        info : dict
            Diagnostics including log-evidence and its uncertainty.
        """
        import dynesty

        if checkpoint_file is not None:
            self._checkpoint_file = checkpoint_file
        if wt_kwargs is None:
            wt_kwargs = {"pfrac": 0.8}

        n_params = self.n_params

        logger.info(f"Dynesty dynamic nested sampling: "
              f"nlive_init={nlive_init}, nlive_batch={nlive_batch}, "
              f"maxbatch={maxbatch}")

        sampler = dynesty.DynamicNestedSampler(
            self._log_likelihood,
            self.prior_transform,
            n_params,
            rstate=rstate,
            **kwargs,
        )
        sampler.run_nested(
            nlive_init=nlive_init,
            nlive_batch=nlive_batch,
            dlogz_init=dlogz_init,
            maxbatch=maxbatch,
            wt_kwargs=wt_kwargs,
        )

        results = sampler.results

        from dynesty.utils import resample_equal
        weights = np.exp(results.logwt - results.logz[-1])
        samples = resample_equal(results.samples, weights)

        self.samples = samples
        self._dynesty_results = results
        self._info = {
            "logz": results.logz[-1],
            "logzerr": results.logzerr[-1],
            "nlive_init": nlive_init,
            "nlive_batch": nlive_batch,
            "niter": results.niter,
            "ncall": sum(results.ncall),
            "eff": results.eff,
            "n_samples": len(samples),
            "n_chains": 1,
            "n_warmup": 0,
        }

        logger.info(f"Dynamic nested sampling complete: {results.niter} "
              f"iterations, {len(samples)} posterior samples")
        logger.info(f"  log(Z) = {results.logz[-1]:.2f} "
              f"+/- {results.logzerr[-1]:.2f}")

        if self._checkpoint_file is not None:
            self.save_checkpoint()

        return samples, self._info

    def save_checkpoint(self, path=None):
        """
        Save sampler results to disk.

        Parameters
        ----------
        path : str, optional
            File path (saved as ``.npz``).  If None, uses the
            default checkpoint file.
        """
        if path is None:
            path = self._checkpoint_file
        if path is None:
            raise ValueError(
                "No path provided and no checkpoint_file set.")

        save_kwargs = {
            "samples": np.asarray(self.samples) if self.samples is not None else np.array([]),
        }

        if self._info is not None:
            save_kwargs["logz"] = self._info.get("logz", np.nan)
            save_kwargs["logzerr"] = self._info.get("logzerr", np.nan)

        # Preserve MAP solutions
        if (hasattr(self, "all_theta_maps")
                and self.all_theta_maps is not None):
            save_kwargs["theta_map"] = self.theta_map
            save_kwargs["all_theta_maps"] = np.array(
                self.all_theta_maps, dtype=object)
            if hasattr(self, "map_loglikes"):
                save_kwargs["map_loglikes"] = self.map_loglikes

        np.savez(path, **save_kwargs)

        n_samples = self.samples.shape[0] if self.samples is not None else 0
        logger.debug(f"Checkpoint saved to {path} ({n_samples} samples)")

    def load_checkpoint(self, checkpoint_file=None):
        """
        Restore sampler results from a checkpoint file.

        Parameters
        ----------
        checkpoint_file : str, optional
            Path to a ``.npz`` checkpoint file.
        """
        if checkpoint_file is not None:
            self._checkpoint_file = checkpoint_file
        path = self._checkpoint_file
        if path is None:
            raise ValueError("No checkpoint_file provided or set.")

        data = np.load(path, allow_pickle=True)

        if "samples" in data and data["samples"].size > 0:
            self.samples = data["samples"].copy()

        self._info = {
            "logz": float(data["logz"]) if "logz" in data else None,
            "logzerr": float(data["logzerr"]) if "logzerr" in data else None,
            "n_samples": self.samples.shape[0] if self.samples is not None else 0,
            "n_chains": 1,
            "n_warmup": 0,
        }

        if "all_theta_maps" in data:
            self.all_theta_maps = list(data["all_theta_maps"])
            self.theta_map = (data["theta_map"].item()
                              if data["theta_map"].ndim == 0
                              else data["theta_map"])
            if "map_loglikes" in data:
                self.map_loglikes = np.asarray(data["map_loglikes"])

        data.close()

        n_samples = self.samples.shape[0] if self.samples is not None else 0
        logz = self._info["logz"]
        logz_str = f", log(Z)={logz:.2f}" if logz is not None else ""
        logger.info(f"Checkpoint loaded from {path} "
              f"({n_samples} samples{logz_str})")

    @staticmethod
    def load_samples(path, flatten_chains=True):
        """
        Read samples from a checkpoint file.

        Parameters
        ----------
        path : str
            Path to a ``.npz`` checkpoint file.
        flatten_chains : bool
            Ignored (included for API compatibility with
            BlackJAXSampler).

        Returns
        -------
        samples : np.ndarray
            Shape ``(n_samples, n_params)``.
        """
        data = np.load(path)
        samples = data["samples"].copy()
        data.close()
        return samples

