"""
results.py — Restart-safe result objects for per-star spotgp fits.

Provides a lightweight container (`MAPResult`) for MAP/ACF fit output,
plus helpers to save/load it as an ``.npz`` checkpoint and to mark/check
completion via a ``_SUCCESS`` sentinel file. This lets a batch driver
looping over many stars (e.g. a KIC catalog) safely resume after a
crash or interruption without redoing already-finished stars.

Typical usage in a batch loop::

    from spotgp.results import MAPResult, is_complete, mark_complete

    for kic_id in catalog["KIC"]:
        star_dir = os.path.join(results_dir, f"KIC_{kic_id}")
        if is_complete(star_dir):
            continue  # already fit in a previous run

        gp = GPSolver(ts, model, bounds, mean=0).build_jax()
        theta_map, opt_result = gp.fit_map(nopt=1, method="L-BFGS-B")

        result = MAPResult.from_opt_result(
            star_id=f"KIC_{kic_id}", theta_map=theta_map,
            opt_result=opt_result, prot=prot, bounds=bounds,
        )
        result.save(os.path.join(star_dir, "map_result.npz"))
        mark_complete(star_dir)
"""

import logging
import os
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("spotgp")

__all__ = ["MAPResult", "is_complete", "mark_complete", "SUCCESS_FILENAME"]

SUCCESS_FILENAME = "_SUCCESS"


@dataclass
class MAPResult:
    """
    Container for a single star's MAP (or ACF) fit result.

    Parameters
    ----------
    star_id : str
        Identifier for the star (e.g. "KIC_12735580").
    theta_map : dict
        Best-fit hyperparameters, keyed by name.
    success : bool
        Optimizer convergence flag.
    fun : float
        Final objective value (negative log-posterior) at the optimum.
    nit : int
        Number of optimizer iterations.
    message : str
        Optimizer termination message.
    meta : dict
        Free-form extra info to persist alongside the fit (e.g.
        ``prot``, ``bounds``, catalog values used to build the model).
    """

    star_id: str
    theta_map: dict
    success: bool = False
    fun: float = float("nan")
    nit: int = 0
    message: str = ""
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_opt_result(cls, star_id, theta_map, opt_result, **meta):
        """Build a MAPResult from a ``GPSolver.fit_map(...)`` return pair."""
        return cls(
            star_id=star_id,
            theta_map=dict(theta_map),
            success=bool(getattr(opt_result, "success", False)),
            fun=float(getattr(opt_result, "fun", np.nan)),
            nit=int(getattr(opt_result, "nit", 0)),
            message=str(getattr(opt_result, "message", "")),
            meta=meta,
        )

    def save(self, path):
        """Save this result to ``path`` (an ``.npz`` file)."""
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        np.savez(
            path,
            star_id=self.star_id,
            theta_map=self.theta_map,
            success=self.success,
            fun=self.fun,
            nit=self.nit,
            message=self.message,
            meta=self.meta,
        )
        logger.debug(f"MAPResult for {self.star_id} saved to {path}")

    @classmethod
    def load(cls, path):
        """Load a MAPResult previously written with :meth:`save`."""
        data = np.load(path, allow_pickle=True)
        result = cls(
            star_id=str(data["star_id"]),
            theta_map=data["theta_map"].item(),
            success=bool(data["success"]),
            fun=float(data["fun"]),
            nit=int(data["nit"]),
            message=str(data["message"]),
            meta=data["meta"].item() if "meta" in data else {},
        )
        data.close()
        return result


def is_complete(star_dir):
    """Check whether ``star_dir`` has a ``_SUCCESS`` marker (already fit)."""
    return os.path.exists(os.path.join(star_dir, SUCCESS_FILENAME))


def mark_complete(star_dir):
    """Write the ``_SUCCESS`` marker into ``star_dir``, creating it if needed."""
    os.makedirs(star_dir, exist_ok=True)
    open(os.path.join(star_dir, SUCCESS_FILENAME), "a").close()
