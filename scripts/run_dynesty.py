"""
Sample the GP posterior using dynesty dynamic nested sampling.

Usage:
    python scripts/run_dynesty_mcmc.py                # single core
    python scripts/run_dynesty_mcmc.py --ncores 4     # 4 cores
    python scripts/run_dynesty_mcmc.py --ncores -1    # all available cores
"""
import os
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
import argparse
import numpy as np
import jax.numpy as jnp
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import corner
import dynesty
from dynesty.utils import resample_equal

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from starspot import LightcurveModel
import jax
from gp_solver import GPSolver, _gp_log_likelihood

np.random.seed(42)

# Module-level globals set by _init_worker in each child process
_logl_jit = None
_bounds_lo = None
_bounds_hi = None


def _init_worker(x, y, yerr, mean_val, n_harmonics, n_lat, lat_range,
                 fit_sigma_n, quad_nodes, quad_weights, bounds_lo, bounds_hi):
    """Initialize globals in each worker process."""
    global _logl_jit, _bounds_lo, _bounds_hi
    _bounds_lo = bounds_lo
    _bounds_hi = bounds_hi
    _logl_jit = jax.jit(lambda theta_arr: _gp_log_likelihood(
        theta_arr, x, y, yerr, mean_val,
        n_harmonics, n_lat, lat_range, fit_sigma_n,
        quad_nodes=quad_nodes, quad_weights=quad_weights,
    ))


def log_likelihood(theta):
    """Pure GP log-likelihood for dynesty (prior handled by prior_transform)."""
    theta_arr = jnp.array(theta, dtype=jnp.float64)
    val = float(_logl_jit(theta_arr))
    if not np.isfinite(val):
        return -1e300
    return val


def prior_transform(u):
    """Transform unit cube to parameter space (uniform priors)."""
    return _bounds_lo + u * (_bounds_hi - _bounds_lo)


def main():
    global _logl_jit, _bounds_lo, _bounds_hi

    # =============================================================================
    # Parse arguments
    # =============================================================================

    parser = argparse.ArgumentParser(description="Dynesty dynamic nested sampling for GP posterior")
    parser.add_argument("--ncores", type=int, default=6,
                        help="Number of cores to use. -1 for all available (default: 1)")
    parser.add_argument("--nlive", type=int, default=250,
                        help="Number of initial live points (default: 250)")
    args = parser.parse_args()

    ncores = cpu_count() if args.ncores == -1 else args.ncores


    # =============================================================================
    # 1. Generate synthetic lightcurve
    # =============================================================================

    theta_full = dict(
        peq=5.0, kappa=0.3, inc=np.pi/3, nspot=40,
        lspot=10.0, tau=5.0, alpha_max=0.05, fspot=0.0,
    )

    lc = LightcurveModel(
        **theta_full, tsim=100, tsamp=0.4,
        lat=[-np.pi/2, np.pi/2], long=[0, 2*np.pi],
    )
    tobs = lc.t
    flux = lc.flux
    flux_err = np.abs(np.random.normal(0, 0.2 * np.std(lc.flux), lc.flux.shape))

    gp_true = GPSolver(tobs, flux, flux_err, theta_full)
    theta_true = gp_true.get_theta()

    # =============================================================================
    # 2. Set up GP solver
    # =============================================================================

    theta0 = dict(
        peq=4.0, kappa=0.0, inc=np.pi/4, nspot=20.0,
        lspot=8.0, tau=3.0, alpha_max=0.1, fspot=0.0,
    )

    bounds = {
        "peq":     (3.0, 7.0),
        "kappa":   (-1.0, 1.0),
        "inc":     (0.0, np.pi / 2),
        "lspot":   (0.1, 20.0),
        "tau":     (0.1, 20.0),
        "sigma_k": (1e-3, 1e-1),
    }

    gp = GPSolver(tobs, flux, flux_err, theta0, bounds=bounds)
    param_keys = list(gp.param_keys)
    ndim = gp.n_params

    # Bounds arrays for prior transform
    bounds_lo = np.array([bounds[k][0] for k in param_keys])
    bounds_hi = np.array([bounds[k][1] for k in param_keys])

    # Worker init args (shared across parent + children)
    init_args = (gp.x, gp.y, gp.yerr, gp.mean_val,
                 gp.n_harmonics, gp.n_lat, gp.lat_range, gp.fit_sigma_n,
                 gp._quad_nodes, gp._quad_weights, bounds_lo, bounds_hi)

    # Initialize globals in the parent process too
    _init_worker(*init_args)

    print(f"Sampling {ndim} parameters: {param_keys}")
    print(f"Bounds: {dict(zip(param_keys, zip(bounds_lo, bounds_hi)))}")


    # =============================================================================
    # 3. Find MAP estimate for reference
    # =============================================================================

    # theta_map, result = gp.find_map(keys=bounds.keys(), method="nelder-mead", disp=True)
    # print(f"MAP estimate: {theta_map}")


    # =============================================================================
    # 4. Run dynesty dynamic nested sampling
    # =============================================================================

    pool = Pool(ncores, initializer=_init_worker, initargs=init_args) if ncores > 1 else None
    try:
        sampler = dynesty.DynamicNestedSampler(
            log_likelihood,
            prior_transform,
            ndim,
            nlive=args.nlive,
            pool=pool,
            queue_size=ncores if ncores > 1 else 1,
        )

        print(f"Running dynesty dynamic nested sampling with {ncores} core(s)...")
        sampler.run_nested(wt_kwargs={'pfrac': 1.0},
                           stop_kwargs={'pfrac': 1.0, 'post_thresh': 0.02},
                           print_progress=True)
        
        results = sampler.results
    except Exception as e:
        raise RuntimeError(f"Dynesty sampler failed: {e}") from e
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    print(f"\nlog(Z) = {results.logz[-1]:.2f} +/- {results.logzerr[-1]:.2f}")


    # =============================================================================
    # 5. Extract weighted posterior samples
    # =============================================================================

    weights = np.exp(results.logwt - results.logz[-1])
    samples = resample_equal(results.samples, weights)
    print(f"Extracted {len(samples)} equally-weighted posterior samples")


    # =============================================================================
    # 6. Print summary
    # =============================================================================

    print(f"\n{'param':>12s}  {'mean':>10s}  {'std':>10s}  "
          f"{'16%':>10s}  {'50%':>10s}  {'84%':>10s}")
    print("-" * 68)
    for i, key in enumerate(param_keys):
        col = samples[:, i]
        q16, q50, q84 = np.percentile(col, [16, 50, 84])
        m, s = np.mean(col), np.std(col)
        print(f"{key:>12s}  {m:10.5f}  {s:10.5f}  "
              f"{q16:10.5f}  {q50:10.5f}  {q84:10.5f}")


    # =============================================================================
    # 7. Corner plot
    # =============================================================================

    truths = [theta_true.get(k, None) for k in param_keys]

    fig = corner.corner(samples, labels=param_keys, truths=truths, show_titles=True)
    fig.savefig("dynesty_corner.png", dpi=150, bbox_inches="tight")
    print("\nCorner plot saved to dynesty_corner.png")


    # =============================================================================
    # 8. Save results
    # =============================================================================

    np.savez(
        "dynesty_results.npz",
        samples=samples,
        logz=results.logz[-1],
        logzerr=results.logzerr[-1],
        param_keys=param_keys,
        data=dict(tobs=tobs, flux=flux, flux_err=flux_err)
    )
    print("Results saved to dynesty_results.npz")


if __name__ == "__main__":
    main()
