import os
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_PLATFORMS"] = "gpu"

import sys
import numpy as np

sys.path.append("../src")
from starspot import LightcurveModel
from mcmc import BlackJAXSampler
from gp_solver import GPSolver
import corner

np.random.seed(42)

# ===================================================================
# Generate synthetic lightcurve data
# ===================================================================

# True parameters
theta_full = dict(peq=5.0, kappa=0.3, inc=np.pi/3, nspot=40,
                  lspot=10.0, tau=5.0, alpha_max=0.05, fspot=0.)

lc = LightcurveModel(**theta_full, tsim=100, tsamp=0.5, lat=[-np.pi/2, np.pi/2], long=[0, 2*np.pi])
tobs = lc.t
flux = lc.flux
flux_err = np.abs(np.random.normal(0, 0.2*np.std(lc.flux), lc.flux.shape))
print(f"Generated synthetic lightcurve with {len(tobs)} observations.")

gp_true = GPSolver(tobs, flux, flux_err, theta_full)
theta_true = gp_true.get_theta()

# ===================================================================
# Estimate MAP solution
# ===================================================================

# Initial guess: {peq, kappa, inc, nspot, lspot, tau, alpha_max, sigma_k}
theta0 = {
    "peq":       5.0,
    "kappa":     0.0,
    "inc":       np.pi/4,
    "lspot":     8.0,
    "tau":       3.0,
    "nspot":     20.0,
    "alpha_max": 0.1,
    "fspot":     0.0,
}

# Optional: custom bounds (otherwise uses DEFAULT_BOUNDS)
bounds = {
    "peq":       (3.0, 7.0),
    "kappa":     (-1.0, 1.0),
    "inc":       (0.0, np.pi/2),
    "lspot":     (0.1, 20.0),
    "tau":       (0.1, 20.0),
    "sigma_k":   (1e-3, 1e-1),
}

gp = GPSolver(tobs, flux, flux_err, theta0, bounds=bounds)

theta_opts = []
fit_vals = []
for _ in range(10):
    theta_opt, result = gp.find_map(keys=bounds.keys(), method="nelder-mead")
    theta_opts.append(theta_opt)
    fit_vals.append(result.fun)
theta_map = theta_opts[np.argmin(fit_vals)]    
    
# ===================================================================
# Run MCMC with BlackJAX
# ===================================================================

sampler = BlackJAXSampler(gp)

samples, info = sampler.run_nuts(
      n_samples=6000,
      n_warmup=1000,
      theta_init=theta_map,
      mass_matrix_method="hessian_map",
      progress_bar=True,
  )

# ===================================================================
# Save and visualize results
# ===================================================================

np.savez("blackjax_mcmc_results.npz", samples=np.array(samples), info=info, 
         theta_map=theta_map, theta_true=theta_true, theta_full=theta_full,
         tobs=tobs, flux=flux, flux_err=flux_err)

fig = corner.corner(np.array(samples), labels=list(bounds.keys()), truths=[theta_true[key] for key in bounds.keys()])
fig.savefig("blackjax_corner_plot.png")

fig, axes = sampler.plot_covariance(method="hessian_map", true_params=theta_true)
fig.savefig("blackjax_covariance_plot.png")