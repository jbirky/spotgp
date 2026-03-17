import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cuda")
_ = jax.devices("cuda")

import os
import sys
import tqdm
import corner
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../src_jax")
from starspot import LightcurveModel
from mcmc import BlackJAXSampler
from gp_solver import GPSolver

np.random.seed(64)

last_run = [int(x.split("trial")[-1]) for x in os.listdir("results")]
results_dir = "results/trial" + str(max(last_run)+1)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# ===================================================================
# Generate synthetic lightcurve data
# ===================================================================

tsim = 400
nspot_per_day = 0.2
nspot = int(tsim * nspot_per_day)

# True parameters
theta_full = dict(peq=5.0, kappa=0.3, inc=np.pi/3, nspot=nspot,
                  lspot=10.0, tau=5.0, alpha_max=0.05, fspot=0.)

lc = LightcurveModel(**theta_full, tsim=tsim, tsamp=0.5, lat=[-np.pi/2, np.pi/2], long=[0, 2*np.pi])
tobs = lc.t
flux = lc.flux
flux_err = np.abs(np.random.normal(0, 0.2*np.std(lc.flux), lc.flux.shape))

print(f"Generated synthetic lightcurve with {len(tobs)} observations.")
plt.figure(figsize=[12,5])
plt.errorbar(tobs, flux*100 - 100, yerr=flux_err*100, fmt=".k", capsize=0)
plt.plot(tobs, lc.flux*100 - 100, "r-")
plt.xlabel("Time [days]", fontsize=22)
plt.ylabel(r"$\Delta$Flux [\%]", fontsize=22)
plt.savefig(os.path.join(results_dir, "synthetic_lightcurve.png"), dpi=300)
plt.close()

# ===================================================================
# Estimate MAP solution
# ===================================================================

bounds = {
    "peq":          (3.0, 7.0),
    "kappa":        (-1.0, 1.0),
    "inc":          (0.0, np.pi/2),
    "lspot":        (0.1, 20.0),
    "tau":          (0.1, 20.0),
    "log_sigma_k":  (-4.0, 0.0),
}

gp = GPSolver(tobs, flux, flux_err, theta_full, bounds=bounds,
              matrix_solver="cholesky_banded")
theta_true = gp.get_theta()
print(f"True parameters: {theta_true}\n")

theta_opts = []
fit_vals = []
print("\nFinding MAP solution...")
for _ in tqdm.tqdm(range(10)):
    theta_opt, result = gp.find_map(keys=bounds.keys(), method="nelder-mead")
    theta_opts.append(theta_opt)
    fit_vals.append(result.fun)
theta_map = theta_opts[np.argmin(fit_vals)]
print(f"MAP solution: {theta_map}\n")

# ===================================================================
# Plot MAP solution: lightcurve fit, ACF, PSD
# ===================================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 12))
gp.plot_prediction(theta=theta_map, ax=axes[0])
gp.plot_acf(theta=theta_map, ax=axes[1], tlags=np.arange(0, 50, 0.5))
gp.plot_psd(theta=theta_map, ax=axes[2])
fig.tight_layout()
fig.savefig(os.path.join(results_dir, "map_fit.png"), dpi=150)
plt.close(fig)

# ===================================================================
# Run MCMC with BlackJAX
# ===================================================================

sampler = BlackJAXSampler(gp)

# Initial run: warmup + first batch of samples
n_batches = 10
batch_size = 100
checkpoint_file = os.path.join(results_dir, "mcmc_checkpoint.npz")

samples, info = sampler.run_nuts(
      n_samples=batch_size,
      n_warmup=500,
      theta_init=theta_map,
      mass_matrix_method="hessian_map",
      progress_bar=False,
      checkpoint_file=checkpoint_file,
  )
sampler.save_checkpoint()  # appends samples to disk, clears memory

# Resume in batches — constant memory usage regardless of total samples
for _ in range(n_batches - 1):
    samples, info = sampler.resume_nuts(n_samples=batch_size)
    sampler.save_checkpoint()  # appends & clears

# ===================================================================
# Save and visualize results
# ===================================================================

# Load all samples from disk (only when needed for analysis)
all_samples = BlackJAXSampler.load_samples(checkpoint_file)
print(f"Total samples: {all_samples.shape[0]}")

np.savez(os.path.join(results_dir, "blackjax_mcmc_results.npz"), 
         samples=all_samples, info=info, 
         theta_map=theta_map, theta_true=theta_true, theta_full=theta_full,
         tobs=tobs, flux=flux, flux_err=flux_err)

fig = corner.corner(all_samples, labels=list(bounds.keys()),
                    truths=[theta_true(k) for k in bounds.keys()])
fig.savefig(f"{results_dir}/blackjax_corner_plot.png")

fig, axes = sampler.plot_covariance(method="hessian_map", true_params=theta_true)
fig.savefig(f"{results_dir}/blackjax_covariance_plot.png")