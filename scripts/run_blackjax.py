import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cuda")
_ = jax.devices("cuda")

import sys
import corner
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src_jax"))

from starspot import LightcurveModel
from mcmc import BlackJAXSampler
from gp_solver import GPSolver

np.random.seed(64)

# last_run = [int(x.split("trial")[-1]) for x in os.listdir("results")]
# results_dir = "results/trial" + str(max(last_run)+1)
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir)
results_dir = os.path.join(SCRIPT_DIR, "results", "trial20")

# ===================================================================
# Generate synthetic lightcurve data
# ===================================================================

tsim = 200
tsamp = 0.5
nspot_per_day = 0.25
nspot = int(tsim * nspot_per_day)

# True parameters — use nspot_rate (spots/day) for correct kernel amplitude
# nspot is passed separately to LightcurveModel (total count for the simulation)
theta_full = dict(peq=3.0, kappa=0.3, inc=np.pi/3, nspot_rate=nspot_per_day,
                  lspot=12.0, tau=6.0, alpha_max=0.05, fspot=0.)

lc = LightcurveModel(peq=theta_full['peq'], kappa=theta_full['kappa'],
                     inc=theta_full['inc'], nspot=nspot,
                     lspot=theta_full['lspot'], tau=theta_full['tau'],
                     alpha_max=theta_full['alpha_max'], fspot=theta_full['fspot'],
                     tsim=tsim, tsamp=tsamp, lat=[-np.pi/2, np.pi/2], long=[0, 2*np.pi])
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
# Define prior
# ===================================================================

bounds = {
    "peq":      (1.0, 5.0),
    "kappa":    (-1.0, 1.0),
    "inc":      (0.0, np.pi/2),
    "lspot":    (0.1, 20.0),
    "tau":      (0.1, 20.0),
    "sigma_k":  (1e-5, 0.1),
}

_bounds_arr  = jnp.array(list(bounds.values()), dtype=jnp.float64)
_sk_idx      = list(bounds.keys()).index("sigma_k")
_sk_lo, _sk_hi = _bounds_arr[_sk_idx, 0], _bounds_arr[_sk_idx, 1]

def log_prior(theta_arr):
    lo, hi = _bounds_arr[:, 0], _bounds_arr[:, 1]
    k = 500
    # Differentiable soft boundary barriers (same as default prior)
    barriers = (jnp.sum(jax.nn.log_sigmoid(k * (theta_arr - lo)))
                + jnp.sum(jax.nn.log_sigmoid(k * (hi - theta_arr))))
    # Uniform log-density for all params: -log(hi - lo)
    log_p = -jnp.sum(jnp.log(hi - lo))
    # Replace sigma_k uniform term with log-uniform: p(x) ∝ 1/x
    log_p = log_p + jnp.log(_sk_hi - _sk_lo)                   # remove uniform sigma_k
    log_p = log_p - jnp.log(theta_arr[_sk_idx]) \
                  - jnp.log(jnp.log(_sk_hi / _sk_lo))           # add log-uniform sigma_k
    return barriers + log_p

# ===================================================================
# Estimate MAP solution
# ============================================

gp = GPSolver(tobs, flux, flux_err, theta_full, bounds=bounds,
              log_prior=log_prior, matrix_solver="cholesky_banded",
              save_dir=results_dir)
theta_true = gp.get_theta()
print(f"True parameters: {theta_true}\n")

print("\nFinding MAP solution...")
theta_map, _ = gp.fit_map(nopt=10, method="nelder-mead")
print(f"MAP solution: {theta_map}\n")

print("\nFinding ACF fit solution...")
theta_acf, _ = gp.fit_acf(nopt=10, method="nelder-mead")
print(f"ACF fit solution: {theta_acf}\n")


# ===================================================================
# Plot MAP solution: lightcurve fit, ACF, PSD
# ===================================================================

tlag_plot = np.arange(0, 30, tsamp)

fig, axes = plt.subplots(3, 1, figsize=(12, 12))
gp.plot_prediction(theta=theta_map, ax=axes[0], model_color="r", model_label="MAP fit")
gp.plot_acf(theta=theta_map, ax=axes[1], tlags=tlag_plot, model_color="r", model_label="MAP fit")
gp.plot_psd(theta=theta_map, ax=axes[2], model_color="r", model_label="MAP fit")

gp.plot_prediction(theta=theta_acf, ax=axes[0], model_color="b", data_color=None, model_label="ACF fit", data_label=None)
gp.plot_acf(theta=theta_acf, ax=axes[1], tlags=tlag_plot, model_color="b", model_label="ACF fit")
gp.plot_psd(theta=theta_acf, ax=axes[2], model_color="b", model_label="ACF fit")

t_envelope = theta_acf["lspot"] + 2 * theta_acf["tau"]
for ii in range(int(t_envelope / theta_acf["peq"])):
    axes[1].axvline(ii * theta_acf["peq"], color="b", alpha=0.5, ls="--")
axes[1].axvline(t_envelope, color="b", alpha=0.8, linewidth=2, label="Envelope")
axes[1].text(t_envelope, 1.0, r"$l_{\rm spot} + 2\tau_{\rm spot}$",
             color="b", rotation=90, va="top", ha="right", fontsize=12,
             transform=axes[1].get_xaxis_transform())

fig.tight_layout()
fig.savefig(os.path.join(results_dir, "kernel_fit.png"), dpi=150)
plt.close(fig)

# ===================================================================
# Run MCMC with BlackJAX
# ===================================================================

sampler = BlackJAXSampler(gp, save_dir=results_dir)

n_batches = 10
batch_size = 200
checkpoint_file = os.path.join(results_dir, "mcmc_checkpoint.npz")

samples, info = sampler.run_nuts(
      n_samples=batch_size,
      n_warmup=500,
      n_chains=2,
      theta_init=theta_map,
      mass_matrix_method="hessian_map",
      progress_bar=False,
      checkpoint_file=checkpoint_file,
  )
sampler.save_checkpoint(plot_corner=True)  # appends samples to disk, clears memory

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
                    truths=[theta_true[k] for k in bounds.keys()])
fig.savefig(f"{results_dir}/blackjax_corner_plot.png")

fig, axes = sampler.plot_covariance(method="hessian_map", true_params=theta_true)
fig.savefig(f"{results_dir}/blackjax_covariance_plot.png")