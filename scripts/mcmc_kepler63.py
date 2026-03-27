import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cuda")
_ = jax.devices("cuda")

# Force XLA CUDA timer calibration before any real computation.
# Without this the first real kernel launch triggers the calibration and
# produces a "Delay kernel timed out" warning from cuda_timer.cc.
jax.block_until_ready(jax.jit(lambda x: x + 1)(jnp.zeros(1, dtype=jnp.float64)))

import os
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
import time

from spotgp import (
    TimeSeriesData,
    TrapezoidSymmetricEnvelope,
    VisibilityFunction,
    SpotEvolutionModel,
    GPSolver,
    BlackJAXSampler,
)

t0 = time.time()

# ===================================================================
# Load Kepler-63 lightcurve
# ===================================================================

year = 1
results_dir = f"results_kepler63/year_{year}"
os.makedirs(results_dir, exist_ok=True)

data = np.load("../scripts/results_kepler63/lc_data_masked.npz", allow_pickle=True)
sel_yr = (data["time"] > 365*(year-1)) & (data["time"] <= 365*year)

ts_kepler = TimeSeriesData(x=data["time"][sel_yr], y=data["flux"][sel_yr], yerr=data["flux_err"][sel_yr], normalize=True)
ts_kepler.downsample(dt=0.5)

print(f"Data: N={ts_kepler.N}, baseline={ts_kepler.baseline:.1f} days, "f"cadence={ts_kepler.median_dt:.3f} days")

# ===================================================================
# Set up spot model and GP solver
# ===================================================================

# Kepler-63: P_rot ~ 5.4 days, inc ~ 138 deg (Sanchis-Ojeda+2013)
envelope = TrapezoidSymmetricEnvelope(lspot=10.0, tau_spot=3.0)
visibility = VisibilityFunction(peq=5.4, kappa=0.1, inc=np.deg2rad(138))
model = SpotEvolutionModel(
    envelope=envelope, visibility=visibility, sigma_k=0.005,
)

bounds = {
    "peq":      (4.0, 8.0),
    "kappa":    (-1.0, 1.0),
    "inc":      (np.deg2rad(100), np.deg2rad(170)),
    "lspot":    (1.0, 30.0),
    "tau_spot": (0.5, 20.0),
    "sigma_k":  (1e-5, 0.05),
}

gp = GPSolver(
    ts_kepler, model,
    bounds=bounds,
    matrix_solver="cholesky_banded",
)
gp.build_jax()

# # ===================================================================
# # Find MAP estimate
# # ===================================================================

print("\nFinding MAP solution (10 restarts)...")
theta_map, result = gp.fit_map_parallel(nopt=10, keys=list(bounds.keys()))
np.savez(os.path.join(results_dir, "map_solution.npz"), theta_map=theta_map)
print(f"MAP solution: {theta_map}")

# Plot MAP fit
fig, axes = plt.subplots(3, 1, figsize=(12, 12))
gp.plot_prediction(theta=theta_map, ax=axes[0])
gp.plot_acf(theta=theta_map, ax=axes[1])
gp.plot_psd(theta=theta_map, ax=axes[2])
fig.tight_layout()
fig.savefig(os.path.join(results_dir, "map_fit.png"), dpi=150)
plt.close(fig)

# ===================================================================
# Run MCMC with BlackJAX NUTS
# ===================================================================

sampler = BlackJAXSampler(gp, save_dir=results_dir)

resume = False
n_warmup = 1000
batch_size = 200
n_batches = 5
n_chains = 4
checkpoint_file = os.path.join(results_dir, "mcmc_checkpoint.npz")

if not resume:
    samples, info = sampler.run_nuts(
        n_samples=batch_size,
        n_warmup=1000,
        n_chains=n_chains,
        theta_init=theta_map,
        mass_matrix_method="hessian_map",
        progress_bar=False,
        checkpoint_file=checkpoint_file,
    )
    sampler.save_checkpoint(plot_corner=True)

    for _ in range(n_batches - 1):
        samples, info = sampler.resume_nuts(n_samples=batch_size, n_chains=n_chains)
        sampler.save_checkpoint()
else:
    sampler.load_checkpoint(checkpoint_file)
    print(f"Resuming MCMC from checkpoint: {checkpoint_file}")

    for _ in range(n_batches):
        samples, info = sampler.resume_nuts(n_samples=batch_size, n_chains=n_chains)
        sampler.save_checkpoint()

print(f"\nMCMC completed in {time.time() - t0:.1f} seconds")

# ===================================================================
# Results
# ===================================================================

# theta_map = {'peq': 5.5295255446872815, 'kappa': -0.08035928636533085, 'inc': 2.109954185612277, 'lspot': 19.8682610869352, 'tau_spot': 6.459234871565357, 'sigma_k': 0.0027589217455601714}

all_samples = BlackJAXSampler.load_samples(checkpoint_file)
print(f"\nTotal samples: {all_samples.shape[0]}")

np.savez(
    os.path.join(results_dir, "mcmc_results.npz"),
    samples=all_samples,
    theta_map=theta_map,
    param_keys=list(bounds.keys()),
)

# Corner plot
import corner
fig = corner.corner(all_samples, labels=list(bounds.keys()), show_titles=True)
fig.savefig(os.path.join(results_dir, "corner_plot.png"), dpi=150)
plt.close(fig)

# Covariance comparison
fig, axes = sampler.plot_covariance(
    method="hessian_map", samples=all_samples,
    savefig=os.path.join(results_dir, "covariance_plot.png"),
)
plt.close(fig)

print(f"\nResults saved to {results_dir}/")
