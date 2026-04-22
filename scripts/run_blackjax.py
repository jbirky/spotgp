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

import sys
import time
import argparse
import corner
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

from src import (
    TrapezoidSymmetricEnvelope,
    VisibilityFunction,
    SpotEvolutionModel,
    LightcurveModel,
    GPSolver,
)
from src.mcmc import BlackJAXSampler

# ===================================================================
# CLI
# ===================================================================

parser = argparse.ArgumentParser(description="Run BlackJAX NUTS sampler for spotgp.")
parser.add_argument("--tsim",   type=float, default=200,  help="Simulation time in days (default: 200)")
parser.add_argument("--tsamp",  type=float, default=0.5,  help="Sampling cadence in days (default: 0.5)")
parser.add_argument("--peq",    type=float, default=3.0,  help="Equatorial rotation period in days (default: 3.0)")
parser.add_argument("--nchain", type=int,   default=3,    help="Number of MCMC chains (default: 3)")
parser.add_argument("--resume", action="store_true",      help="Skip lightcurve generation and MAP fit; resume MCMC from checkpoint")
parser.add_argument("--nbatches", type=int, default=10, help="Number of MCMC batches to run (default: 10)")
parser.add_argument("--batchsize", type=int, default=200, help="Number of MCMC samples per batch (default: 200)")
args = parser.parse_args()

tsim  = args.tsim
tsamp = args.tsamp
peq   = args.peq

tstart = time.time()
np.random.seed(64)

# ===================================================================
# Build the spot evolution model
# ===================================================================

nspot_per_day = 0.25
nspot         = int(tsim * nspot_per_day)

results_dir = os.path.join(SCRIPT_DIR, "results", f"trial_{str(peq).replace('.', 'p')}peq_{tsim}tsim_{str(tsamp).replace('.', 'p')}tsamp_{nspot}nspot")
os.makedirs(results_dir, exist_ok=True)

envelope   = TrapezoidSymmetricEnvelope(lspot=12.0, tau_spot=6.0)
visibility = VisibilityFunction(peq=peq, kappa=0.3, inc=np.pi / 3)
model = SpotEvolutionModel(
    envelope=envelope,
    visibility=visibility,
    nspot_rate=nspot_per_day,
    alpha_max=0.05,
    fspot=0.0,
)

print("param_keys:", model.param_keys)
print(model)

# ===================================================================
# Generate synthetic lightcurve data  (skipped on --resume)
# ===================================================================

data_file = os.path.join(results_dir, "data.npz")

if not args.resume:
    lc = LightcurveModel.from_spot_model(
        spot_model=model,
        nspot=nspot,
        tsim=tsim,
        tsamp=tsamp,
        long=[0, 2 * np.pi],
    )

    tobs     = lc.t
    flux     = lc.flux
    flux_err = np.abs(np.random.normal(0, 0.2 * np.std(lc.flux), lc.flux.shape))

    np.savez(data_file, tobs=tobs, flux=flux, flux_err=flux_err)
    print(f"Generated synthetic lightcurve with {len(tobs)} observations.")

    plt.figure(figsize=[12, 5])
    plt.errorbar(tobs, flux * 100 - 100, yerr=flux_err * 100, fmt=".k", capsize=0)
    plt.plot(tobs, lc.flux * 100 - 100, "r-")
    plt.xlabel("Time [days]", fontsize=22)
    plt.ylabel(r"$\Delta$Flux [\%]", fontsize=22)
    plt.savefig(os.path.join(results_dir, "synthetic_lightcurve.png"), dpi=300)
    plt.close()
else:
    data = np.load(data_file)
    tobs     = data["tobs"]
    flux     = data["flux"]
    flux_err = data["flux_err"]
    print(f"Loaded lightcurve with {len(tobs)} observations from {data_file}.")

# ===================================================================
# Define prior bounds
# ===================================================================

bounds = {
    "peq":         (1.0, 5.0),
    "kappa":       (-1.0, 1.0),
    "inc":         (0.0, np.pi / 2),
    "lspot":       (0.1, 20.0),
    "tau_spot":    (0.1, 20.0),
    "log_sigma_k": (-5.0, -1.0),   # sample sigma_k in log10 space
}

_bounds_arr      = jnp.array(list(bounds.values()), dtype=jnp.float64)
_lsk_idx         = list(bounds.keys()).index("log_sigma_k")
_lsk_lo, _lsk_hi = _bounds_arr[_lsk_idx, 0], _bounds_arr[_lsk_idx, 1]

def log_prior(theta_arr):
    lo, hi = _bounds_arr[:, 0], _bounds_arr[:, 1]
    k = 500
    barriers = (jnp.sum(jax.nn.log_sigmoid(k * (theta_arr - lo)))
                + jnp.sum(jax.nn.log_sigmoid(k * (hi - theta_arr))))
    # Uniform log-density over bounded box
    log_p = -jnp.sum(jnp.log(hi - lo))
    # log_sigma_k is already sampling in log space — uniform over log10 range
    # (i.e. log-uniform over sigma_k): no extra correction needed
    return barriers + log_p

# ===================================================================
# Build GP solver and warm up JAX
# ===================================================================

gp = GPSolver(
    tobs, flux, flux_err, model,
    bounds=bounds,
    log_prior=log_prior,
    save_dir=results_dir,
).build_jax()

theta_true = gp.get_theta()
print(f"\nTrue parameters: {theta_true}")

# ===================================================================
# MAP estimate  (skipped on --resume)
# ===================================================================

map_file = os.path.join(results_dir, "map_fit_results.npz")

if not args.resume:
    print("\nFinding MAP solution...")
    theta_map, _ = gp.fit_map(nopt=10, method="nelder-mead")
    print(f"MAP solution: {theta_map}\n")
else:
    _map_data = np.load(map_file, allow_pickle=True)
    theta_map = _map_data["theta_map"].item()
    print(f"Loaded MAP solution from {map_file}: {theta_map}\n")

# ===================================================================
# Run MCMC with BlackJAX
# ===================================================================

sampler         = BlackJAXSampler(gp, save_dir=results_dir)
n_batches       = args.nbatches
batch_size      = args.batchsize
checkpoint_file = os.path.join(results_dir, "mcmc_checkpoint.npz")

if args.resume:
    sampler.load_checkpoint(checkpoint_file)
    print(f"Resuming MCMC from checkpoint: {checkpoint_file}")
else:
    sampler.run_warmup(
        n_warmup=1000,
        n_chains=args.nchain,
        theta_init=theta_map,
        mass_matrix_method="hessian_map",
        checkpoint_file=checkpoint_file,
    )

for _ in range(n_batches):
    samples, info = sampler.run_sampling(n_samples=batch_size)
    sampler.save_checkpoint(plot_corner=True)

# ===================================================================
# Save and visualize results
# ===================================================================

all_samples = BlackJAXSampler.load_samples(checkpoint_file)
print(f"Total samples: {all_samples.shape[0]}")

np.savez(os.path.join(results_dir, "blackjax_mcmc_results.npz"),
         samples=all_samples, info=info,
         theta_map=theta_map, theta_true=theta_true,
         tobs=tobs, flux=flux, flux_err=flux_err)

fig = corner.corner(all_samples, labels=list(bounds.keys()),
                    truths=[theta_true[k] for k in bounds.keys()])
fig.savefig(os.path.join(results_dir, "blackjax_corner_plot.png"))

print(f"\nTotal runtime: {time.time() - tstart:.2f} seconds ({(time.time() - tstart)/60:.2f} minutes)")
