```python
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
```


## Generate synthetic lightcurve data


```python
tsim = 200
tsamp = 0.5
nspot_per_day = 0.25
nspot = int(tsim * nspot_per_day)
```


True parameters — use `nspot_rate` (spots/day) for the kernel amplitude.
`nspot` is the total count used only for the forward simulation.


```python
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
```


## Estimate ACF solution

Bounds use physical `sigma_k` (not `log_sigma_k`) because `fit_acf` optimizes
least-squares on the ACF and requires the physical amplitude scale.


```python
bounds = {
    "peq":      (1.0, 5.0),
    "kappa":    (-1.0, 1.0),
    "inc":      (0.0, np.pi/2),
    "lspot":    (0.1, 20.0),
    "tau":      (0.1, 20.0),
    "sigma_k":  (1e-5, 0.1),   # physical (not log) — required by fit_acf
}

gp = GPSolver(tobs, flux, flux_err, theta_full, bounds=bounds,
              matrix_solver="cholesky_banded")
theta_true = gp.get_theta()
print(f"True parameters: {theta_true}\n")

acf_keys = list(bounds.keys())
theta_acfs = []
vals_acfs = []
print("\nFinding ACF fit solution...")
for _ in tqdm.tqdm(range(10)):
    theta_opt, result = gp.fit_acf(keys=acf_keys, method="nelder-mead")
    theta_acfs.append(theta_opt)
    vals_acfs.append(result.fun)
theta_acf = theta_acfs[np.argmin(vals_acfs)]
print(f"ACF fit solution: {theta_acf}\n")
np.savez(os.path.join(results_dir, "acf_fit_results.npz"),
         theta_acf=theta_acf, theta_true=theta_true)
```


## Plot ACF solution: lightcurve fit, ACF, PSD


```python
acf_res = np.load(os.path.join(results_dir, "acf_fit_results.npz"), allow_pickle=True)
theta_acf  = acf_res["theta_acf"].item()
theta_true = acf_res["theta_true"].item()

tlag_plot = np.arange(0, 30, tsamp)

fig, axes = plt.subplots(3, 1, figsize=(12, 12))
gp.plot_prediction(theta=theta_acf, ax=axes[0], model_color="b",
                   data_color=None, model_label="ACF fit", data_label=None)
gp.plot_acf(theta=theta_acf, ax=axes[1], tlags=tlag_plot,
            model_color="b", model_label="ACF fit")
gp.plot_psd(theta=theta_acf, ax=axes[2], model_color="b", model_label="ACF fit")

t_envelope = theta_acf["lspot"] + 2 * theta_acf["tau"]
for ii in range(int(t_envelope / theta_acf["peq"])):
    axes[1].axvline(ii * theta_acf["peq"], color="b", alpha=0.5, ls="--")
axes[1].axvline(t_envelope, color="b", alpha=0.8, linewidth=2, label="Envelope")

fig.tight_layout()
fig.savefig(os.path.join(results_dir, "acf_fit.png"), dpi=150)
plt.close(fig)
```


## Run MCMC with BlackJAX

Use the ACF fit as the starting point for NUTS sampling. Samples are written to
disk in batches so memory usage stays constant regardless of total sample count.


```python
sampler = BlackJAXSampler(gp)

n_batches = 10
batch_size = 200
checkpoint_file = os.path.join(results_dir, "mcmc_checkpoint.npz")

# Initial run: warmup + first batch
samples, info = sampler.run_nuts(
      n_samples=batch_size,
      n_warmup=500,
      theta_init=theta_acf,
      mass_matrix_method="hessian_map",
      progress_bar=False,
      checkpoint_file=checkpoint_file,
  )
sampler.save_checkpoint()  # appends samples to disk, clears memory

# Resume in batches — constant memory usage regardless of total samples
for _ in range(n_batches - 1):
    samples, info = sampler.resume_nuts(n_samples=batch_size)
    sampler.save_checkpoint()  # appends & clears
```


## Save and visualize results


```python
# Load all samples from disk (only when needed for analysis)
all_samples = BlackJAXSampler.load_samples(checkpoint_file)
print(f"Total samples: {all_samples.shape[0]}")

np.savez(os.path.join(results_dir, "blackjax_mcmc_results.npz"),
         samples=all_samples, info=info,
         theta_acf=theta_acf, theta_true=theta_true, theta_full=theta_full,
         tobs=tobs, flux=flux, flux_err=flux_err)

fig = corner.corner(all_samples, labels=list(bounds.keys()),
                    truths=[theta_true[k] for k in bounds.keys()])
fig.savefig(f"{results_dir}/blackjax_corner_plot.png")

fig, axes = sampler.plot_covariance(method="hessian_map", true_params=theta_true)
fig.savefig(f"{results_dir}/blackjax_covariance_plot.png")
```
