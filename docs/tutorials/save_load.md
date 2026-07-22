# Saving and Loading Results

spotgp supports saving all fit state — data, model, configuration, MAP/ACF results,
MCMC chains, and sampler restart state — to a single HDF5 file. This lets you
checkpoint long-running fits and resume them later without re-running optimization
or warm-up.

## Saving a GPSolver

After setting up and fitting a `GPSolver`, save the full state with:

```python
import numpy as np
from spotgp import GPSolver, SpotEvolutionModel, TimeSeriesData
from spotgp import TrapezoidSymmetricEnvelope, VisibilityFunction

# Set up data + model + solver
data = TimeSeriesData(t, flux, flux_err)
model = SpotEvolutionModel(
    envelope=TrapezoidSymmetricEnvelope(lspot=15.0, tau_spot=5.0),
    visibility=VisibilityFunction(peq=10.0, kappa=0.3, inc=np.pi / 3),
    sigma_k=0.01,
)
bounds = {"peq": (5, 30), "kappa": (0, 0.5), "log_sigma_k": (-4, -1)}
gp = GPSolver(data, model, bounds=bounds).build_jax()

# Run MAP optimization
theta_map, result = gp.fit_map(nopt=5)

# Save everything to HDF5
gp.save("star.h5")
```

The file stores data arrays, model component classes and parameters, solver
configuration (bounds, kernel settings), and any completed fit results.

### Incremental saves

`save()` opens the file in append mode, so you can call it at each stage of
a fit. Only the groups that changed are overwritten:

```python
gp = GPSolver(data, model, bounds=bounds).build_jax()
gp.save("star.h5")          # saves data + model + config

theta_map, result = gp.fit_map(nopt=5)
gp.save("star.h5")          # adds MAP results

theta_acf = gp.fit_acf()
gp.save("star.h5")          # adds ACF results

gp.mass_matrix_hessian_map()
gp.save("star.h5")          # adds mass matrix
```

---

## Loading a GPSolver

Reconstruct a `GPSolver` from a saved file:

```python
gp = GPSolver.load("star.h5")
```

The loaded solver has all fit results (MAP estimate, ACF fit, mass matrix) restored,
but is **not** JIT-compiled. Call `.build_jax()` before running further fits or
sampling:

```python
gp = GPSolver.load("star.h5")
gp.build_jax()

# Continue fitting or start sampling
theta_map, result = gp.fit_map(nopt=10)
```

---

## Sampler checkpoints

Both `BlackJAXSampler` and `DynestySampler` support HDF5 checkpoints. Use a
`.h5` or `.hdf5` file extension and the sampler will automatically use the HDF5
format instead of `.npz`:

### BlackJAX (NUTS)

```python
from spotgp import BlackJAXSampler

sampler = BlackJAXSampler(gp, save_dir="results", checkpoint_file="star.h5")

# Run MAP + warm-up + sampling (checkpoints automatically)
sampler.run_map(nopt=5)
sampler.run_warmup(n_warmup=500)
sampler.run_sampling(n_samples=2000)

# The .h5 file now contains GPSolver state + sampler state + samples
```

Samples are **appended** to the HDF5 file using resizable datasets, so calling
`run_sampling()` multiple times accumulates samples without loading the full
array into memory.

### Dynesty

```python
from spotgp import DynestySampler

sampler = DynestySampler(gp, save_dir="results", checkpoint_file="star.h5")
sampler.run_map(nopt=5)
samples, info = sampler.run_sampling(nlive=500)
```

---

## Restarting MCMC from a checkpoint

To resume a BlackJAX NUTS run from where it left off:

```python
# Reconstruct the solver
gp = GPSolver.load("star.h5")
gp.build_jax()

# Create a new sampler and load the checkpoint
sampler = BlackJAXSampler(gp)
sampler.load_checkpoint("star.h5")

# The sampler's NUTS state, step size, inverse mass matrix, and
# RNG key are all restored — continue sampling from the last position
sampler.run_sampling(n_samples=2000)
```

---

## Loading samples without a sampler

To read posterior samples from a checkpoint file without reconstructing
the full sampler:

```python
from spotgp import load_samples

samples = load_samples("star.h5")  # shape (n_samples, n_params)
```

Or equivalently via the sampler class:

```python
samples = BlackJAXSampler.load_samples("star.h5")
```

---

## Standalone functions

The `save_gp` / `load_gp` functions and sampler I/O are also available as
standalone functions in `spotgp.io`:

```python
from spotgp import save_gp, load_gp, save_sampler, load_sampler, load_samples

save_gp("star.h5", gp)
gp = load_gp("star.h5")

save_sampler("star.h5", sampler)
load_sampler("star.h5", sampler)
```

---

## HDF5 file structure

The HDF5 file is organized into groups:

```text
star.h5
├── data/                       # observed time series
│   ├── x, y, yerr
│
├── model/                      # spot evolution model
│   ├── attrs: sigma_k, fspot, alpha_max
│   ├── envelope/               # envelope class + parameters
│   ├── visibility/             # visibility class + parameters
│   └── latitude/               # latitude distribution class + parameters
│
├── config/                     # solver configuration
│   ├── attrs: kernel_type, mean_val, fit_sigma_n, ...
│   ├── param_keys              # ordered parameter names
│   ├── bounds                  # parameter bounds (n_params × 2)
│   └── lat_range               # latitude integration range
│
├── fits/                       # optimization results
│   ├── map/                    # MAP estimate + metadata
│   ├── acf/                    # ACF fit + data
│   └── mass_matrix/            # Fisher / Hessian matrices
│
└── sampler/                    # MCMC state
    ├── attrs: sampler_type
    ├── samples                 # posterior samples (resizable)
    ├── map_solutions/          # MAP solutions from sampler
    ├── nuts_state/             # BlackJAX NUTS restart state
    └── dynesty_results/        # Dynesty log-evidence
```

You can inspect the file with `h5py` directly:

```python
import h5py

with h5py.File("star.h5", "r") as f:
    f.visit(lambda name: print(name))
```

---

## Limitations

- **Custom callables**: The `mean` function is stored as its float value;
  custom `log_prior` functions use the default soft-uniform prior on load.
- **Custom model subclasses**: Only built-in envelope, visibility, and latitude
  classes are supported. User-defined subclasses will fail on load.
- **Format selection**: The file extension determines the format — `.h5` or
  `.hdf5` for HDF5, `.npz` for the legacy NumPy format.
