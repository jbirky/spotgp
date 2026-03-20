# Code Overview

`spotgp` models stellar photometric variability caused by rotating starspots as
a Gaussian Process. The GP covariance kernel is derived analytically from a
physical spot evolution model, enabling fast, gradient-based inference of stellar
rotation periods, differential rotation, spot lifetimes, and inclination.

---

## Module architecture

```{mermaid}
flowchart TD
    subgraph foundation ["Foundation"]
        PA[params.py\nhparam validation]
        BC[banded_cholesky.py\nsparse linear algebra]
        PSD[psd.py\nLomb–Scargle PSD]
    end

    subgraph model ["Model layer  —  spot_model.py + envelope.py"]
        EF["EnvelopeFunction\n(abstract base)"]
        TE[TrapezoidSymmetricEnvelope]
        AE[TrapezoidAsymmetricEnvelope]
        SE[SkewedGaussianEnvelope]
        EE[ExponentialEnvelope]
        CE["... custom subclass"]
        TE & AE & SE & EE & CE -->|inherits| EF

        LD["LatitudeDistributionFunction\n(base / subclass)"]
        VF[VisibilityFunction]
        SM[SpotEvolutionModel]

        EF --> SM
        VF --> SM
        LD --> SM
    end

    subgraph kernel ["Kernel layer"]
        AK[AnalyticKernel\nanalytic_kernel.py]
        NK[NumericalKernel\nnumerical_kernel.py]
    end

    subgraph sim ["Simulation"]
        LC[LightcurveModel\nlightcurve.py]
    end

    subgraph inference ["Inference layer"]
        GP[GPSolver\ngp_solver.py]
        MC[MCMCSampler / BlackJAXSampler\nmcmc.py]
    end

    PA -->|resolve_hparam| SM
    SM --> AK
    SM --> LC
    AK --> GP
    LC --> NK
    NK -.->|benchmark| AK
    BC --> GP
    GP --> MC
```

---

## Module descriptions

### Foundation

| Module | Key exports | Role |
|---|---|---|
| `params.py` | `resolve_hparam`, `KERNEL_HPARAM_KEYS` | Validates and normalizes raw hyperparameter dicts. Single source of truth for parameter names, envelope detection, and amplitude modes. |
| `banded_cholesky.py` | `banded_cholesky_compact`, `banded_solve_compact` | O(n·b) memory Cholesky factorization and solve for banded symmetric positive-definite matrices. Used internally by `GPSolver`. |
| `psd.py` | `compute_psd` | Lomb–Scargle PSD of unevenly sampled time series via `astropy`. |

---

### Model layer

The model layer defines the **physics** of spot evolution and stellar geometry.
It is composed of three independent components that are combined into a single
`SpotEvolutionModel`.

#### `envelope.py` — Spot size evolution

`EnvelopeFunction` is an abstract base class. Subclass it and implement
`tau_spot` (property) and `Gamma(t)` to define a new spot shape. Everything
else has working defaults:

| Method | Default | Purpose |
|---|---|---|
| `Gamma(t)` | **required** | Normalized spot-size envelope, peak = 1 |
| `tau_spot` | **required** | Characteristic timescale [days] |
| `R_Gamma(lag)` | FFT interpolation | Autocorrelation ∫ Γ(t) Γ(t+lag) dt |
| `Gamma_hat(omega)` | FFT interpolation | Fourier transform magnitude \|FT[Γ]\|(ω) |
| `lspot` | `0.0` | Plateau duration [days] |
| `param_dict` | `{}` | Free parameters exposed to `GPSolver` |
| `kernel_support()` | `lspot + 6·tau_spot` | Lag beyond which R_Γ ≈ 0 |

Use `check_functions()` to compare an analytic override of `R_Gamma` or
`Gamma_hat` against the FFT baseline.

Built-in envelopes:

| Class | Parameters | Notes |
|---|---|---|
| `TrapezoidSymmetricEnvelope` | `lspot`, `tau_spot` | Analytic `Gamma_hat` and `R_Gamma` |
| `TrapezoidAsymmetricEnvelope` | `lspot`, `tau_em`, `tau_dec` | Analytic `R_Gamma`; rise ≠ decay |
| `SkewedGaussianEnvelope` | `sigma_sn`, `n_sn` | Skew-normal shape |
| `ExponentialEnvelope` | `tau_spot` | Analytic `Gamma_hat`, `R_Gamma` |

#### `spot_model.py` — Rotation geometry and model assembly

**`VisibilityFunction`** encodes how much flux a spot at latitude φ contributes
as the star rotates under differential rotation. The contribution is expanded as
a Fourier series in rotation harmonics with coefficients cₙ(inc, φ).

Parameters: `peq` (equatorial period), `kappa` (differential rotation shear),
`inc` (stellar inclination).

**`LatitudeDistributionFunction`** defines the probability density p(φ) over
stellar latitude. The default is uniform over [−π/2, π/2]. Subclass and
override `__call__` and/or `lat_range` to define a custom distribution.

| Attribute / method | Purpose |
|---|---|
| `lat_range` | `(min, max)` latitude bounds used for spot placement and kernel integration |
| `__call__(phi)` | Unnormalized PDF evaluated at φ; normalization is handled internally |

The PDF weights the latitude integral inside `AnalyticKernel`, and `lat_range`
sets the uniform sampling bounds for spot placement in `LightcurveModel`.

**`SpotEvolutionModel`** assembles the three components:

```python
model = SpotEvolutionModel(
    envelope=TrapezoidSymmetricEnvelope(lspot=15.0, tau_spot=5.0),
    visibility=VisibilityFunction(peq=10.0, kappa=0.3, inc=np.pi / 3),
    sigma_k=0.01,
    latitude_distribution=GaussianLatitude(sigma_deg=20.0),  # optional
)
```

`param_keys` exposes the full ordered parameter vector
`(peq, kappa, inc, <envelope params>, sigma_k)` used by `GPSolver`.

---

### Kernel layer

#### `analytic_kernel.py` — `AnalyticKernel`

Computes the GP covariance kernel by integrating the per-latitude kernel
contributions over the latitude distribution:

$$K(\tau) = \sigma_k^2 \, R_\Gamma(\tau)
  \int p(\phi)\, \left[ c_0^2(\phi) + 2\sum_{n=1}^{N} c_n^2(\phi)
  \cos(n\,\omega_0(\phi)\,\tau) \right] d\phi$$

where Rᵧ(τ) is the autocorrelation of the spot envelope and cₙ are the
visibility Fourier coefficients.

The latitude integral uses `jax.lax.scan` for O(M) memory regardless of the
number of latitude points. Call `build_jax()` once after construction to
pre-compile the XLA kernels.

Key parameters: `n_harmonics` (default 3), `n_lat` (default 64),
`quadrature` (`"trapezoid"` or `"gauss-legendre"`).

#### `numerical_kernel.py` — `NumericalKernel`

Estimates the kernel empirically by simulating many lightcurves with
`LightcurveModel` and averaging their autocovariance. Used for benchmarking
and validating the analytic kernel against Monte Carlo simulations.

---

### Simulation

#### `lightcurve.py` — `LightcurveModel`

Simulates a stellar lightcurve by summing the flux deficit of `nspot`
independently evolving spots. Each spot is placed at a random longitude,
latitude (drawn uniformly within `lat_range`), and emergence time; its
angular size follows the envelope `Gamma(t)`.

Spot positions rotate at the latitude-dependent rate ω₀(φ) = 2π(1 − κ sin²φ)/Peq.

```python
lc = LightcurveModel.from_spot_model(model, nspot=30, tsim=150, tsamp=0.5)
```

Includes `plot_lightcurve()` and `animate_lightcurve()` for visualization.

---

### Inference layer

#### `gp_solver.py` — `GPSolver`

Builds the GP covariance matrix from `AnalyticKernel`, factorises it via
Cholesky (full or banded), and evaluates the marginal log-posterior. Four
JIT-compiled functions are exposed: `log_posterior`, `neg_log_posterior`,
`grad_log_posterior`, `grad_neg_log_posterior`.

Call `build_jax()` once before fitting to pre-compile all four. The banded
Cholesky solver (default) uses the kernel support to determine bandwidth and
achieves O(n·b) memory and O(n·b²) time.

```python
gp = GPSolver(t, flux, flux_err, model, bounds=bounds).build_jax()
theta_map, result = gp.fit_map(nopt=5)
```

Key methods:

| Method | Purpose |
|---|---|
| `fit_map(nopt=N)` | L-BFGS-B MAP optimization, N random restarts |
| `predict()` | GP posterior mean and variance at new times |
| `plot_prediction()` | Posterior mean ± σ bands over the data |
| `plot_acf()` | Empirical ACF vs analytic kernel |
| `plot_psd()` | Lomb–Scargle PSD vs analytic PSD |
| `plot_covariance_matrix()` | Banded covariance matrix with sparsity annotation |
| `mass_matrix_hessian_map()` | Hessian-based posterior covariance at MAP |

#### `mcmc.py` — `MCMCSampler` / `BlackJAXSampler`

Wraps `GPSolver` with MCMC sampling. `BlackJAXSampler` uses the BlackJAX NUTS
sampler with gradient information from `grad_log_posterior`. Provides posterior
summaries, corner plots, convergence diagnostics, and checkpointing.

---

## Data flow: fitting a lightcurve

```text
Observed flux (t, y, yerr)
         │
         ▼
   SpotEvolutionModel           ← EnvelopeFunction
   (peq, kappa, inc,            ← VisibilityFunction
    lspot, tau_spot, sigma_k)   ← LatitudeDistributionFunction
         │
         ▼
   AnalyticKernel.kernel(lag)
         │
         ▼
   GPSolver.build_jax()
         │
         ├─ fit_map()           → theta_MAP (point estimate)
         │
         └─ BlackJAXSampler     → posterior samples
```

---

## Extending the library

Three extension points allow custom physics without modifying core code:

| Extension point | Base class | Minimum required |
|---|---|---|
| Custom spot shape | `EnvelopeFunction` | `tau_spot`, `Gamma(t)` |
| Custom latitude distribution | `LatitudeDistributionFunction` | `__call__(phi)` |
| Custom amplitude parameterization | pass `sigma_k` directly | — |

See the tutorials for worked examples:

- [Custom Gaussian envelope](tutorials/custom_envelope_gaussian.ipynb)
- [Custom latitude distribution](tutorials/custom_latitude_distribution.ipynb)
