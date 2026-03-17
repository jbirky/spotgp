# `spotgp`

Gaussian Process kernels for stellar variability from starspot models.

**spotgp** provides analytic and numerical GP kernels derived from physically
motivated starspot models, along with GP solvers for fitting stellar
lightcurves. Both NumPy and JAX backends are available.

<br>

![Lightcurve animation](tutorials/lightcurve_animation.gif)

## Installation

```bash
git clone https://github.com/jbirky/spotgp.git
cd spotgp
pip install -e .
```

For JAX acceleration:

```bash
pip install -e ".[jax]"
```

## Modules

Both `src_numpy` and `src_jax` expose the same interface. The JAX backend adds JIT compilation, `vmap`-vectorized latitude integration, and automatic differentiation (required for NUTS sampling). The main class modules for the package are:

| Module | Description |
|---|---|
| `starspot` | `LightcurveModel` — physical simulator for rotating spotted stars. Generates synthetic flux time series from spot parameters (size, contrast, lifetime, latitude). Includes animation utilities. |
| `analytic_kernel` | `AnalyticKernel` — closed-form GP kernel derived from a trapezoidal spot envelope. Computes the autocorrelation $R_\Gamma(\tau)$ and power spectral density analytically via Fourier coefficients of the spot visibility function averaged over latitude. |
| `numerical_kernel` | `NumericalKernel` — empirical kernel estimated by Monte Carlo averaging of synthetic lightcurve autocorrelations. Useful for validating the analytic kernel equations. |
| `gp_solver` | `GPSolver` — Gaussian Process inference engine. Builds the covariance matrix from a kernel, evaluates the marginal log-likelihood via Cholesky decomposition, and finds the MAP estimate. The JAX version supports autodiff gradients. |
| `mcmc` | `MCMCSampler`  — A class for analyzing and visualizing MCMC results. Sampling functions using specific packages are build in inheritance classes (e.g. `BlackJAXSampler`). |

<br>
<br>

## Documentation Contents

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/lightcurve_demo
tutorials/gp_solver_quickstart
tutorials/custom_envelope
tutorials/custom_latitude_distribution
tutorials/gp_optimization
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/src_numpy
api/src_jax
```
