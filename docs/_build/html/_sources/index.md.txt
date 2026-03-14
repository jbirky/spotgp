# `spotgp`

Gaussian Process kernels for stellar variability from starspot models.

**spotgp** provides analytic and numerical GP kernels derived from physically
motivated starspot models, along with GP solvers for fitting stellar
lightcurves. Both NumPy and JAX backends are available.

<br>

![Lightcurve animation](tutorials/lightcurve_animation.gif)

## Installation

```bash
pip install spotgp
```

For JAX acceleration:

```bash
pip install spotgp[jax]
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/lightcurve_demo
tutorials/gp_solver_quickstart
tutorials/run_blackjax
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/src
```
