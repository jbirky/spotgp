# Intro

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

## Development 

This project is build and maintained by [Jess Birky](https://github.com/jbirky)--keep an eye out for the paper (coming soon). If you are interested in contributing, feel free to reach out or make a pull request! 

<br>

## Documentation Contents

```{toctree}
:maxdepth: 1
:caption: Basics

self
tutorials/conceptual_intro
overview
tutorials/lightcurve_demo
tutorials/trapezoid_symmetric_tutorial
tutorials/gp_optimization
```

```{toctree}
:maxdepth: 2
:caption: Advanced 

tutorials/custom_envelope_gaussian
tutorials/custom_latitude_distribution
tutorials/mcmc_sampling
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/src
```
