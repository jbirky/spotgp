# Home

**spotgp:** Gaussian Process kernels for stellar starspot variability implemented in `JAX`.

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
:maxdepth: 2
:caption: Basics

self

:maxdepth: 2
  :caption: Model Concepts

tutorials/time_domain
tutorials/fourier_domain
```

```{toctree}
:maxdepth: 2

overview
tutorials/lightcurve_demo
tutorials/jax_jit
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
