# Home

**spotgp:** Gaussian Process kernels for stellar starspot variability implemented in `JAX`.

<br>

![Lightcurve animation](tutorials/lightcurve_animation.gif)

## Installation

### From PyPI

```bash
pip install spotgp
```

With JAX acceleration (recommended):

```bash
pip install "spotgp[jax]"
```

### From source

```bash
git clone https://github.com/jbirky/spotgp.git
cd spotgp
pip install -e ".[jax]"
```

Alternatively, clone the repo and add it to your Python path:

```bash
git clone https://github.com/jbirky/spotgp.git
echo 'export PYTHONPATH="$PYTHONPATH:/path/to/spotgp"' >> ~/.bashrc
source ~/.bashrc
```

## Development 

This project is build and maintained by [Jess Birky](https://github.com/jbirky)--keep an eye out for the paper (coming soon). If you are interested in contributing, feel free to reach out or make a pull request! 

<br>

## Documentation Contents

```{toctree}
:maxdepth: 2
:caption: Basics

self
tutorials/time_domain
tutorials/fourier_domain
overview
tutorials/lightcurve_demo
tutorials/jax_jit
tutorials/gp_optimization
tutorials/data_preprocessing
```

```{toctree}
:maxdepth: 2
:caption: Advanced 

tutorials/custom_envelope_gaussian
tutorials/custom_visibility_function
tutorials/custom_latitude_distribution
tutorials/mcmc_sampling
tutorials/cramer_rao_bound
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/spotgp
```
