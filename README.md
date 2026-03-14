# `spotgp`

[![Documentation Status](https://readthedocs.org/projects/spotgp/badge/?version=latest)](https://spotgp.readthedocs.io/en/latest/?badge=latest)

Gaussian Process kernels for stellar variability from starspot models.

**spotgp** provides analytic and numerical GP kernels derived from physically
motivated starspot models, along with GP solvers for fitting stellar
lightcurves. Both NumPy and JAX backends are available.

<br>

![Lightcurve animation](docs/tutorials/lightcurve_animation.gif)

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