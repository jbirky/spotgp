# `spotgp`

[![Documentation Status](https://readthedocs.org/projects/spotgp/badge/?version=latest)](https://spotgp.readthedocs.io/en/latest/?badge=latest)

**`spotgp`**: Gaussian Process kernels for stellar starspot variability implemented in `JAX`.

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