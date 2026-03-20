# `spotgp`

[![PyPI](https://img.shields.io/pypi/v/spotgp.svg)](https://pypi.org/project/spotgp/)
[![Tests](https://github.com/jbirky/spotgp/actions/workflows/tests.yml/badge.svg)](https://github.com/jbirky/spotgp/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/jbirky/spotgp/branch/main/graph/badge.svg)](https://codecov.io/gh/jbirky/spotgp)
[![Documentation Status](https://readthedocs.org/projects/spotgp/badge/?version=latest)](https://spotgp.readthedocs.io/en/latest/?badge=latest)

**`spotgp`**: Gaussian Process kernels for stellar starspot variability implemented in `JAX`.

<br>

![Lightcurve animation](docs/tutorials/lightcurve_animation.gif)

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
