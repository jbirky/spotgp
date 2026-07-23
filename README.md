# `spotgp`

[![PyPI](https://img.shields.io/pypi/v/spotgp.svg)](https://pypi.org/project/spotgp/)
[![Tests](https://github.com/jbirky/spotgp/actions/workflows/tests.yml/badge.svg)](https://github.com/jbirky/spotgp/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/jbirky/spotgp/branch/main/graph/badge.svg)](https://codecov.io/gh/jbirky/spotgp)
[![Documentation Status](https://readthedocs.org/projects/spotgp/badge/?version=latest)](https://spotgp.readthedocs.io/en/latest/?badge=latest)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://pypi.org/project/spotgp/)
[![Platform](https://img.shields.io/badge/platform-linux%20|%20macOS-lightgrey)](https://github.com/jbirky/spotgp/actions/workflows/tests.yml)

**`spotgp`**: Gaussian Process kernels for stellar starspot variability implemented in `JAX`.

<br>

![Lightcurve animation](docs/tutorials/lightcurve_animation.gif)

## Installation

### From PyPI

```bash
pip install spotgp
```

### From source

```bash
git clone https://github.com/jbirky/spotgp.git
cd spotgp
pip install -e .
```

### Optional extras

Install all optional features (PGM rendering, `jaxopt`, and the spectral
contrast model with `pyphot` + `Korg.jl`) in one shot:

```bash
pip install "spotgp[extras]"
```

Or pick individual extras:

```bash
pip install "spotgp[pgm]"        # daft PGM rendering
pip install "spotgp[jaxopt]"     # jaxopt optimizers
pip install "spotgp[spectral]"   # pyphot bandpasses
pip install "spotgp[korg]"       # pyphot + Korg.jl model atmospheres
```

Alternatively, clone the repo and add it to your Python path:

```bash
git clone https://github.com/jbirky/spotgp.git
echo 'export PYTHONPATH="$PYTHONPATH:/path/to/spotgp"' >> ~/.bashrc
source ~/.bashrc
```
