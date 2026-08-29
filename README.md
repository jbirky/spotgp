# `spotgp`

[![PyPI](https://img.shields.io/pypi/v/spotgp.svg)](https://pypi.org/project/spotgp/)
[![Tests](https://github.com/jbirky/spotgp/actions/workflows/tests.yml/badge.svg)](https://github.com/jbirky/spotgp/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/jbirky/spotgp/branch/main/graph/badge.svg)](https://codecov.io/gh/jbirky/spotgp)
[![Documentation Status](https://readthedocs.org/projects/spotgp/badge/?version=latest)](https://spotgp.readthedocs.io/en/latest/?badge=latest)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://pypi.org/project/spotgp/)
[![Platform](https://img.shields.io/badge/platform-linux%20|%20macOS-lightgrey)](https://github.com/jbirky/spotgp/actions/workflows/tests.yml)

**`spotgp`**: Gaussian Process kernels for stellar starspot variability implemented in `JAX`.

`spotgp` provides a GP kernel for photometric stellar variability that is derived
analytically from a model of starspots on a differentially rotating star, so that
every kernel hyperparameter is a physical quantity: the equatorial rotation period,
differential rotation shear, stellar inclination, spot lifetime and emergence/decay
timescale, and amplitude. It is aimed at researchers fitting *Kepler*, *K2*, *TESS*,
or ground-based light curves who want physically interpretable spot and rotation
parameters with calibrated uncertainties. The package includes a matched forward
light-curve simulator, a banded Cholesky GP solver that scales as O(N b²), analytic
marginalization of per-segment flux offsets, composable kernel terms, and interfaces
to gradient-based (`BlackJAX`) and nested (`dynesty`) samplers. Full documentation and
tutorials are at <https://spotgp.readthedocs.io>.

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

## Quick example

```python
import jax.numpy as jnp
from spotgp import (TrapezoidSymmetricEnvelope, VisibilityFunction,
                    SpotEvolutionModel, GPSolver)

model = SpotEvolutionModel(
    envelope=TrapezoidSymmetricEnvelope(lspot=20.0, tau_spot=5.0),  # spot lifetime [d]
    visibility=VisibilityFunction(peq=5.0, kappa=0.2, inc=jnp.radians(60.0)),
    sigma_k=0.01,
)
gp = GPSolver(t, flux, flux_err, model)   # t, flux, flux_err: 1-D arrays
print(gp.log_likelihood())
```

See the [quickstart](https://spotgp.readthedocs.io/en/latest/tutorials/quickstart.html)
for fitting, sampling, and plotting.

## Tests

```bash
pip install pytest
JAX_PLATFORMS=cpu pytest tests/
```

## Contributing, support, and citing

Bug reports and feature requests are welcome on the
[issue tracker](https://github.com/jbirky/spotgp/issues); see
[CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and the
pull-request process. If you use `spotgp` in your research, please cite it using
the metadata in [CITATION.cff](CITATION.cff). `spotgp` is released under the MIT
licence (see [LICENSE](LICENSE)).
