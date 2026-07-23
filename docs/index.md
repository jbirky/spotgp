# Home

**spotgp:** Gaussian Process kernels for stellar starspot variability implemented in `JAX`.

<br>

![Lightcurve animation](tutorials/lightcurve_animation.gif)

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

Rendering probabilistic graphical models requires `daft`:

```bash
pip install "spotgp[pgm]"
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
:caption: Fundamentals

self
tutorials/quickstart
tutorials/time_domain
tutorials/fourier_domain
overview
tutorials/lightcurve_demo
tutorials/sympy_tools
tutorials/analytic_vs_numerical_kernel
tutorials/jax_jit
tutorials/gp_optimization
tutorials/data_preprocessing
tutorials/save_load
```

```{toctree}
:maxdepth: 2
:caption: Custom Functions

tutorials/parameter_registry
tutorials/custom_envelope_gaussian
tutorials/custom_visibility_function
tutorials/custom_latitude_distribution
tutorials/custom_spot_distributions
tutorials/limb_darkening
```

```{toctree}
:maxdepth: 2
:caption: Probabilistic Inference

tutorials/pgm_visualization
tutorials/blackjax_sampling
tutorials/dynesty_sampling
tutorials/cramer_rao_bound
```

```{toctree}
:maxdepth: 2
:caption: Multiband Photometry

tutorials/multiband_gp
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/spotgp
```
