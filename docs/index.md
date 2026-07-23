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

`Korg.jl` itself is installed from Python after the extras:

```bash
python -c "import juliapkg; juliapkg.add('Korg', 'acafc109-a718-429c-b0e5-afd7f8c7ae46'); juliapkg.resolve()"
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
:caption: Getting Started

self
tutorials/quickstart
```

```{toctree}
:maxdepth: 2
:caption: Fundamentals

overview
tutorials/time_domain
tutorials/fourier_domain
tutorials/lightcurve_demo
tutorials/sympy_tools
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
tutorials/spectral_contrast
tutorials/spots_and_faculae
```

```{toctree}
:maxdepth: 2
:caption: Performance Tests

tutorials/analytic_vs_numerical_kernel
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/spotgp
```
