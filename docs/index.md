# Home

**spotgp:** Gaussian Process kernels for stellar starspot variability implemented in `JAX`.

<br>

![Lightcurve animation](tutorials/lightcurve_animation.gif)


## Development and support

This project is built and maintained by [Jess Birky](https://github.com/jbirky).
Bug reports, feature requests, and usage questions are welcome on the
[GitHub issue tracker](https://github.com/jbirky/spotgp/issues); see the
[contributing guide](https://github.com/jbirky/spotgp/blob/main/CONTRIBUTING.md)
for development setup, running the tests, and the pull-request process. `spotgp`
is released under the MIT licence. A paper describing the software and its
companion derivation paper are in preparation; citation metadata is kept in
[CITATION.cff](https://github.com/jbirky/spotgp/blob/main/CITATION.cff).

<br>

## Documentation Contents

```{toctree}
:maxdepth: 2
:caption: Getting Started

self
tutorials/installation
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

custom_components
tutorials/parameter_registry
tutorials/custom_envelope_gaussian
tutorials/custom_visibility_function
tutorials/custom_latitude_distribution
tutorials/custom_spot_distributions
tutorials/limb_darkening
```

```{toctree}
:maxdepth: 2
:caption: Kernels

tutorials/composite_kernels
tutorials/random_variable_declarations
tutorials/nonstationary_kernel
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
