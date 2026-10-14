---
title: 'spotgp: Physically Interpretable Gaussian Process Kernels for Starspot Variability'
tags:
  - Python
  - astronomy
  - Gaussian processes
  - stellar rotation
  - starspots
  - differential rotation
  - time series
  - JAX
authors:
  - name: Jessica Birky
    orcid: 0000-0002-7961-6881
    corresponding: true
    affiliation: "1, 2"
  - name: Eric Agol
    orcid: 0000-0002-0802-9145
    affiliation: 1
  - name: Rory Barnes
    orcid: 0000-0001-6487-5445
    affiliation: 1
  - name: James R. A. Davenport
    orcid: 0000-0002-0637-835X
    affiliation: 1
  - name: Sean Matt
    orcid: 0000-0001-9590-2274
    affiliation: 2
affiliations:
  - name: Department of Astronomy, University of Washington, Seattle, WA, USA
    index: 1
  - name: Homer L. Dodge Department of Physics and Astronomy, University of Oklahoma, Norman, OK 73019, USA
    index: 2
date: 21 August 2026
bibliography: paper.bib
---

# Summary

Large-scale time-domain surveys, such as *Kepler*, *K2*, *TESS* and *PLATO*, are enabling astronomers to study the dynamic surfaces of millions of stars. In particular, low-mass stars are covered in dark magnetic spots that rotate in and out of view, as well as emerge, grow, and decay stochastically over time. These spots cause a star's observed brightness to vary quasi-periodically with time, encoding not only how fast a star spins but also how it is tilted relative to our line of sight, whether its equator rotates faster than its poles (differential rotation), and how long its spots live. 

`spotgp` is a Python package for modeling spotted lightcurves and statistically inferring spot properties from stellar lightcurves using Gaussian processes. A Gaussian process (GP) [@rasmussen_gaussian_2006] is a probabilistic model whose covariance function, or kernel, specifies how strongly the brightness at two different times is correlated. The `spotgp` kernel is derived analytically from a model of circular spots on a differentially rotating star [@kipping_analytic_2012] (\autoref{fig:model}), so each of its hyperparameters is a physical quantity: the equatorial rotation period $P_{\mathrm{eq}}$, the differential rotation shear $\kappa$, the stellar inclination $I$, the spot lifetime $\ell_{\mathrm{spot}}$, the spot emergence and decay timescales, and an amplitude $\sigma_k$ [@birky_starspot_2026]. Fitting the kernel to a lightcurve therefore gives posterior distributions over stellar and spot properties directly.

The package is built on `JAX` [@bradbury_jax_2018] for just-in-time compilation, automatic differentiation, and GPU execution, and includes a forward light-curve simulator, a scalable GP solver, and interfaces to gradient-based and nested Markov chain Monte Carlo methods for posterior sampling.

![The forward model underlying the `spotgp` kernel. Circular spots emerge and decay at different latitudes on a differentially rotating, inclined star (left; color shows the local rotation rate), and each contributes a flux deficit as it rotates across the visible disk (right; thin lines are individual spots, the black line is the total). The GP kernel is the analytic autocovariance of this process.\label{fig:model}](figures/spot_model_lightcurve.png)

# Statement of need

Rotation periods are now measured routinely for large samples of stars, but stellar inclination, differential rotation, and spot lifetimes are not, despite their importance for stellar dynamo theory, gyrochronology, and the treatment of stellar activity as noise in exoplanet surveys. These quantities have mostly been measured with spectroscopy, asteroseismology, or spot-crossing transits, which restricts them to stars with high-resolution spectra, detectable oscillations, or transiting planets. Measuring them from photometry alone has proven difficult. A blind hare-and-hounds exercise found little correlation between the reported and true differential rotation for existing photometric methods [@aigrain_testing_2015], an information-content analysis of analytic spot models suggests that differential rotation is nearly inaccessible from lightcurves alone [@basri_information_2020], and the hyperparameters of the widely used quasi-periodic GP kernel do not map cleanly onto spot properties [@nicholson_quasi-periodic_2022].
`spotgp` is designed to make these quantities measurable from a single photometric time series, and is aimed at researchers analyzing stellar lightcurves who want physically interpretable spot and rotation parameters with probabilistic uncertainties.

# State of the field

There are a number of tools available for forward modeling lightcurves of spotted stars (e.g., `macula` [@kipping_analytic_2012], `butterpy` [@claytor_recovery_2022], `starry` [@luger_starry_2019]) using analytic models of the flux blocked by spots on a rotating star. Fitting them to data means sampling the properties of every individual spot, and they provide no covariance function for GP inference.

Simultaneously, there are also a number of packages specialized for fitting GPs to astronomical time series. `celerite` and `celerite2` [@foreman-mackey_fast_2017; @foreman-mackey_scalable_2018], `george` [@ambikasaran_fast_2015], `tinygp` [@foreman-mackey_tinygp_2024], and `exoplanet` [@foreman-mackey_exoplanet_2021] provide fast solvers together with quasi-periodic or damped-oscillator kernels that fit spotted-star lightcurves well and recover rotation periods [@angus_inferring_2018].

However, the kernel models implemented in these packages are not designed specifically for describing the physical properties of starspots. Apart from the period, their hyperparameters are not tied to spot physics, and none includes differential rotation.
The closest relative is `starry_process` [@luger_mapping_2021-1], an interpretable GP whose hyperparameters describe the latitude, size, and contrast distribution of spots on a static surface; it does not model spot emergence and decay or differential rotation, which are the temporal effects `spotgp` targets.

# Software design

`spotgp` is designed to be modular and flexible, allowing users to easily customize their model for different physical assumptions, while also being optimized for computational speed. Figure \autoref{fig:arch} shows an overview of the architecture: a model layer (yellow) where the physical parameters (purple) and assumptions of the model are defined, a simulation layer (orange) containing the framework to generate sythetic light curves, a kernel layer (blue) containing the tools to compute the GP kernel matrix, a data layer (green) for processing the lightcurve data, and finally, an inference layer implementing the GP solver and interface with the MCMC samplers.

The kernel function describes the autocorrelation of the flux as a function of time lag $\tau$, which factorizes as
$$
k(\tau) = \sigma_k^2 \int \,
  \underbrace{R_\Gamma(\tau)\vphantom{\sum_{n\geq 1}}}_{\substack{\text{Envelope}\\\text{function}}} \,
  \Big[\underbrace{|c_0(I,\Phi)|^2 + 2\sum_{n\geq
  1}|c_n(I,\Phi)|^2\cos\!\big(n\,\omega_0(\Phi)\,\tau\big)}_{\substack{\text{Visibility}\\\text{function}}}\Big]\,
  \underbrace{p(\Phi)\vphantom{\sum_{n\geq 1}} \, d\Phi}_{\substack{\text{Latitude}\\\text{distribution}}} ,
$$
where $R_\Gamma$ is the autocorrelation of the spot emergence-and-decay envelope, $c_n$ are Fourier coefficients of the spot visibility function, $\omega_0(\Phi)=2\pi(1-\kappa\sin^2\Phi)/P_{\mathrm{eq}}$ is the latitude-dependent rotation frequency, and $p(\Phi)$ is the spot latitude distribution. 

Each factor is a base class (`EnvelopeFunction`, `VisibilityFunction`,
`LatitudeDistributionFunction`) which defines the spot evolution model (`SpotEvolutionModel`). This modularity allows for users to create custom inheritance clases allowing the user to change one physical assumption without touching the kernel evaluation, the latitude quadrature, the compiled likelihood, or the samplers. 

The same `SpotEvolutionModel` can be used for the time series forward model (`LightcurveModel`) to generate synthetic lightcurves or can be used directly to compute the GP covariance matrix from the kernel function. Because the simulation layer and kernel layer share the same `SpotEvolutionModel`, it is easy to validate the analytic kernel against simulations with identical assumptions (through the `NumericalKernel` class), allowing the user to do quick sanity checks when implementing new analytic equations into their model.

![Architecture of the `spotgp` package. \label{fig:arch}](figures/architecture.png)

All modules are implemented `JAX` so that the whole pipeline (latitude quadrature, harmonic sum, covariance assembly, Cholesky factorization, and log-likelihood) is one JIT-compiled, differentiable function. It can be sampled with the No-U-Turn Sampler [@hoffman_no-u-turn_2014] through `BlackJAX` [@cabezas_blackjax_2024] or with nested sampling through `dynesty` [@speagle_dynesty_2020]. 

Naive GP solvers perform poorly with the number of datapoints $N$, scaling as $\mathcal{O}(N^3)$ [@rasmussen_gaussian_2006]. However, for finite-lived spots our covariance matrix becomes banded with a bandwidth $b$ set by the spot lifetimes and cadence. Our package exploits this structure to save memory and computation time implement a blocked banded Cholesky factorization [@golub_matrix_2013] written with `jax.lax.scan` which runs in $\mathcal{O}(Nb^2)$ time.

# Research impact statement

`spotgp` is the computational basis of a companion paper [@birky_starspot_2026] that derives the kernel, verifies the derivation with `SymPy`, validates the kernel against Monte Carlo simulations, and uses MCMC fits and Cramér–Rao bounds on simulated lightcurves to map how well rotation, inclination, and spot parameters can be recovered as a function of cadence, baseline, and noise level. That paper is fully reproducible: every figure is generated by `spotgp` scripts under `showyourwork`, including benchmarks of the `spotgp` solver against `tinygp` and `george`. 

`spotgp` has been developed on GitHub since July 2023 with comprehensive unit tests, run in continuous integration on Linux and macOS for Python 3.10–3.12. It is available through PyPI and can e installed via `pip install spotgp`.

# AI usage disclosure

Generative AI tools (Anthropic's Claude, via the Claude Code assistant) were used as a coding and writing aid during the development of `spotgp` and the preparation of this manuscript: to draft and edit prose in the documentation and in this paper, to scaffold unit tests and boilerplate, and to assist with refactoring. All scientific claims, derivations, and results were produced and checked by the authors.

# Acknowledgements

E.A. was supported by the National Aeronautics and Space Administration Exoplanet Research Program (NASA XRP) grant 80NSSC21K1111. This work was performed in part by the Virtual Planetary Laboratory Team, which is a member of the NASA Nexus for Exoplanet System Science, and funded via the NASA Astrobiology Program ICAR Grant 80NSSC23K1398. 

# References
