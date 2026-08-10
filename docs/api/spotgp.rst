API Reference (``spotgp``)
============================

.. module:: spotgp

Spot Evolution Model
--------------------

The central model object.  It composes an envelope function, a visibility
function, and a latitude distribution into the spot-evolution description
that every kernel and solver consumes.

.. autoclass:: spotgp.spot_model.SpotEvolutionModel
   :members:
   :undoc-members:
   :show-inheritance:

Envelope Functions
------------------

:class:`~spotgp.envelope.EnvelopeFunction` is the base class; subclass it to
define a custom spot emergence/decay profile.

.. autoclass:: spotgp.envelope.EnvelopeFunction
   :members:
   :undoc-members:

.. autoclass:: spotgp.envelope.TrapezoidSymmetricEnvelope
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.envelope.TrapezoidAsymmetricEnvelope
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.envelope.SkewedGaussianEnvelope
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.envelope.ExponentialEnvelope
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.envelope.ExponentialAsymmetricEnvelope
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: spotgp.envelope.compute_R_Gamma_numerical

Visibility Functions
--------------------

:class:`~spotgp.visibility.VisibilityFunction` is the base class; subclass it
to define a custom stellar visibility law.  The ``harmonics`` argument selects
which rotation harmonic orders are retained, and the kernel inherits that set
unless overridden.

.. autoclass:: spotgp.visibility.VisibilityFunction
   :members:
   :undoc-members:

.. autoclass:: spotgp.visibility.EdgeOnVisibilityFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.visibility.FullGeometryVisibilityFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.visibility.LimbDarkenedVisibilityFunction
   :members:
   :undoc-members:
   :show-inheritance:

Latitude Distributions
----------------------

:class:`~spotgp.latitude.LatitudeDistributionFunction` is the base class;
subclass it to define a custom spot latitude density.

.. autoclass:: spotgp.latitude.LatitudeDistributionFunction
   :members:
   :undoc-members:

.. autoclass:: spotgp.latitude.UniformDoubleHemisphereBand
   :members:
   :undoc-members:
   :show-inheritance:

Parameter Distributions
-----------------------

:class:`~spotgp.distributions.ParameterDistribution` is the base class for
the priors attached to individual model parameters.

.. autoclass:: spotgp.distributions.ParameterDistribution
   :members:
   :undoc-members:

.. autoclass:: spotgp.distributions.DeltaDistribution
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.distributions.UniformDistribution
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.distributions.GaussianDistribution
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.distributions.LogNormalDistribution
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: spotgp.distributions.as_distribution

.. autofunction:: spotgp.distributions.is_distributed

Lightcurve Model
----------------

``inherited-members`` picks up the ``animate_*`` methods that
``LightcurveModel`` gets from its ``AnimationMixin`` base class.

.. autoclass:: spotgp.lightcurve.LightcurveModel
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

.. autofunction:: spotgp.lightcurve.compute_sigmak

Analytic Kernel
---------------

.. autoclass:: spotgp.analytic_kernel.AnalyticKernel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.analytic_kernel.NonstationaryAnalyticKernel
   :members:
   :undoc-members:
   :show-inheritance:

Composable Kernel Terms
-----------------------

.. autoclass:: spotgp.terms.Term
   :members:
   :undoc-members:
   :exclude-members: stationary, prefix

.. autoclass:: spotgp.terms.KernelSum
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.terms.SpotTerm
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.terms.SharedVisibilitySpotSum
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.terms.PopulationSpotTerm
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.terms.SHOTerm
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.terms.Matern32Term
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.terms.JitterTerm
   :members:
   :undoc-members:
   :show-inheritance:

Random-Variable Declarations
----------------------------

.. autoclass:: spotgp.random_variables.SpotRandomVariables
   :members:
   :undoc-members:

.. autoclass:: spotgp.random_variables.Hyper
   :members:

.. autoclass:: spotgp.random_variables.Latent
   :members:

.. autoclass:: spotgp.random_variables.LogNormalLatent
   :members:
   :show-inheritance:

.. autoclass:: spotgp.random_variables.NormalLatent
   :members:
   :show-inheritance:

.. autoclass:: spotgp.random_variables.UniformLatent
   :members:
   :show-inheritance:

.. autoclass:: spotgp.random_variables.Derived
   :members:

.. autoclass:: spotgp.random_variables.UniformEmergence
   :members:

.. autoclass:: spotgp.random_variables.UniformLongitude
   :members:

.. autofunction:: spotgp.random_variables.gnevyshev_waldmeier

Numerical Kernel
----------------

.. autoclass:: spotgp.numerical_kernel.NumericalKernel
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: spotgp.numerical_kernel.generate_sims

.. autofunction:: spotgp.numerical_kernel.avg_covariance_tlag

GP Solver
---------

``inherited-members`` is set so that the fitting, mass-matrix, and plotting
methods ``GPSolver`` picks up from its mixin base classes are documented here
alongside the ones it defines itself.

.. autoclass:: spotgp.gp_solver.GPSolver
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

Multi-Band GP
-------------

.. autoclass:: spotgp.multiband.MultiBandData
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.multiband.MultiBandGPSolver
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.multiband.SpotFaculaeGPSolver
   :members:
   :undoc-members:
   :show-inheritance:

Spot Contrast
-------------

.. autofunction:: spotgp.contrast.spot_contrast

.. autofunction:: spotgp.contrast.contrast_factor

.. autofunction:: spotgp.contrast.contrast_matrix

Spectral Contrast
-----------------

.. autoclass:: spotgp.spectral.SpectralContrastModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.spectral.BlackbodyProvider
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.spectral.KorgProvider
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.spectral.BandpassSet
   :members:
   :undoc-members:
   :show-inheritance:

Transit Model
-------------

.. autoclass:: spotgp.transit.KeplerianOrbit
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.transit.QuadLimbDarkLightCurve
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.transit.SpotTransitModel
   :members:
   :undoc-members:
   :show-inheritance:

Probabilistic Graphical Model
-----------------------------

.. autoclass:: spotgp.pgm.PGModelVis
   :members:
   :undoc-members:
   :show-inheritance:

Observations
------------

.. autoclass:: spotgp.observations.TimeSeriesData
   :members:
   :undoc-members:
   :show-inheritance:

Power Spectral Density
----------------------

.. autofunction:: spotgp.psd.compute_psd

MCMC Sampler
------------

:class:`~spotgp.mcmc.MCMCSampler` is the base class; the concrete samplers
below implement it.

.. autoclass:: spotgp.mcmc.MCMCSampler
   :members:
   :undoc-members:

.. autoclass:: spotgp.mcmc.BlackJAXSampler
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.mcmc.DynestySampler
   :members:
   :undoc-members:
   :show-inheritance:

Fit Results
-----------

.. autoclass:: spotgp.results.MAPResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: spotgp.results.is_complete

.. autofunction:: spotgp.results.mark_complete

Saving and Loading
------------------

.. autofunction:: spotgp.io.save_gp

.. autofunction:: spotgp.io.load_gp

.. autofunction:: spotgp.io.save_sampler

.. autofunction:: spotgp.io.load_sampler

.. autofunction:: spotgp.io.load_samples

Plotting
--------

.. autofunction:: spotgp.plotting.crb_corner_plot

Sensitivity Analysis
--------------------

.. autofunction:: spotgp.sensitivity.sobol_indices

Hyperparameter Specification
----------------------------

Registry that maps a raw hyperparameter dict onto the envelope and amplitude
parameterization it describes.

.. Both are dataclasses whose fields are already described by the docstring's
.. Attributes section, which napoleon renders.  Documenting the fields as
.. members too would emit each one twice, so member listing is turned off.

.. autoclass:: spotgp.params.EnvelopeSpec
   :no-members:
   :show-inheritance:

.. autoclass:: spotgp.params.AmplitudeSpec
   :no-members:
   :show-inheritance:

.. autofunction:: spotgp.params.register_envelope

.. autofunction:: spotgp.params.register_amplitude

.. autofunction:: spotgp.params.resolve_hparam

.. autodata:: spotgp.params.KERNEL_HPARAM_KEYS

.. autodata:: spotgp.params.HPARAM_KEYS_WITH_NOISE
