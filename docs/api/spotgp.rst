API Reference (``spotgp``)
============================

.. module:: spotgp

Lightcurve Model
----------------

.. autoclass:: spotgp.lightcurve.LightcurveModel
   :members:
   :undoc-members:
   :show-inheritance:

Analytic Kernel
---------------

.. autoclass:: spotgp.analytic_kernel.AnalyticKernel
   :members:
   :undoc-members:
   :show-inheritance:

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

.. autoclass:: spotgp.gp_solver.GPSolver
   :members:
   :undoc-members:
   :show-inheritance:

Power Spectral Density
----------------------

.. autofunction:: spotgp.psd.compute_psd

MCMC Sampler
------------

.. autoclass:: spotgp.mcmc.MCMCSampler
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.mcmc.BlackJAXSampler
   :members:
   :undoc-members:
   :show-inheritance:

Spot Model
----------

.. autoclass:: spotgp.spot_model.SpotEvolutionModel
   :members:
   :undoc-members:
   :show-inheritance:

Observations
------------

.. autoclass:: spotgp.observations.TimeSeriesData
   :members:
   :undoc-members:
   :show-inheritance:

Distributions
-------------

.. autoclass:: spotgp.distributions.ParameterDistribution
   :members:
   :undoc-members:
   :show-inheritance:

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

Parameters
----------

.. autoclass:: spotgp.params.EnvelopeSpec
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.params.AmplitudeSpec
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: spotgp.params.register_envelope

.. autofunction:: spotgp.params.register_amplitude

.. autofunction:: spotgp.params.resolve_hparam

Envelope Functions
------------------

.. autoclass:: spotgp.envelope.EnvelopeFunction
   :members:
   :undoc-members:
   :show-inheritance:

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

Latitude Distribution
---------------------

.. autoclass:: spotgp.latitude.LatitudeDistributionFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.latitude.UniformDoubleHemisphereBand
   :members:
   :undoc-members:
   :show-inheritance:

Visibility Functions
--------------------

.. autoclass:: spotgp.visibility.VisibilityFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.visibility.EdgeOnVisibilityFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.visibility.FullGeometryVisibilityFunction
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

Sensitivity Analysis
--------------------

.. autofunction:: spotgp.sensitivity.sobol_indices

Plotting
--------

.. autofunction:: spotgp.plotting.crb_corner_plot
