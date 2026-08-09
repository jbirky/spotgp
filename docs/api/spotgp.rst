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

.. autoclass:: spotgp.gp_solver.GPSolver
   :members:
   :undoc-members:
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

.. autoclass:: spotgp.mcmc.MCMCSampler
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spotgp.mcmc.BlackJAXSampler
   :members:
   :undoc-members:
   :show-inheritance:
