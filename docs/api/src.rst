API Reference (``src``)
============================

.. module:: src

Lightcurve Model
----------------

.. autoclass:: src.starspot.LightcurveModel
   :members:
   :undoc-members:
   :show-inheritance:

Analytic Kernel
---------------

.. autoclass:: src.analytic_kernel.AnalyticKernel
   :members:
   :undoc-members:
   :show-inheritance:

Numerical Kernel
----------------

.. autoclass:: src.numerical_kernel.NumericalKernel
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: src.numerical_kernel.generate_sims

.. autofunction:: src.numerical_kernel.avg_covariance_tlag

GP Solver
---------

.. autoclass:: src.gp_solver.GPSolver
   :members:
   :undoc-members:
   :show-inheritance:

Power Spectral Density
----------------------

.. autofunction:: src.psd.compute_psd

MCMC Sampler
------------

.. autoclass:: src.mcmc.MCMCSampler
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.mcmc.BlackJAXSampler
   :members:
   :undoc-members:
   :show-inheritance:
