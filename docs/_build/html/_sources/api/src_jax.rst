API Reference (``src_jax``)
============================

.. module:: src_jax

Lightcurve Model
----------------

.. autoclass:: src_jax.starspot.LightcurveModel
   :members:
   :undoc-members:
   :show-inheritance:

Analytic Kernel
---------------

.. autoclass:: src_jax.analytic_kernel.AnalyticKernel
   :members:
   :undoc-members:
   :show-inheritance:

Numerical Kernel
----------------

.. autoclass:: src_jax.numerical_kernel.NumericalKernel
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: src_jax.numerical_kernel.generate_sims

.. autofunction:: src_jax.numerical_kernel.avg_covariance_tlag

GP Solver
---------

.. autoclass:: src_jax.gp_solver.GPSolver
   :members:
   :undoc-members:
   :show-inheritance:

Power Spectral Density
----------------------

.. autofunction:: src_jax.psd.compute_psd

MCMC Sampler
------------

.. autoclass:: src_jax.mcmc.MCMCSampler
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src_jax.mcmc.BlackJAXSampler
   :members:
   :undoc-members:
   :show-inheritance:
