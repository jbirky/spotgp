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
