API Reference (``src_numpy``)
=============================

.. module:: src_numpy

Lightcurve Model
----------------

.. autoclass:: src_numpy.starspot.LightcurveModel
   :members:
   :undoc-members:
   :show-inheritance:

Analytic Kernel
---------------

.. autoclass:: src_numpy.analytic_kernel.AnalyticKernel
   :members:
   :undoc-members:
   :show-inheritance:

Numerical Kernel
----------------

.. autoclass:: src_numpy.numerical_kernel.NumericalKernel
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: src_numpy.numerical_kernel.generate_sims

.. autofunction:: src_numpy.numerical_kernel.avg_covariance_tlag

GP Solver
---------

.. autoclass:: src_numpy.gp_solver.GPSolver
   :members:
   :undoc-members:
   :show-inheritance:

Power Spectral Density
----------------------

.. autofunction:: src_numpy.psd.compute_psd

MCMC Sampler
------------

.. autoclass:: src_numpy.mcmc.MCMCSampler
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src_numpy.mcmc.BlackJAXSampler
   :members:
   :undoc-members:
   :show-inheritance:
