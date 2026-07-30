"""spotgp — Gaussian Process kernels for stellar variability from starspot models."""

__version__ = "1.3.0"

from .distributions import (
    ParameterDistribution, DeltaDistribution, UniformDistribution,
    GaussianDistribution, LogNormalDistribution,
    as_distribution, is_distributed,
)
from .envelope import (
    EnvelopeFunction, TrapezoidSymmetricEnvelope, TrapezoidAsymmetricEnvelope,
    SkewedGaussianEnvelope, ExponentialEnvelope, ExponentialAsymmetricEnvelope,
    compute_R_Gamma_numerical,
)
from .latitude import LatitudeDistributionFunction, UniformDoubleHemisphereBand
from .visibility import (
    VisibilityFunction, EdgeOnVisibilityFunction, FullGeometryVisibilityFunction,
    LimbDarkenedVisibilityFunction,
)
from .spot_model import SpotEvolutionModel
from .lightcurve import LightcurveModel, compute_sigmak
from .analytic_kernel import AnalyticKernel, NonstationaryAnalyticKernel
from .terms import (
    Term, KernelSum, SpotTerm, SharedVisibilitySpotSum,
    PopulationSpotTerm, SHOTerm, Matern32Term, JitterTerm,
)
from .random_variables import (
    Hyper, Latent, LogNormalLatent, NormalLatent, UniformLatent,
    Derived, UniformEmergence, UniformLongitude, SpotRandomVariables,
    gnevyshev_waldmeier,
)
from .numerical_kernel import NumericalKernel, generate_sims, avg_covariance_tlag
from .psd import compute_psd
from .contrast import spot_contrast, contrast_factor, contrast_matrix
from .gp_solver import GPSolver
from .multiband import MultiBandData, MultiBandGPSolver, SpotFaculaeGPSolver
from .spectral import (
    KorgProvider, BlackbodyProvider, BandpassSet, SpectralContrastModel,
)
from .pgm import PGModelVis
from .mcmc import MCMCSampler, BlackJAXSampler, DynestySampler
from .observations import TimeSeriesData
from .results import MAPResult, is_complete, mark_complete
from .io import save_gp, load_gp, save_sampler, load_sampler, load_samples
from .plotting import crb_corner_plot
from .sensitivity import sobol_indices
from .transit import KeplerianOrbit, QuadLimbDarkLightCurve, SpotTransitModel
from .params import (
    EnvelopeSpec, AmplitudeSpec, register_envelope, register_amplitude,
    resolve_hparam, KERNEL_HPARAM_KEYS, HPARAM_KEYS_WITH_NOISE,
)

__all__ = [
    # distributions
    "ParameterDistribution", "DeltaDistribution", "UniformDistribution",
    "GaussianDistribution", "LogNormalDistribution",
    "as_distribution", "is_distributed",
    # envelope
    "EnvelopeFunction", "TrapezoidSymmetricEnvelope", "TrapezoidAsymmetricEnvelope",
    "SkewedGaussianEnvelope", "ExponentialEnvelope", "ExponentialAsymmetricEnvelope",
    "compute_R_Gamma_numerical",
    # latitude
    "LatitudeDistributionFunction", "UniformDoubleHemisphereBand",
    # visibility
    "VisibilityFunction", "EdgeOnVisibilityFunction", "FullGeometryVisibilityFunction",
    "LimbDarkenedVisibilityFunction",
    # spot_model
    "SpotEvolutionModel",
    # lightcurve
    "LightcurveModel", "compute_sigmak",
    # kernel
    "AnalyticKernel", "NonstationaryAnalyticKernel",
    "NumericalKernel", "generate_sims", "avg_covariance_tlag",
    # composable kernel terms
    "Term", "KernelSum", "SpotTerm", "SharedVisibilitySpotSum",
    "PopulationSpotTerm", "SHOTerm", "Matern32Term", "JitterTerm",
    # random-variable declarations
    "Hyper", "Latent", "LogNormalLatent", "NormalLatent", "UniformLatent",
    "Derived", "UniformEmergence", "UniformLongitude",
    "SpotRandomVariables", "gnevyshev_waldmeier",
    # psd
    "compute_psd",
    # contrast
    "spot_contrast", "contrast_factor", "contrast_matrix",
    # solver
    "GPSolver",
    # multiband
    "MultiBandData", "MultiBandGPSolver", "SpotFaculaeGPSolver",
    # spectral contrast
    "KorgProvider", "BlackbodyProvider", "BandpassSet", "SpectralContrastModel",
    # pgm
    "PGModelVis",
    # mcmc
    "MCMCSampler", "BlackJAXSampler", "DynestySampler",
    # observations
    "TimeSeriesData",
    # results
    "MAPResult", "is_complete", "mark_complete",
    # io
    "save_gp", "load_gp", "save_sampler", "load_sampler", "load_samples",
    # plotting
    "crb_corner_plot",
    # sensitivity
    "sobol_indices",
    # transit
    "KeplerianOrbit", "QuadLimbDarkLightCurve", "SpotTransitModel",
    # params
    "EnvelopeSpec", "AmplitudeSpec", "register_envelope", "register_amplitude",
    "resolve_hparam", "KERNEL_HPARAM_KEYS", "HPARAM_KEYS_WITH_NOISE",
]
