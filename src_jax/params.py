"""
params.py — single source of truth for hyperparameter schemas,
envelope specifications, amplitude specifications, and hparam
validation/normalization.

All other modules (analytic_kernel, numerical_kernel, gp_solver, mcmc,
starspot) should import constants and call resolve_hparam() from here
rather than duplicating validation logic.

Extending with a new envelope
------------------------------
Define your resolve function, then register it::

    from params import EnvelopeSpec, register_envelope

    def _resolve_gaussian(raw: dict) -> dict:
        # raw contains 'tau_gauss'; inject 'tau' for scalar-tau compat
        return {"tau": raw["tau_gauss"]}

    register_envelope(EnvelopeSpec(
        name="gaussian",
        signature_keys=frozenset({"tau_gauss"}),
        resolve=_resolve_gaussian,
        description="Gaussian decay: tau_gauss sets the 1/e timescale",
    ))

Your kernel can then read raw["tau_gauss"] directly after calling
resolve_hparam(); every other module (gp_solver, mcmc) is unchanged
because resolve_hparam always injects "sigma_k" and "tau".

Extending with a new amplitude parameterization
-------------------------------------------------
Register an AmplitudeSpec with a formula callable::

    from params import AmplitudeSpec, register_amplitude

    register_amplitude(AmplitudeSpec(
        name="contrast_weighted",
        signature_keys=frozenset({"nspot_rate", "fspot", "alpha_max", "contrast"}),
        formula=lambda raw: (
            np.sqrt(raw["nspot_rate"]) * raw["contrast"] * raw["alpha_max"]**2
        ),
        description="sigma_k weighted by an explicit contrast parameter",
    ), priority="high")

Detection order
---------------
Specs are matched by checking whether signature_keys ⊆ raw.keys().
More-specific specs (larger signature_keys) are tested first so that a
user who provides a superset of keys always gets the most specific match.
Within the same specificity tier, registration order determines priority;
use priority="high" to prepend instead of append.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, FrozenSet

__all__ = [
    # Spec classes
    "EnvelopeSpec",
    "AmplitudeSpec",
    # Registry API
    "register_envelope",
    "register_amplitude",
    # Core function
    "resolve_hparam",
    # Constants
    "BASE_REQUIRED_KEYS",
    "KERNEL_HPARAM_KEYS",
    "HPARAM_KEYS_WITH_NOISE",
    # Backward-compat aliases (used by existing imports in analytic_kernel / gp_solver)
    "_REQUIRED_KEYS",
    "_AMPLITUDE_KEYS_SIGMA",
    "_AMPLITUDE_KEYS_PHYSICAL_RATE",
    "_AMPLITUDE_KEYS_PHYSICAL",
]


# ── Key constants ──────────────────────────────────────────────────────────────

# Base keys required by every kernel (rotation geometry + spot size)
BASE_REQUIRED_KEYS: FrozenSet[str] = frozenset({"peq", "kappa", "inc", "lspot"})

# Canonical ordered tuple for theta vectors, corner-plot labels, etc.
# GPSolver and MCMCSampler use this to map array positions to param names.
KERNEL_HPARAM_KEYS: tuple[str, ...] = ("peq", "kappa", "inc", "lspot", "tau", "sigma_k")

# Extended version that includes the optional white-noise term
HPARAM_KEYS_WITH_NOISE: tuple[str, ...] = KERNEL_HPARAM_KEYS + ("sigma_n",)

# Backward-compat aliases — existing modules import these by name
_REQUIRED_KEYS: FrozenSet[str] = BASE_REQUIRED_KEYS | {"tau"}
_AMPLITUDE_KEYS_SIGMA: FrozenSet[str] = frozenset({"sigma_k"})
_AMPLITUDE_KEYS_PHYSICAL_RATE: FrozenSet[str] = frozenset({"nspot_rate", "fspot", "alpha_max"})
_AMPLITUDE_KEYS_PHYSICAL: FrozenSet[str] = frozenset({"nspot", "fspot", "alpha_max"})


# ── EnvelopeSpec ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EnvelopeSpec:
    """
    Specification for a spot envelope shape.

    Attributes
    ----------
    name : str
        Unique identifier (e.g. "trapezoid_symmetric").
    signature_keys : frozenset[str]
        Keys that identify this envelope in a raw hparam dict.
        Detection checks ``signature_keys <= raw.keys()``; the most
        specific match (largest signature) wins.
    resolve : callable
        ``(raw: dict) -> dict``  Returns *additional* key-value pairs to
        merge into the resolved hparam.  Must always include ``"tau"``
        (a scalar timescale) for backward compatibility with modules that
        require a single timescale value.
    description : str
        Human-readable summary shown in error messages.
    """
    name: str
    signature_keys: FrozenSet[str]
    resolve: Callable[[dict], dict]
    description: str = ""


# Ordered list; sorted by len(signature_keys) descending at lookup time
# so more-specific specs always win.
_ENVELOPE_REGISTRY: list[EnvelopeSpec] = []


def register_envelope(spec: EnvelopeSpec, priority: str = "low") -> None:
    """
    Register an envelope specification.

    Parameters
    ----------
    spec : EnvelopeSpec
    priority : {"low", "high"}
        "high" prepends (checked first within its specificity tier);
        "low" appends (default).  Specificity (len of signature_keys)
        always takes precedence over priority.
    """
    if any(s.name == spec.name for s in _ENVELOPE_REGISTRY):
        raise ValueError(f"EnvelopeSpec {spec.name!r} is already registered.")
    if priority == "high":
        _ENVELOPE_REGISTRY.insert(0, spec)
    else:
        _ENVELOPE_REGISTRY.append(spec)


def _detect_envelope(raw: dict) -> EnvelopeSpec | None:
    """Return the best-matching registered EnvelopeSpec, or None."""
    candidates = [s for s in _ENVELOPE_REGISTRY if s.signature_keys <= raw.keys()]
    if not candidates:
        return None
    # Most-specific match wins; ties broken by registration order (list order)
    return max(candidates, key=lambda s: len(s.signature_keys))


# ── AmplitudeSpec ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AmplitudeSpec:
    """
    Specification for a kernel amplitude parameterization.

    Attributes
    ----------
    name : str
        Unique identifier.
    signature_keys : frozenset[str]
        Keys that identify this amplitude mode in a raw hparam dict.
    formula : callable
        ``(raw: dict) -> float``  Computes sigma_k from the raw dict.
    description : str
        Human-readable summary shown in error messages.
    """
    name: str
    signature_keys: FrozenSet[str]
    formula: Callable[[dict], float]
    description: str = ""


_AMPLITUDE_REGISTRY: list[AmplitudeSpec] = []


def register_amplitude(spec: AmplitudeSpec, priority: str = "low") -> None:
    """
    Register an amplitude specification.

    Parameters
    ----------
    spec : AmplitudeSpec
    priority : {"low", "high"}
        Same semantics as register_envelope.
    """
    if any(s.name == spec.name for s in _AMPLITUDE_REGISTRY):
        raise ValueError(f"AmplitudeSpec {spec.name!r} is already registered.")
    if priority == "high":
        _AMPLITUDE_REGISTRY.insert(0, spec)
    else:
        _AMPLITUDE_REGISTRY.append(spec)


def _detect_amplitude(raw: dict) -> AmplitudeSpec | None:
    """Return the best-matching registered AmplitudeSpec, or None."""
    candidates = [s for s in _AMPLITUDE_REGISTRY if s.signature_keys <= raw.keys()]
    if not candidates:
        return None
    return max(candidates, key=lambda s: len(s.signature_keys))


# ── Built-in envelope registrations ───────────────────────────────────────────

def _resolve_trapezoid_symmetric(raw: dict) -> dict:
    return {"tau": raw["tau"]}


def _resolve_trapezoid_asymmetric(raw: dict) -> dict:
    tau_em = float(raw["tau_em"])
    tau_dec = float(raw["tau_dec"])
    return {
        "tau_em": tau_em,
        "tau_dec": tau_dec,
        # Scalar tau for modules that need a single timescale (e.g. PSD, bandwidth)
        "tau": (tau_em + tau_dec) / 2.0,
    }


register_envelope(EnvelopeSpec(
    name="trapezoid_symmetric",
    signature_keys=frozenset({"tau"}),
    resolve=_resolve_trapezoid_symmetric,
    description="Symmetric trapezoid: lspot (plateau) + tau (rise/decay timescale)",
))

register_envelope(EnvelopeSpec(
    name="trapezoid_asymmetric",
    signature_keys=frozenset({"tau_em", "tau_dec"}),
    resolve=_resolve_trapezoid_asymmetric,
    description="Asymmetric trapezoid: lspot + tau_em (rise) + tau_dec (decay)",
))


def _resolve_skew_normal(raw: dict) -> dict:
    sigma_sn = float(raw["sigma_sn"])
    n_sn = float(raw["n_sn"])
    return {
        "sigma_sn": sigma_sn,
        "n_sn": n_sn,
        # scalar tau for modules that need a single timescale
        "tau": sigma_sn,
    }


register_envelope(EnvelopeSpec(
    name="skew_normal",
    signature_keys=frozenset({"sigma_sn", "n_sn"}),
    resolve=_resolve_skew_normal,
    description=(
        "Skew-normal: sigma_sn (scale [days]) + n_sn (skewness, dimensionless). "
        "Eq. (1) of Baranyi et al. (2021) A&A 653, A59. "
        "n_sn < 0: rapid rise / slow decay; "
        "n_sn > 0: slow rise / rapid decay; "
        "n_sn = 0: Gaussian envelope. "
        "lspot is required by the base schema but unused; set to 0."
    ),
))


# ── Built-in amplitude registrations ──────────────────────────────────────────

register_amplitude(AmplitudeSpec(
    name="sigma_k_direct",
    signature_keys=frozenset({"sigma_k"}),
    formula=lambda raw: float(raw["sigma_k"]),
    description="sigma_k provided directly",
))

register_amplitude(AmplitudeSpec(
    name="physical_rate",
    signature_keys=frozenset({"nspot_rate", "fspot", "alpha_max"}),
    formula=lambda raw: (
        np.sqrt(float(raw["nspot_rate"]))
        * (1.0 - float(raw["fspot"]))
        * float(raw["alpha_max"]) ** 2
    ),
    description=(
        "sigma_k = sqrt(nspot_rate) * (1 - fspot) * alpha_max^2  "
        "[nspot_rate in spots/day; preferred]"
    ),
))

register_amplitude(AmplitudeSpec(
    name="physical_count",
    signature_keys=frozenset({"nspot", "fspot", "alpha_max"}),
    formula=lambda raw: (
        np.sqrt(float(raw["nspot"]))
        * (1.0 - float(raw["fspot"]))
        * float(raw["alpha_max"]) ** 2
        / np.pi
    ),
    description=(
        "Legacy: sigma_k = sqrt(nspot) * (1 - fspot) * alpha_max^2 / pi  "
        "[nspot is total count, biased by tsim; prefer physical_rate]"
    ),
))


# ── resolve_hparam ─────────────────────────────────────────────────────────────

def resolve_hparam(raw: dict) -> dict:
    """
    Validate and normalise a raw hyperparameter dict.

    Steps
    -----
    1. Checks that base required keys (peq, kappa, inc, lspot) are present.
    2. Auto-detects the envelope type from registered EnvelopeSpecs and
       merges any derived keys (e.g. scalar ``tau`` from tau_em/tau_dec).
    3. Auto-detects the amplitude mode from registered AmplitudeSpecs and
       injects the computed ``sigma_k``.

    The most-specific matching spec (largest signature_keys) wins for
    both envelope and amplitude detection.

    Returns
    -------
    dict
        A new dict containing all original keys plus any keys injected by
        the envelope and amplitude resolvers.  The returned dict always
        contains ``"tau"`` and ``"sigma_k"``.

    Raises
    ------
    TypeError
        If *raw* is not a dict.
    ValueError
        If required keys are missing, or if no registered spec matches.
    """
    if not isinstance(raw, dict):
        raise TypeError(f"hparam must be a dict, got {type(raw).__name__!r}")

    missing = BASE_REQUIRED_KEYS - raw.keys()
    if missing:
        raise ValueError(
            f"hparam missing required keys: {sorted(missing)}.  "
            f"All of {sorted(BASE_REQUIRED_KEYS)} must be present."
        )

    out = dict(raw)

    # ── Envelope ──────────────────────────────────────────────────────────────
    env_spec = _detect_envelope(raw)
    if env_spec is None:
        _fmt = "\n".join(
            f"  {s.name}: keys={sorted(s.signature_keys)}  — {s.description}"
            for s in sorted(_ENVELOPE_REGISTRY, key=lambda s: -len(s.signature_keys))
        )
        raise ValueError(
            f"No envelope spec matched the provided keys {sorted(raw.keys())}.\n"
            f"Registered envelopes:\n{_fmt}"
        )
    out.update(env_spec.resolve(raw))

    # ── Amplitude ─────────────────────────────────────────────────────────────
    amp_spec = _detect_amplitude(raw)
    if amp_spec is None:
        _fmt = "\n".join(
            f"  {s.name}: keys={sorted(s.signature_keys)}  — {s.description}"
            for s in sorted(_AMPLITUDE_REGISTRY, key=lambda s: -len(s.signature_keys))
        )
        raise ValueError(
            f"No amplitude spec matched the provided keys {sorted(raw.keys())}.\n"
            f"Registered amplitude modes:\n{_fmt}"
        )
    out["sigma_k"] = amp_spec.formula(raw)

    return out
