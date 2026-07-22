# Parameter Registry

The `params` module is the single source of truth for hyperparameter validation
and normalization in spotgp. Every module that needs hyperparameters —
`AnalyticKernel`, `NumericalKernel`, `GPSolver`, `MCMCSampler` — delegates to
`resolve_hparam()` rather than implementing its own validation.

The registry is **extensible**: you can register custom envelope shapes and
amplitude parameterizations without modifying any core code. Once registered,
your custom parameters work transparently with the full fitting and sampling
pipeline.

---

## Overview

A hyperparameter dict in spotgp describes the stellar and spot properties needed
to build a GP kernel. Every dict must contain four **base keys**:

| Key | Description | Units |
|---|---|---|
| `peq` | Equatorial rotation period | days |
| `kappa` | Differential rotation shear | dimensionless |
| `inc` | Stellar inclination | radians |
| `lspot` | Spot plateau duration | days |

On top of these, two things must be determined from the remaining keys:

1. **Envelope** — how the spot size evolves over time (sets `tau_spot`)
2. **Amplitude** — the kernel amplitude (sets `sigma_k`)

The registry detects both automatically by matching the keys you provide against
registered specs.

---

## How `resolve_hparam()` works

```python
from spotgp import resolve_hparam

hparam = {
    "peq": 10.0,
    "kappa": 0.2,
    "inc": 1.0,
    "lspot": 5.0,
    "tau_spot": 1.0,      # ← matched by the "trapezoid_symmetric" EnvelopeSpec
    "sigma_k": 0.01,      # ← matched by the "sigma_k_direct" AmplitudeSpec
}

resolved = resolve_hparam(hparam)
```

Internally, `resolve_hparam` does three things in order:

1. **Checks base keys.** Raises `ValueError` if any of `{peq, kappa, inc,
   lspot}` are missing.
2. **Detects the envelope.** Scans the envelope registry for a spec whose
   `signature_keys` are a subset of the dict's keys. The most specific match
   (largest `signature_keys`) wins. Calls the spec's `resolve()` function and
   merges the result — this always injects a scalar `tau_spot`.
3. **Detects the amplitude.** Same logic against the amplitude registry.
   Calls the spec's `formula()` function and injects the computed `sigma_k`.

The returned dict contains all original keys plus any keys injected by the
resolvers. Extra keys you include (e.g., `sigma_n` for white noise) are
passed through untouched.

---

## Built-in envelope specs

| Name | Signature keys | Description |
|---|---|---|
| `trapezoid_symmetric` | `{tau_spot}` | Symmetric trapezoid envelope. `tau_spot` is the rise/decay timescale. |
| `trapezoid_asymmetric` | `{tau_em, tau_dec}` | Asymmetric trapezoid. `tau_em` = rise, `tau_dec` = decay. Injects `tau_spot = (tau_em + tau_dec) / 2`. |
| `skew_normal` | `{sigma_sn, n_sn}` | Skew-normal shape (Baranyi et al. 2021). `sigma_sn` = scale [days], `n_sn` = skewness. Injects `tau_spot = sigma_sn`. |

### Usage examples

```python
# Symmetric (simplest)
resolve_hparam({"peq": 10, "kappa": 0.2, "inc": 1.0, "lspot": 5,
                "tau_spot": 1.0, "sigma_k": 0.01})

# Asymmetric
resolve_hparam({"peq": 10, "kappa": 0.2, "inc": 1.0, "lspot": 5,
                "tau_em": 0.5, "tau_dec": 1.5, "sigma_k": 0.01})

# Skew-normal
resolve_hparam({"peq": 10, "kappa": 0.2, "inc": 1.0, "lspot": 0,
                "sigma_sn": 3.0, "n_sn": -2.0, "sigma_k": 0.01})
```

---

## Built-in amplitude specs

| Name | Signature keys | Formula | Description |
|---|---|---|---|
| `sigma_k_direct` | `{sigma_k}` | `sigma_k` | Use directly when you already know the kernel amplitude. |
| `physical_rate` | `{nspot_rate, fspot, alpha_max}` | `sqrt(nspot_rate) * (1 - fspot) * alpha_max^2` | Physical params with spot emergence rate [spots/day]. Preferred. |
| `physical_count` | `{nspot, fspot, alpha_max}` | `sqrt(nspot) * (1 - fspot) * alpha_max^2 / pi` | Legacy: total spot count (biased by simulation length). |

### Usage examples

```python
# Direct
resolve_hparam({"peq": 10, "kappa": 0.2, "inc": 1.0, "lspot": 5,
                "tau_spot": 1.0, "sigma_k": 0.01})

# Physical rate (preferred for physical modeling)
resolve_hparam({"peq": 10, "kappa": 0.2, "inc": 1.0, "lspot": 5,
                "tau_spot": 1.0,
                "nspot_rate": 0.5, "fspot": 0.1, "alpha_max": 0.05})
```

---

## Specificity-based detection

When multiple specs could match the keys you provide, the **most specific**
spec wins — the one with the largest `signature_keys`. This means you can
safely provide a superset of keys and the registry will always pick the right
spec.

For example, if you provide both `sigma_k` *and* `{nspot_rate, fspot,
alpha_max}`, the `physical_rate` spec wins (3 signature keys vs. 1), and
`sigma_k` is overwritten with the computed value.

Within the same specificity tier, **registration order** determines priority.
Use `priority="high"` when registering to prepend instead of append.

---

## Adding a custom envelope

To add a new envelope shape, define a resolve function and register an
`EnvelopeSpec`. The resolve function receives the raw hparam dict and must
return a dict that includes `tau_spot` (a scalar timescale for backward
compatibility with modules like `GPSolver` and PSD estimation).

### Example: Gaussian decay envelope

```python
from spotgp import EnvelopeSpec, register_envelope

def _resolve_gaussian(raw: dict) -> dict:
    tau_gauss = float(raw["tau_gauss"])
    return {
        "tau_gauss": tau_gauss,
        "tau_spot": tau_gauss,  # required: scalar timescale for GPSolver/PSD
    }

register_envelope(EnvelopeSpec(
    name="gaussian",
    signature_keys=frozenset({"tau_gauss"}),
    resolve=_resolve_gaussian,
    description="Gaussian decay: tau_gauss sets the 1/e timescale",
))
```

After registration, `resolve_hparam` accepts dicts with `tau_gauss`:

```python
hp = resolve_hparam({
    "peq": 10, "kappa": 0.2, "inc": 1.0, "lspot": 0,
    "tau_gauss": 3.0,
    "sigma_k": 0.01,
})

print(hp["tau_spot"])   # 3.0
print(hp["tau_gauss"])  # 3.0
```

### Requirements for the resolve function

- **Input:** the full raw hparam dict (read-only).
- **Output:** a dict of key-value pairs to merge into the resolved output.
- **Must include `"tau_spot"`:** a scalar timescale used by `GPSolver` for
  bandwidth estimation, PSD computation, and other modules that need a single
  representative spot lifetime.

---

## Adding a custom amplitude parameterization

Register an `AmplitudeSpec` with a `formula` callable that computes `sigma_k`
from the raw dict.

### Example: contrast-weighted amplitude

```python
import numpy as np
from spotgp.params import AmplitudeSpec, register_amplitude

register_amplitude(AmplitudeSpec(
    name="contrast_weighted",
    signature_keys=frozenset({"nspot_rate", "fspot", "alpha_max", "contrast"}),
    formula=lambda raw: (
        np.sqrt(float(raw["nspot_rate"]))
        * float(raw["contrast"])
        * float(raw["alpha_max"]) ** 2
    ),
    description="sigma_k weighted by an explicit spot contrast parameter",
))
```

This spec has 4 signature keys, so it is more specific than the built-in
`physical_rate` (3 keys). If you provide all four keys, the contrast-weighted
formula is used:

```python
hp = resolve_hparam({
    "peq": 10, "kappa": 0.2, "inc": 1.0, "lspot": 5,
    "tau_spot": 1.0,
    "nspot_rate": 0.5, "fspot": 0.1, "alpha_max": 0.05,
    "contrast": 0.8,
})

# sigma_k = sqrt(0.5) * 0.8 * 0.05^2 = 0.001414...
```

### Requirements for the formula callable

- **Input:** the full raw hparam dict.
- **Output:** a scalar `sigma_k` value (the kernel amplitude).

---

## Using custom parameters with `GPSolver`

Custom parameters integrate directly with the fitting pipeline. The key is
that `GPSolver` and `MCMCSampler` call `resolve_hparam` internally — once your
spec is registered, everything downstream works automatically.

```python
import numpy as np
from spotgp import (
    EnvelopeSpec, register_envelope,
    TimeSeriesData, SpotEvolutionModel,
    TrapezoidSymmetricEnvelope, VisibilityFunction,
    AnalyticKernel, GPSolver,
)

# 1. Register a custom envelope (at import time or in your script)
def _resolve_exp_decay(raw):
    return {"tau_spot": float(raw["tau_exp"])}

register_envelope(EnvelopeSpec(
    name="exponential_decay",
    signature_keys=frozenset({"tau_exp"}),
    resolve=_resolve_exp_decay,
    description="Simple exponential decay with timescale tau_exp",
))

# 2. Build the model using the envelope class that implements Gamma(t)
model = SpotEvolutionModel(
    envelope=TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=3.0),
    visibility=VisibilityFunction(peq=10.0, kappa=0.2, inc=np.pi/4),
    sigma_k=0.01,
)

# 3. Fit — resolve_hparam is called internally by GPSolver
data = TimeSeriesData(t, flux, flux_err)
gp = GPSolver(data, model, bounds=bounds).build_jax()
theta_map, result = gp.fit_map()
```

---

## EnvelopeSpec vs EnvelopeFunction

These serve different roles in the spotgp architecture:

| | `EnvelopeSpec` (params.py) | `EnvelopeFunction` (envelope.py) |
|---|---|---|
| **Purpose** | Validates and normalizes hyperparameter *dicts* | Defines the spot envelope *shape* (Gamma(t), R_Gamma, etc.) |
| **Used by** | `resolve_hparam()`, `GPSolver`, `MCMCSampler` | `AnalyticKernel`, `SpotEvolutionModel` |
| **What you define** | Which keys identify the envelope, how to extract `tau_spot` | The actual time-domain envelope function and its transforms |
| **When to use** | You want a new hparam key layout accepted by `resolve_hparam` | You want a new physical spot shape used in kernel computation |

For a fully custom envelope shape, you typically need both: an `EnvelopeFunction`
subclass that defines the physics, and an `EnvelopeSpec` that tells
`resolve_hparam` how to parse the new parameters.

---

## API reference

### `EnvelopeSpec`

```python
@dataclass(frozen=True)
class EnvelopeSpec:
    name: str                           # unique identifier
    signature_keys: frozenset[str]      # keys that trigger this spec
    resolve: Callable[[dict], dict]     # returns derived keys (must include tau_spot)
    description: str = ""               # shown in error messages
```

### `AmplitudeSpec`

```python
@dataclass(frozen=True)
class AmplitudeSpec:
    name: str                           # unique identifier
    signature_keys: frozenset[str]      # keys that trigger this spec
    formula: Callable[[dict], float]    # computes sigma_k
    description: str = ""               # shown in error messages
```

### `register_envelope(spec, priority="low")`

Add an `EnvelopeSpec` to the registry. Raises `ValueError` if a spec with the
same name is already registered.

### `register_amplitude(spec, priority="low")`

Add an `AmplitudeSpec` to the registry. Raises `ValueError` if a spec with the
same name is already registered.

### `resolve_hparam(raw) -> dict`

Validate and normalize a raw hyperparameter dict. Returns a new dict with all
original keys plus `tau_spot` and `sigma_k`. Raises `TypeError` if input is
not a dict, `ValueError` if required keys are missing or no spec matches.

### Constants

| Constant | Value |
|---|---|
| `BASE_REQUIRED_KEYS` | `frozenset({"peq", "kappa", "inc", "lspot"})` |
| `KERNEL_HPARAM_KEYS` | `("peq", "kappa", "inc", "lspot", "tau_spot", "sigma_k")` |
| `HPARAM_KEYS_WITH_NOISE` | `KERNEL_HPARAM_KEYS + ("sigma_n",)` |

---

## Error handling

`resolve_hparam` provides informative error messages. If no spec matches, it
lists all registered specs and their required keys:

```python
>>> resolve_hparam({"peq": 10, "kappa": 0.2, "inc": 1.0, "lspot": 5,
...                 "sigma_k": 0.01})
ValueError: No envelope spec matched the provided keys [...].
Registered envelopes:
  trapezoid_asymmetric: keys=['tau_dec', 'tau_em']  — Asymmetric trapezoid: ...
  skew_normal: keys=['n_sn', 'sigma_sn']  — Skew-normal: ...
  trapezoid_symmetric: keys=['tau_spot']  — Symmetric trapezoid: ...
```

Duplicate spec names are rejected at registration time:

```python
>>> register_envelope(EnvelopeSpec(
...     name="trapezoid_symmetric", ...))
ValueError: EnvelopeSpec 'trapezoid_symmetric' is already registered.
```
