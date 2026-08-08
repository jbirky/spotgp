"""
random_variables.py — Unified declaration of per-spot random variables.

Every spot in the generative model carries a set of random variables:
an emergence time ``t_ref``, a longitude ``phi_ref``, a latitude
``lam_ref``, and morphology parameters (lifetime, timescales,
amplitude).  The analytic kernel is the expectation of the single-spot
covariance over all of them.  This module makes that structure
explicit: a :class:`SpotRandomVariables` declaration names every
random variable and its distribution in one place, and each is routed
to the marginalization strategy that exploits its structure:

- ``t_ref`` ~ uniform — marginalized in closed form: the uniform
  emergence-time integral *is* the envelope autocorrelation R_Gamma.
  Uniformity is load-bearing (it makes the kernel stationary, which
  the Toeplitz/banded solver path relies on), so any other
  emergence-time distribution is rejected here rather than silently
  breaking stationarity.
- ``phi_ref`` ~ uniform — marginalized in closed form: uniform
  longitude kills the cross-harmonic terms and yields the ``|c_n|^2``
  structure.  Non-uniform longitudes (active longitudes) phase-lock
  the process and are likewise rejected.
- ``lam_ref`` — marginalized by the existing latitude quadrature;
  declare it on ``SpotEvolutionModel.latitude_distribution``.
- morphology parameters — marginalized by reparameterized Gauss
  quadrature: *fixed* base nodes pushed through hyperparameter-
  dependent transforms, so the marginalized kernel stays JAX-
  differentiable with respect to the fitted hyperparameters and the
  quadrature never recompiles.

Correlations between morphology parameters are expressed with
:class:`Derived` deterministic couplings, which cost nothing extra:
they reuse the latent variable's quadrature dimension.  The
Gnevyshev–Waldmeier relation (spot area proportional to lifetime) is
provided as :func:`gnevyshev_waldmeier`.

A morphology entry in the declaration dict may be:

- a plain ``float`` — the same fixed value for every spot (delta
  distribution, not fitted);
- a :class:`Hyper` — the same value for every spot, fitted as a
  hyperparameter;
- a :class:`Latent` subclass — distributed across spots, marginalized
  over one quadrature dimension, with its distribution's
  hyperparameters fitted;
- a :class:`Derived` (or bare callable) — a deterministic function of
  previously declared entries.

Example (Gnevyshev–Waldmeier)::

    rv = SpotRandomVariables({
        "T":        LogNormalLatent(mean=Hyper("Tbar", 8.0, (0.5, 50.0)),
                                    sigma=Hyper("sigma_T", 0.4, (0.01, 1.5))),
        "sigma0":   Hyper("sigma0", 0.01, (1e-4, 1.0)),
        "f":        0.2,
        "lspot":    Derived(lambda p: (1.0 - 2.0 * p["f"]) * p["T"]),
        "tau_spot": Derived(lambda p: p["f"] * p["T"]),
        "sigma_k":  Derived(lambda p: p["sigma0"] * p["T"] / p["Tbar"]),
    })
"""

import numpy as np
import jax.numpy as jnp

__all__ = [
    "Hyper",
    "Latent",
    "LogNormalLatent",
    "NormalLatent",
    "UniformLatent",
    "Derived",
    "UniformEmergence",
    "UniformLongitude",
    "SpotRandomVariables",
    "gnevyshev_waldmeier",
]


# Names owned by the star's geometry; a morphology declaration must not
# touch them (geometry-coupled joints do not factor out of the latitude
# quadrature and are not supported yet).
GEOMETRY_KEYS = ("peq", "kappa", "inc", "lat_min", "lat_max")


class Hyper:
    """
    A fitted population hyperparameter.

    Appears in the flat parameter vector of any term built from a
    declaration that references it.  The same instance may be shared
    between several specs; two *different* Hyper objects with the same
    name and different values are rejected at declaration time.

    Parameters
    ----------
    name : str
        Key reported to the solver (namespaced by the term's prefix).
    value : float
        Initial value.
    bounds : (float, float) or None
        Default prior bounds; ``None`` defers to user-supplied bounds.
    """

    def __init__(self, name, value, bounds=None):
        self.name = str(name)
        self.value = float(value)
        self.bounds = None if bounds is None else (float(bounds[0]),
                                                   float(bounds[1]))

    def __repr__(self):
        return f"Hyper({self.name!r}, {self.value}, bounds={self.bounds})"


class UniformEmergence:
    """
    Marker: ``t_ref`` ~ Uniform (Poisson emergence times).

    The only supported emergence-time distribution.  Its
    marginalization is analytic — the envelope autocorrelation
    R_Gamma(tau) — and its uniformity is what makes the kernel
    stationary.  A non-uniform emergence rate (e.g. an activity cycle)
    is a non-stationary feature, not a parameter distribution; model
    it as an explicit modulation term instead.
    """

    def __repr__(self):
        return "UniformEmergence()"


class UniformLongitude:
    """
    Marker: ``phi_ref`` ~ Uniform over [0, 2 pi).

    The only supported longitude distribution.  Uniform longitude is
    marginalized analytically into the harmonic ``|c_n|^2`` structure;
    non-uniform longitudes (active longitudes) phase-lock the process
    and make the ensemble kernel non-stationary.
    """

    def __repr__(self):
        return "UniformLongitude()"


class Derived:
    """
    A deterministic function of previously declared entries.

    ``fn`` receives the namespace dict mapping every earlier-declared
    name (morphology entries and hyperparameters) to its value —
    scalars for hypers/floats, ``(M,)`` arrays for latent and derived
    quantities — and must return a value built from ``jax.numpy``
    operations so gradients flow.  Entries must be declared before
    they are referenced (declaration order is evaluation order).
    """

    def __init__(self, fn):
        if not callable(fn):
            raise TypeError("Derived requires a callable")
        self.fn = fn

    def __repr__(self):
        return f"Derived({getattr(self.fn, '__name__', 'fn')})"


# ── Latent variables ────────────────────────────────────────────────────────

class Latent:
    """
    A per-spot random variable marginalized by fixed-node quadrature.

    Subclasses declare ``_slots`` (their parameter names, each backed
    by a :class:`Hyper` or a fixed float), a standard-space
    ``base_quadrature`` that does not depend on the hyperparameters,
    and a JAX-traceable ``transform`` from standard nodes to physical
    values.  This is the reparameterization trick: nodes and weights
    are static arrays, hyperparameters enter only through the
    transform, so the marginalized kernel is differentiable in the
    hypers and never triggers recompilation.
    """

    _slots = ()

    def __init__(self, n_quad):
        self.n_quad = int(n_quad)
        if self.n_quad < 1:
            raise ValueError("n_quad must be >= 1")

    @property
    def hypers(self):
        """Fitted Hyper objects among this latent's slots, in order."""
        return tuple(s for s in (getattr(self, "_" + n) for n in self._slots)
                     if isinstance(s, Hyper))

    def _slot_values(self, namespace):
        """Slot values (traced hyper entries or float constants)."""
        out = []
        for n in self._slots:
            spec = getattr(self, "_" + n)
            out.append(namespace[spec.name] if isinstance(spec, Hyper)
                       else spec)
        return out

    def base_quadrature(self):
        """Fixed standard-space nodes and weights, ``(u, w)`` (numpy).

        Weights sum to 1 (they carry the probability measure)."""
        raise NotImplementedError

    def transform(self, u, *slot_values):
        """JAX-traceable map from standard nodes to physical values."""
        raise NotImplementedError

    def sample_standard(self, n, rng):
        """Draw n standard-space samples (numpy) for simulation."""
        raise NotImplementedError

    def __repr__(self):
        slots = ", ".join(f"{n}={getattr(self, '_' + n)!r}"
                          for n in self._slots)
        return f"{type(self).__name__}({slots}, n_quad={self.n_quad})"


def _as_slot(value, what):
    """Validate a latent slot: Hyper passes through, numbers become floats."""
    if isinstance(value, Hyper):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        raise TypeError(
            f"{what} must be a Hyper or a number, got {type(value).__name__}")


class LogNormalLatent(Latent):
    """
    Log-normally distributed latent, parameterized by its *linear-space*
    mean: ``X = mean * exp(sigma * z - sigma^2 / 2)`` with z standard
    normal, so ``E[X] = mean`` exactly.  Marginalized on Gauss-Hermite
    nodes.

    Parameters
    ----------
    mean : Hyper or float
        E[X] (linear space).
    sigma : Hyper or float
        Standard deviation of log(X).
    n_quad : int
        Number of Gauss-Hermite nodes (default 16).
    """

    _slots = ("mean", "sigma")

    def __init__(self, mean, sigma, n_quad=16):
        super().__init__(n_quad)
        self._mean = _as_slot(mean, "mean")
        self._sigma = _as_slot(sigma, "sigma")

    def base_quadrature(self):
        u, w = np.polynomial.hermite.hermgauss(self.n_quad)
        return np.sqrt(2.0) * u, w / np.sqrt(np.pi)

    def transform(self, z, mean, sigma):
        return mean * jnp.exp(sigma * z - 0.5 * sigma ** 2)

    def sample_standard(self, n, rng):
        return rng.standard_normal(n)


class NormalLatent(Latent):
    """
    Normally distributed latent: ``X = mu + sigma * z``.  Marginalized
    on Gauss-Hermite nodes.  Prefer :class:`LogNormalLatent` for
    positive quantities (timescales, amplitudes) — a normal latent
    can produce negative nodes.
    """

    _slots = ("mu", "sigma")

    def __init__(self, mu, sigma, n_quad=16):
        super().__init__(n_quad)
        self._mu = _as_slot(mu, "mu")
        self._sigma = _as_slot(sigma, "sigma")

    def base_quadrature(self):
        u, w = np.polynomial.hermite.hermgauss(self.n_quad)
        return np.sqrt(2.0) * u, w / np.sqrt(np.pi)

    def transform(self, z, mu, sigma):
        return mu + sigma * z

    def sample_standard(self, n, rng):
        return rng.standard_normal(n)


class UniformLatent(Latent):
    """
    Uniformly distributed latent on [lo, hi]:
    ``X = lo + (hi - lo) * (u + 1) / 2`` on Gauss-Legendre nodes
    ``u`` in [-1, 1].
    """

    _slots = ("lo", "hi")

    def __init__(self, lo, hi, n_quad=16):
        super().__init__(n_quad)
        self._lo = _as_slot(lo, "lo")
        self._hi = _as_slot(hi, "hi")

    def base_quadrature(self):
        u, w = np.polynomial.legendre.leggauss(self.n_quad)
        return u, w / 2.0

    def transform(self, u, lo, hi):
        return lo + (hi - lo) * (u + 1.0) / 2.0

    def sample_standard(self, n, rng):
        return rng.uniform(-1.0, 1.0, n)


# ── The declaration ─────────────────────────────────────────────────────────

class SpotRandomVariables:
    """
    Declaration of every per-spot random variable in one place.

    Parameters
    ----------
    params : dict
        Ordered mapping of morphology names to specs (float, Hyper,
        Latent, Derived, or bare callable).  Evaluation follows
        declaration order, so entries must be declared before they are
        referenced by a ``Derived``.
    t_ref : UniformEmergence, optional
        Emergence-time distribution.  Only the uniform default is
        supported (see :class:`UniformEmergence`).
    phi_ref : UniformLongitude, optional
        Longitude distribution.  Only the uniform default is supported
        (see :class:`UniformLongitude`).
    lam_ref : None
        Latitude is marginalized by the latitude quadrature; declare
        it via ``SpotEvolutionModel.latitude_distribution``.  Anything
        other than None raises (correlated latitude joints are not
        supported yet).
    """

    def __init__(self, params, t_ref=None, phi_ref=None, lam_ref=None):
        if t_ref is None:
            t_ref = UniformEmergence()
        if not isinstance(t_ref, UniformEmergence):
            raise NotImplementedError(
                f"t_ref must be UniformEmergence, got {t_ref!r}.  A "
                "non-uniform emergence-time density makes the kernel "
                "non-stationary and would silently break the "
                "Toeplitz/banded solver path; model activity cycles as "
                "an explicit modulation term instead.")
        if phi_ref is None:
            phi_ref = UniformLongitude()
        if not isinstance(phi_ref, UniformLongitude):
            raise NotImplementedError(
                f"phi_ref must be UniformLongitude, got {phi_ref!r}.  "
                "Non-uniform longitudes (active longitudes) phase-lock "
                "the process and make the ensemble kernel "
                "non-stationary.")
        if lam_ref is not None:
            raise NotImplementedError(
                "lam_ref is marginalized by the latitude quadrature — "
                "declare it via SpotEvolutionModel.latitude_distribution.  "
                "Latitude distributions correlated with morphology "
                "parameters are not supported yet.")
        self.t_ref = t_ref
        self.phi_ref = phi_ref

        if not isinstance(params, dict) or not params:
            raise TypeError("params must be a non-empty dict of specs")

        # Normalize specs: bare callables become Derived, numbers floats.
        self._params = {}
        for name, spec in params.items():
            name = str(name)
            if name in GEOMETRY_KEYS:
                raise ValueError(
                    f"'{name}' is a geometry parameter; geometry-coupled "
                    "joint distributions do not factor out of the "
                    "latitude quadrature and are not supported.  Set "
                    "geometry on the SpotEvolutionModel instead.")
            if isinstance(spec, (Hyper, Latent, Derived)):
                self._params[name] = spec
            elif callable(spec):
                self._params[name] = Derived(spec)
            else:
                try:
                    self._params[name] = float(spec)
                except (TypeError, ValueError):
                    raise TypeError(
                        f"spec for '{name}' must be a float, Hyper, "
                        f"Latent, Derived, or callable, got "
                        f"{type(spec).__name__}")

        # Collect hypers in declaration order (params first, then each
        # latent's slots), deduplicating shared instances by name.
        seen = {}
        ordered = []
        def _add(h):
            prev = seen.get(h.name)
            if prev is None:
                seen[h.name] = h
                ordered.append(h)
            elif prev is not h and (prev.value != h.value
                                    or prev.bounds != h.bounds):
                raise ValueError(
                    f"two different Hyper('{h.name}') declarations with "
                    "conflicting value/bounds; share one instance instead")
        for name, spec in self._params.items():
            if isinstance(spec, Hyper):
                _add(spec)
            elif isinstance(spec, Latent):
                for h in spec.hypers:
                    _add(h)
        self._hypers = tuple(ordered)

        # One namespace: morphology names + hyper names must not collide
        # (a Hyper spec whose name matches its param name is the same
        # binding, not a collision).
        for name, spec in self._params.items():
            if name in seen and not (isinstance(spec, Hyper)
                                     and spec.name == name):
                raise ValueError(
                    f"name '{name}' is used both as a morphology "
                    "parameter and as a hyperparameter")

        # Tensor-product quadrature grid over the latents (static).
        latents = [(n, s) for n, s in self._params.items()
                   if isinstance(s, Latent)]
        self._latent_names = tuple(n for n, _ in latents)
        if latents:
            axes = [s.base_quadrature() for _, s in latents]
            grids = np.meshgrid(*[u for u, _ in axes], indexing="ij")
            self._U = {name: g.ravel()
                       for (name, _), g in zip(latents, grids)}
            wgrids = np.meshgrid(*[w for _, w in axes], indexing="ij")
            W = np.ones_like(wgrids[0])
            for wg in wgrids:
                W = W * wg
            self._weights = W.ravel()
        else:
            self._U = {}
            self._weights = np.array([1.0])

        # Dry run to fail fast on bad Derived references.
        try:
            self.resolve(self.hyper0)
        except KeyError as e:
            raise ValueError(
                f"Derived spec references undeclared name {e}; entries "
                "must be declared before they are referenced "
                "(declaration order is evaluation order)") from e

    # ── Hyperparameter layout ───────────────────────────────────────────

    @property
    def hyper_keys(self):
        """Fitted hyperparameter names, in declaration order."""
        return tuple(h.name for h in self._hypers)

    @property
    def hyper0(self):
        """Initial hyperparameter values aligned with ``hyper_keys``."""
        return np.array([h.value for h in self._hypers], dtype=np.float64)

    @property
    def hyper_bounds(self):
        """``{name: (lo, hi)}`` for hypers that declared bounds."""
        return {h.name: h.bounds for h in self._hypers
                if h.bounds is not None}

    @property
    def param_names(self):
        """Morphology names, in declaration (evaluation) order."""
        return tuple(self._params.keys())

    @property
    def n_nodes(self):
        """Total quadrature nodes M (product over the latents)."""
        return int(self._weights.size)

    # ── Evaluation ──────────────────────────────────────────────────────

    def _evaluate(self, latent_cols, hyper_values):
        """
        Shared evaluation core for quadrature and sampling.

        Parameters
        ----------
        latent_cols : dict
            ``{latent name: (M,) array}`` of standard-space columns.
        hyper_values : dict
            ``{hyper name: scalar}`` (traced or concrete).

        Returns
        -------
        namespace : dict
            Every hyper and morphology name mapped to its value.
        """
        ns = dict(hyper_values)
        for name, spec in self._params.items():
            if isinstance(spec, Latent):
                ns[name] = spec.transform(latent_cols[name],
                                          *spec._slot_values(ns))
            elif isinstance(spec, Hyper):
                ns[name] = ns[spec.name]
            elif isinstance(spec, Derived):
                ns[name] = spec.fn(ns)
            else:  # float
                ns[name] = spec
        return ns

    def resolve(self, phi):
        """
        JAX-traceable quadrature evaluation.

        Parameters
        ----------
        phi : array, shape (n_hyper,)
            Hyperparameter values aligned with ``hyper_keys``.

        Returns
        -------
        namespace : dict
            Every hyper and morphology name mapped to its value —
            scalars for hypers/floats, ``(M,)`` arrays for latent and
            derived quantities.
        weights : jnp.ndarray, shape (M,)
            Quadrature weights (sum to 1, independent of phi).
        """
        phi = jnp.asarray(phi)
        h = {name: phi[i] for i, name in enumerate(self.hyper_keys)}
        cols = {name: jnp.asarray(u) for name, u in self._U.items()}
        return self._evaluate(cols, h), jnp.asarray(self._weights)

    def column(self, namespace, name):
        """A namespace entry broadcast to a full ``(M,)`` node column."""
        return jnp.broadcast_to(jnp.asarray(namespace[name]),
                                (self.n_nodes,))

    def sample(self, n, phi=None, rng=None):
        """
        Draw n per-spot parameter sets — the simulation-side view of
        the same declaration (for LightcurveModel validation runs).

        Parameters
        ----------
        n : int
            Number of spots.
        phi : array or None
            Hyperparameter values (default: ``hyper0``).
        rng : np.random.Generator, optional

        Returns
        -------
        dict
            ``{name: (n,) np.ndarray}`` for every morphology name.
        """
        if rng is None:
            rng = np.random.default_rng()
        phi = self.hyper0 if phi is None else np.asarray(phi, dtype=float)
        h = {name: float(phi[i]) for i, name in enumerate(self.hyper_keys)}
        cols = {name: self._params[name].sample_standard(n, rng)
                for name in self._latent_names}
        ns = self._evaluate(cols, h)
        return {name: np.broadcast_to(np.asarray(ns[name]), (n,)).copy()
                for name in self.param_names}

    def __repr__(self):
        entries = ", ".join(f"{n}: {s!r}" for n, s in self._params.items())
        return (f"SpotRandomVariables({{{entries}}}, "
                f"n_nodes={self.n_nodes}, hypers={self.hyper_keys})")


# ── Gnevyshev–Waldmeier convenience ─────────────────────────────────────────

def gnevyshev_waldmeier(Tbar=8.0, sigma_T=0.4, sigma0=0.01, f=0.2,
                        n_quad=16, fit_f=False,
                        Tbar_bounds=(0.5, 50.0),
                        sigma_T_bounds=(0.01, 1.5),
                        sigma0_bounds=(1e-4, 1.0),
                        f_bounds=(0.01, 0.49)):
    """
    Gnevyshev–Waldmeier-coupled spot declaration for a symmetric
    trapezoid envelope.

    One latent — the spot lifetime ``T ~ LogNormal`` with ``E[T] =
    Tbar`` — drives everything: the envelope timescales scale with T
    at a fixed shape ratio f (``tau_spot = f T``, ``lspot = (1-2f) T``,
    so lifetime = lspot + 2 tau_spot = T), and the photometric
    amplitude scales with spot area, which the GW relation ties to
    lifetime (``A = W T``): ``sigma_k = sigma0 * T / Tbar``.  The GW
    constant W cancels in the sigma0 normalization — only the
    T-dependence of the amplitude is observable in a kernel — so
    ``sigma0`` keeps its interpretation as the amplitude of a
    mean-lifetime spot.

    Because amplitude weights the mixture by T^2, the marginalized
    kernel is dominated by the long-lived tail of the distribution —
    the physical effect an independent-parameter model cannot express.

    Parameters
    ----------
    Tbar : float
        Mean spot lifetime [days] (fitted hyper).
    sigma_T : float
        Log-space lifetime scatter (fitted hyper).
    sigma0 : float
        Amplitude of a mean-lifetime spot (fitted hyper).
    f : float
        Rise/decay fraction of the lifetime, 0 < f < 1/2.  Fixed by
        default; fitted when ``fit_f=True``.
    n_quad : int
        Gauss-Hermite nodes for the lifetime latent.
    fit_f : bool
        Fit f as a hyperparameter.
    *_bounds : (float, float)
        Default prior bounds for the corresponding hyper.

    Returns
    -------
    SpotRandomVariables
    """
    if not fit_f and not 0.0 < f < 0.5:
        raise ValueError(f"f must be in (0, 1/2), got {f}")
    return SpotRandomVariables({
        "T": LogNormalLatent(mean=Hyper("Tbar", Tbar, Tbar_bounds),
                             sigma=Hyper("sigma_T", sigma_T, sigma_T_bounds),
                             n_quad=n_quad),
        "sigma0": Hyper("sigma0", sigma0, sigma0_bounds),
        "f": Hyper("f", f, f_bounds) if fit_f else float(f),
        "lspot": Derived(lambda p: (1.0 - 2.0 * p["f"]) * p["T"]),
        "tau_spot": Derived(lambda p: p["f"] * p["T"]),
        "sigma_k": Derived(lambda p: p["sigma0"] * p["T"] / p["Tbar"]),
    })
