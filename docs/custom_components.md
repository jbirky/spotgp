# Defining Custom Model Components

The starspot kernel factorizes into three independent physical ingredients: the
envelope autocorrelation $R_\Gamma(\tau)$, the squared Fourier coefficients
$|c_n(\Phi)|^2$ of the visibility function, and the spot latitude distribution
$p(\Phi)$. The `spotgp` implementation mirrors this factorization: each
ingredient is represented by a base class — `EnvelopeFunction`,
`VisibilityFunction`, and `LatitudeDistributionFunction` — whose default
implementations correspond to the assumptions adopted in the companion
derivation paper (trapezoidal envelope, uniformly bright disk in the small-spot
limit, and uniform latitude distribution).

Any of the three can be replaced by a user-defined subclass without modifying
the kernel evaluation, the latitude quadrature, the JIT-compiled likelihood, or
the samplers, and multiple kernels can be summed into composite models. This
document specifies the extension contract for each component. Complete worked
examples are provided in the [online tutorials](https://spotgp.readthedocs.io).

## Custom envelope functions

A custom spot evolution profile is defined by subclassing `EnvelopeFunction`.
Only two members are required:

- the property `tau_spot`, a characteristic timescale used to size the numerical
  grids and the kernel bandwidth;
- the profile $\Gamma(t)$ itself, written with `jax.numpy` operations so it
  remains differentiable.

Every other quantity the kernel needs — the Fourier transform
$\hat{\Gamma}(\omega)$, the autocorrelation $R_\Gamma(\tau)$, and the
compact-support bound used by the banded solver — has a default implementation
that falls back to FFT-based numerical evaluation of $\Gamma(t)$, so a minimal
subclass is immediately usable. When closed forms are known, overriding
`Gamma_hat` and/or `R_Gamma` improves both speed and accuracy; the built-in
`check_functions()` method compares any analytic override against the FFT
baseline and reports the error.

As an example, a Gaussian envelope $\Gamma(t) = e^{-t^2/2\sigma^2}$ admits closed
forms for both quantities,

$$
\hat{\Gamma}(\omega) = \sigma\sqrt{2\pi}\, e^{-\omega^2\sigma^2/2},
\qquad
R_\Gamma(\tau) = \sigma\sqrt{\pi}\, e^{-\tau^2/4\sigma^2},
$$

and the complete implementation is:

```python
from spotgp import EnvelopeFunction, VisibilityFunction, SpotEvolutionModel
import jax.numpy as jnp

class GaussianEnvelope(EnvelopeFunction):
    """Gaussian spot envelope: Gamma(t) = exp(-t^2 / 2 sigma^2)."""

    def __init__(self, sigma):
        self._sigma = float(sigma)

    @property
    def tau_spot(self):          # characteristic timescale [days]
        return self._sigma

    @property
    def param_dict(self):
        return {"sigma": self._sigma}

    def Gamma(self, t):          # required
        return jnp.exp(-0.5 * (t / self._sigma) ** 2)

    def Gamma_hat(self, omega):  # optional closed-form override
        return self._sigma * jnp.sqrt(2 * jnp.pi) * jnp.exp(-0.5 * (omega * self._sigma) ** 2)

    def R_Gamma(self, lag):      # optional closed-form override
        return self._sigma * jnp.sqrt(jnp.pi) * jnp.exp(-0.25 * (lag / self._sigma) ** 2)

model = SpotEvolutionModel(
    envelope=GaussianEnvelope(sigma=5.0),
    visibility=VisibilityFunction(peq=10.0, kappa=0.3, inc=jnp.pi / 3),
    sigma_k=0.01,
)
```

The finished envelope plugs into `SpotEvolutionModel` and `AnalyticKernel`
exactly like the built-in trapezoidal, exponential, and skewed-Gaussian
envelopes.

## Custom visibility functions and limb darkening

The base `VisibilityFunction` implements the small-spot limit
$\mathcal{V}(t) = \max\{\cos\beta(t),\,0\}$, for which the Fourier coefficients
$c_n(I,\Phi)$ have closed forms derived in the companion paper. A subclass
replaces this physics by overriding `cn_squared(phi, n_harmonics)`, which
returns the squared coefficients $|c_n(\Phi)|^2$ entering the latitude-averaged
kernel. When the modified visibility profile admits no closed form, the
practical strategy is numerical: evaluate $\mathcal{V}$ on a longitude grid
covering one rotation and take its discrete Fourier transform.

```python
class MyVisibility(VisibilityFunction):

    def _compute(self, inc, phi, n_harmonics):
        lon = jnp.linspace(0.0, 2 * jnp.pi, self.n_lon, endpoint=False)
        V = ...                # custom visibility profile over one rotation
        cn = jnp.abs(jnp.fft.rfft(V) / self.n_lon)
        return cn[:n_harmonics + 1] ** 2

    def cn_squared(self, phi, n_harmonics=3):
        # convenience path: PSD, single-latitude kernel, plotting
        return self._compute(self.inc, phi, n_harmonics)

    def cn_sq_jax(self, theta_vis, phi, n_harmonics=3):
        # JAX-traceable path: kernel(), likelihood, gradients.
        # Read inc from theta_vis[2], not self.inc, so gradients reach it.
        return self._compute(theta_vis[2], phi, n_harmonics)
```

One implementation detail is important for fitting. `AnalyticKernel.kernel()`
and the `GPSolver` log-likelihood evaluate the latitude-averaged kernel inside a
JIT-compiled, vectorized routine that cannot call an arbitrary Python method, so
**overriding `cn_squared` alone affects only the PSD and single-latitude
paths.** To participate in the likelihood — including the gradient with respect
to $I$ — the subclass must additionally define the JAX-traceable hook
`cn_sq_jax(theta_vis, phi, n_harmonics)` shown above, which reads the
inclination from the parameter vector rather than from the instance.
`SpotEvolutionModel` detects this hook automatically; classes without it retain
the faster closed-form path.

Two built-in subclasses illustrate the mechanism. `FullGeometryVisibilityFunction`
implements the exact piecewise projected area, removing the small-spot
approximation for large spots. `LimbDarkenedVisibilityFunction` drops the
uniform-disk assumption: writing $\mu = \cos\beta$, the flux deficit of a spot is
weighted by the local specific intensity and normalized by the disk-integrated
flux,

$$
\mathcal{V}(\mu) = \frac{\mu\, I(\mu)}{F},
\qquad
F = \int_0^1 2\mu\, I(\mu)\, d\mu,
$$

which reduces exactly to the base class when $I(\mu) \equiv 1$. Both the
quadratic law, $I(\mu)/I(1) = 1 - u_1(1-\mu) - u_2(1-\mu)^2$ with
$F = 1 - u_1/3 - u_2/6$, and the four-coefficient nonlinear law
([Claret 2011](https://ui.adsabs.harvard.edu/abs/2011A%26A...529A..75C)) are
supported:

```python
from spotgp import LimbDarkenedVisibilityFunction, AnalyticKernel

vis = LimbDarkenedVisibilityFunction(
    peq=4.0, kappa=0.0, inc=jnp.pi / 3,
    u=(0.4, 0.2),       # quadratic law; law="claret" for 4 coefficients
)
model = SpotEvolutionModel(envelope=envelope, visibility=vis, sigma_k=0.01)
kernel = AnalyticKernel(model, n_harmonics=6)
```

The limb-darkening coefficients are fixed stellar properties (looked up by
$T_{\rm eff}$, $\log g$, and bandpass), not fitted hyperparameters, so the
parameter vector is unchanged. Because limb darkening sharpens
$\mathcal{V}(\Lambda)$, it redistributes power toward higher rotational
harmonics — coefficients that vanish identically for a uniform disk acquire real
power — so the harmonic truncation should be raised. We find $N = 4$–$6$ is a
safe starting point, checked by increasing $N$ until the kernel converges.

## Custom latitude distributions

The latitude distribution $p(\Phi)$ plays two roles in the pipeline: it weights
the latitude integral in the analytic kernel, and it sets the sampling bounds for
spot placement when generating synthetic lightcurves with `LightcurveModel`. The
default `LatitudeDistributionFunction` is uniform over $\Phi \in [-\pi/2,
\pi/2]$. A subclass overrides the property `lat_range`, which bounds the
quadrature grid, and `__call__(phi)`, which returns the unnormalized density —
the kernel normalizes internally. For example, Gaussian activity belts of width
$\sigma_\Phi$ centered on a latitude $\Phi_0$:

```python
from spotgp import LatitudeDistributionFunction
import numpy as np

class GaussianLatitude(LatitudeDistributionFunction):
    """Gaussian activity belt centered on `center` [rad]."""

    def __init__(self, sigma, center=0.0):
        self._sigma, self._center = float(sigma), float(center)

    @property
    def lat_range(self):      # bounds for quadrature and spot placement
        return (self._center - 3 * self._sigma, self._center + 3 * self._sigma)

    def __call__(self, phi):  # unnormalized p(phi)
        return float(np.exp(-0.5 * ((phi - self._center) / self._sigma) ** 2))

model = SpotEvolutionModel(
    envelope=envelope, visibility=visibility, sigma_k=0.01,
    latitude_distribution=GaussianLatitude(sigma=np.deg2rad(20.0)),
)
```

Equatorial bands, polar caps, or empirically measured butterfly-diagram
distributions follow the same pattern. The choice of $p(\Phi)$ matters most in
the presence of differential rotation: it determines which rotation frequencies
$\omega_0(\Phi)$ contribute to the kernel, and therefore how quickly the periodic
component dephases and how the harmonic amplitudes are weighted.

## Composite kernels

Several additive signal components — for example two spot populations with
different lifetimes plus a granulation-like noise process — can be modeled
simultaneously by summing stationary kernels,
$\mathcal{K}_{\rm total}(\tau) = \sum_m \mathcal{K}_m(\tau; \theta_m)$. A sum of
stationary kernels is itself stationary, so a composite kernel plugs into
`GPSolver` exactly like a single spot model and the banded solver is preserved
(the bandwidth is set by the widest term). Terms are composed with `KernelSum`;
each term owns a contiguous slice of the flat parameter vector, with parameter
names namespaced by a user-chosen prefix, and a `log_` marker on a parameter
name samples it in $\log_{10}$ space:

```python
from spotgp import GPSolver, KernelSum, SpotTerm, SHOTerm

kernel = KernelSum(
    SpotTerm(hparam_short, prefix="short"),          # rapidly evolving spots
    SpotTerm(hparam_long,  prefix="long"),           # long-lived spots
    SHOTerm(S0=2e-6, Q=0.7, w0=3.1, prefix="gran"),  # quasi-periodic noise floor
)

bounds = {"short.peq": (8.0, 12.0), "short.log_sigma_k": (-3.0, -1.0),
          "long.peq":  (8.0, 12.0), "long.log_sigma_k": (-3.5, -1.5),
          "gran.S0": (1e-8, 1e-4), "gran.Q": (0.3, 5.0), "gran.w0": (1.0, 10.0)}
gp = GPSolver(x, y, yerr, kernel, bounds=bounds).build_jax()
```

Available terms include `SpotTerm` (wrapping any `SpotEvolutionModel`, including
custom envelopes, visibilities, and latitude distributions defined above), the
`celerite`-style `SHOTerm`, and `Matern32Term`; pure white noise is handled more
cheaply by the solver's diagonal $\sigma_n$ term.

When multiple spot populations live on the same star, the rotation and viewing
geometry should be shared rather than duplicated. `SharedVisibilitySpotSum`
factorizes the sum as

$$
\mathcal{K}(\tau) = V(\tau) \sum_i \sigma_{k,i}^2\, R_{\Gamma,i}(\tau),
$$

where $V(\tau)$ is the latitude-averaged rotational term, so that $P_{\rm eq}$,
$\kappa$, and $I$ appear once in the parameter vector and the latitude
quadrature — the dominant cost — is evaluated once per kernel call rather than
once per population.

We caution that additive components can trade off against one another: a
low-quality-factor `SHOTerm` can mimic a short-lived spot population, and two
spot populations differing only in amplitude are not identifiable at all.
Informative bounds and priors become more important as components are added, and
posterior sampling is the honest way to expose the resulting degeneracies.
