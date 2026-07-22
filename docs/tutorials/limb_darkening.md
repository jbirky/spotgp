# Limb-Darkened Visibility

The base `VisibilityFunction` treats the stellar disk as **uniformly bright**: a
spot's flux deficit is proportional to its projected area alone. Real stars are
limb darkened — the disk is brighter at the center than at the edge — so a spot
near disk center blocks more light than the same spot near the limb.

`LimbDarkenedVisibilityFunction` accounts for this by weighting the deficit with
the local specific intensity.

---

## The physics

Write $\mu = \cos\beta$ for the cosine of the angle between the spot normal and
the line of sight,

$$\mu = \cos I \sin \Phi + \sin I \cos \Phi \cos \Lambda(t)$$

where $I$ is the stellar inclination, $\Phi$ the spot latitude, and $\Lambda(t)$
the rotational longitude. In the small-spot limit the base class uses

$$V(\mu) = \max(\mu,\, 0)$$

With limb darkening the deficit is weighted by the intensity at the spot and
renormalized by the disk-integrated flux:

$$V(\mu) = \frac{\mu\, I(\mu)}{F}, \qquad F = \int_0^1 2\mu\, I(\mu)\, d\mu$$

The normalization $F$ is what makes the limb-darkened visibility reduce
**exactly** to the uniform-disk case when $I(\mu) \equiv 1$.

```{admonition} Why the Fourier coefficients are computed numerically
:class: note

The analytic kernel needs the squared Fourier coefficients $|c_n|^2$ of
$V$ over one rotation. For a uniform disk $V$ is a clipped cosine and the
$c_n$ have closed forms (`_cn_general_jax`). For a general $I(\mu)$ there is
no such closed form, so `LimbDarkenedVisibilityFunction` evaluates $V$ on a
longitude grid and takes the DFT — the same strategy
`FullGeometryVisibilityFunction` uses. The grid size is set by `n_lon`.
```

---

## Basic usage

Drop it in wherever you would use `VisibilityFunction`:

```python
import numpy as np
from spotgp import (SpotEvolutionModel, TrapezoidSymmetricEnvelope,
                    LimbDarkenedVisibilityFunction, AnalyticKernel)

vis = LimbDarkenedVisibilityFunction(
    peq=4.0, kappa=0.0, inc=np.pi / 3,
    u=(0.4, 0.2),          # quadratic limb-darkening coefficients
)

model = SpotEvolutionModel(
    envelope=TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=2.0),
    visibility=vis,
    sigma_k=0.01,
)

kernel = AnalyticKernel(model, n_harmonics=4)
K = kernel.kernel(np.linspace(0, 20, 200))
```

The rotation parameters (`peq`, `kappa`, `inc`) behave exactly as in the base
class — the limb-darkening coefficients are *fixed* properties of the star, not
fitted GP hyperparameters, so `param_keys` is unchanged:

```python
>>> vis.param_keys
('peq', 'kappa', 'inc')
```

---

## Intensity laws

Two laws are supported via the `law` argument.

### Quadratic (default)

$$\frac{I(\mu)}{I(1)} = 1 - u_1(1-\mu) - u_2(1-\mu)^2,
\qquad F = 1 - \frac{u_1}{3} - \frac{u_2}{6}$$

```python
vis = LimbDarkenedVisibilityFunction(4.0, 0.0, np.pi/3, u=(0.4, 0.2))
```

### Claret four-coefficient

$$\frac{I(\mu)}{I(1)} = 1 - \sum_{k=1}^{4} c_k \left(1 - \mu^{k/2}\right),
\qquad F = 1 - \sum_{k=1}^{4} \frac{k\, c_k}{k+4}$$

This matches the coefficient convention already used by
`LightcurveModel.limbc`, so you can share coefficients between the forward
model and the GP kernel:

```python
from spotgp import LightcurveModel

lc = LightcurveModel(limb_darkening=True)
vis = LimbDarkenedVisibilityFunction(
    4.0, 0.0, np.pi/3, u=tuple(lc.limbc), law="claret")
```

Coefficients for a given star are usually looked up from tables (e.g. Claret &
Bloemen 2011) by $T_{\rm eff}$, $\log g$, and bandpass.

---

## Effect on the harmonics

Limb darkening sharpens $V(\Lambda)$, which pushes power into **higher
rotational harmonics**. At $I = 60°$, $\Phi = 0.3$ rad:

| case | $F$ | $\lvert c_0\rvert^2$ | $\lvert c_1\rvert^2$ | $\lvert c_2\rvert^2$ |
|---|---|---|---|---|
| none, `u=(0, 0)`     | 1.0000 | 0.116584 | 0.064322 | 0.006992 |
| solar-ish `(0.3, 0.2)` | 0.8667 | 0.127887 | 0.076111 | 0.012379 |
| strong `(0.6, 0.1)`  | 0.7833 | 0.136269 | 0.084882 | 0.017084 |
| Claret 4-coeff       | 0.8294 | 0.131533 | 0.079985 | 0.014370 |

$|c_2|^2$ roughly **doubles** under solar-like limb darkening. The effect is
starkest for harmonics that vanish identically on a uniform disk — at
$I = 90°$, $\Phi = 0.2$:

| $u$ | $\lvert c_3\rvert^2$ |
|---|---|
| $(0, 0)$     | $5.7\times10^{-32}$ (zero) |
| $(0.3, 0.2)$ | $3.7\times10^{-4}$ |
| $(0.6, 0.1)$ | $1.2\times10^{-3}$ |

```{warning}
Because limb darkening populates higher harmonics, raise `n_harmonics` on the
kernel. The default of 3 is tuned for the uniform-disk case; 4–6 is a safer
starting point with limb darkening. Check convergence by increasing it until
the kernel stops changing.
```

---

## Fitting

Nothing special is required — the limb-darkened coefficients participate in the
GP log-likelihood and its gradient, including the gradient with respect to
`inc`:

```python
from spotgp import GPSolver, TimeSeriesData

data = TimeSeriesData(t, flux, flux_err)
bounds = {"peq": (2, 8), "kappa": (0, 0.4), "inc": (0.3, 1.5),
          "lspot": (1, 10), "tau_spot": (1, 6), "log_sigma_k": (-4, -1)}

gp = GPSolver(data, model, bounds=bounds, n_harmonics=4).build_jax()
theta_map, result = gp.fit_map(nopt=5)
```

`inc` is re-evaluated inside the JIT-compiled likelihood, so the DFT is
recomputed at every trial inclination. This costs more than the closed-form
coefficients — see {ref}`Performance <limb-darkening-performance>`.

The model round-trips through HDF5 with its coefficients intact:

```python
gp.save("star.h5")
gp2 = GPSolver.load("star.h5")
gp2.spot_model.visibility.u      # (0.4, 0.2)
gp2.spot_model.visibility.law    # 'quadratic'
```

---

## Verifying against the uniform disk

Setting the coefficients to zero must reproduce the analytic base class. This
is a useful sanity check after changing `n_lon`:

```python
from spotgp import VisibilityFunction

plain  = VisibilityFunction(4.0, 0.0, np.pi/3)
ld_off = LimbDarkenedVisibilityFunction(
    4.0, 0.0, np.pi/3, u=(0.0, 0.0), n_lon=4096)

np.testing.assert_allclose(
    ld_off.cn_squared(0.3, 3), plain.cn_squared(0.3, 3), atol=1e-6)
```

---

(limb-darkening-performance)=
## Performance

`n_lon` controls the DFT grid. The default of 512 gives $|c_n|^2$ converged to
about $10^{-7}$; 4096 is essentially exact but 8× the work per latitude node.
Because the DFT runs at every latitude quadrature node and (during fitting) at
every likelihood evaluation, cost scales as `n_lat × n_lon`.

If limb-darkening coefficients are fixed and you are not fitting `inc`, the
coefficients are constant across the fit — but they are still recomputed each
call. For large runs, reduce `n_lat` or `n_lon` to whatever your convergence
check supports.

---

## Writing your own visibility function

`LimbDarkenedVisibilityFunction` is a worked example of the general extension
mechanism. The important detail:

```{warning}
Overriding `cn_squared()` alone is **not** enough to affect the GP fit.
`AnalyticKernel.kernel()` and the `GPSolver` log-likelihood evaluate the
latitude-averaged kernel inside a JIT-compiled, vmapped routine
(`_kernel_eval`) that cannot call an arbitrary Python method. Without the hook
below, a subclass changes `compute_psd()` and `kernel_single_latitude()` while
the kernel and likelihood silently keep using the built-in analytic $c_n$.
```

To participate fully, define a **JAX-traceable** `cn_sq_jax` method:

```python
class MyVisibility(VisibilityFunction):

    def cn_squared(self, phi, n_harmonics=3):
        """Convenience path: PSD, single-latitude kernel, plotting."""
        return self._compute(self.inc, phi, n_harmonics)

    def cn_sq_jax(self, theta_vis, phi, n_harmonics=3):
        """JAX-traceable path: kernel(), log-likelihood, gradients.

        theta_vis is [peq, kappa, inc] — the leading entries of theta_arr,
        matching param_keys. Read inc from theta_vis[2], *not* self.inc,
        so gradients reach it.
        """
        return self._compute(theta_vis[2], phi, n_harmonics)
```

`SpotEvolutionModel.get_cn_sq_func(n_harmonics)` detects `cn_sq_jax` and wraps
it into the per-latitude-grid form `_kernel_eval` expects. Classes without it
(including the built-in `VisibilityFunction` and `EdgeOnVisibilityFunction`)
return `None` and keep the faster inline analytic path — so this costs nothing
when unused.

```{admonition} Gradient safety
:class: tip

If your intensity law has infinite slope anywhere (the Claret law's
$\mu^{k/2}$ terms at $\mu = 0$), clip to a strictly *positive* floor before
evaluating it. `jnp.where` propagates gradients through **both** branches, so
a floor of exactly `0.0` produces `inf * 0 = NaN` in reverse-mode autodiff
even though the forward value looks correct.
```

---

## See also

- [Custom Visibility Functions](custom_visibility_function.ipynb) — exact
  projected-area geometry for large spots
- [Custom Envelope Functions](custom_envelope_gaussian.ipynb) — the analogous
  `r_gamma_jax` hook for spot evolution
