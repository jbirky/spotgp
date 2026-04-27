"""
celerite.py — O(N) celerite GP solver with JAX acceleration.

Implements the semi-separable matrix algorithm from Foreman-Mackey et al.
(2017, AJ, 154, 220) for scalable Gaussian Process regression.  The kernel
is a mixture of exponentially-damped sinusoids, enabling O(N J^2)
factorization where J is the number of semi-separable terms (typically 2-8).

Kernel terms
------------
RealTerm, ComplexTerm, SHOTerm, RotationTerm, Matern32Term

Solver
------
CeleriteGPSolver — drop-in O(N) solver with an interface matching GPSolver.
"""

import os
import time as _time

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", os.environ.get("JAX_PLATFORMS", "cpu"))

import jax.numpy as jnp
import numpy as np

__all__ = [
    "RealTerm", "ComplexTerm", "SHOTerm", "RotationTerm", "Matern32Term",
    "TermSum", "CeleriteGPSolver",
]


# =====================================================================
# Celerite solver primitives (pure JAX, module-level for JIT tracing)
# =====================================================================

def _celerite_factor(t, c, a, U, V):
    """O(N J^2) semi-separable Cholesky factorization.

    Computes K = L diag(d) L^T where L is unit lower triangular with
    semi-separable off-diagonal structure.

    Parameters
    ----------
    t : (N,) sorted observation times
    c : (J,) exponential decay rates
    a : (N,) diagonal (kernel at zero lag + noise)
    U : (N, J) left semi-separable factor
    V : (N, J) right semi-separable factor

    Returns
    -------
    d : (N,) diagonal of D
    W : (N, J) factor such that L = I + tril(U W^T)
    """
    N = t.shape[0]
    J = U.shape[1]

    d0 = a[0]
    W0 = V[0] / d0
    S0 = jnp.zeros((J, J))

    def step(carry, n):
        d_prev, W_prev, S = carry
        p = jnp.exp(-c * (t[n] - t[n - 1]))
        S = jnp.outer(p, p) * (S + d_prev * jnp.outer(W_prev, W_prev))
        d_n = a[n] - U[n] @ S @ U[n]
        W_n = (V[n] - S @ U[n]) / d_n
        return (d_n, W_n, S), (d_n, W_n)

    _, (d_rest, W_rest) = jax.lax.scan(step, (d0, W0, S0), jnp.arange(1, N))
    d = jnp.concatenate([jnp.array([d0]), d_rest])
    W = jnp.concatenate([W0[None, :], W_rest])
    return d, W


def _celerite_solve_lower(t, c, U, W, y):
    """Forward solve L z = y where L is the unit lower triangular factor.

    Parameters
    ----------
    t : (N,)
    c : (J,)
    U : (N, J)
    W : (N, J) from factorization
    y : (N,) right-hand side

    Returns
    -------
    z : (N,)
    """
    J = U.shape[1]
    z0 = y[0]
    F0 = jnp.zeros(J)

    def step(carry, n):
        z_prev, F = carry
        p = jnp.exp(-c * (t[n] - t[n - 1]))
        F = p * (F + W[n - 1] * z_prev)
        z_n = y[n] - U[n] @ F
        return (z_n, F), z_n

    _, z_rest = jax.lax.scan(step, (z0, F0), jnp.arange(1, t.shape[0]))
    return jnp.concatenate([jnp.array([z0]), z_rest])


def _celerite_solve_upper(t, c, U, W, y):
    """Backward solve L^T z = y.

    Parameters
    ----------
    t : (N,)
    c : (J,)
    U : (N, J)
    W : (N, J) from factorization
    y : (N,) right-hand side

    Returns
    -------
    z : (N,)
    """
    N = t.shape[0]
    J = U.shape[1]
    z_last = y[N - 1]
    F0 = jnp.zeros(J)

    def step(carry, n):
        z_next, F = carry
        p = jnp.exp(-c * (t[n + 1] - t[n]))
        F = p * (F + U[n + 1] * z_next)
        z_n = y[n] - W[n] @ F
        return (z_n, F), z_n

    _, z_rest = jax.lax.scan(step, (z_last, F0), jnp.arange(N - 2, -1, -1))
    return jnp.concatenate([z_rest[::-1], jnp.array([z_last])])


def _celerite_solve_lower_matrix(t, c, U, W, Y):
    """Forward solve L Z = Y with matrix right-hand side.

    Parameters
    ----------
    t : (N,)
    c : (J,)
    U : (N, J)
    W : (N, J)
    Y : (N, M)

    Returns
    -------
    Z : (N, M)
    """
    J = U.shape[1]
    z0 = Y[0]
    F0 = jnp.zeros((J, Y.shape[1]))

    def step(carry, n):
        z_prev, F = carry
        p = jnp.exp(-c * (t[n] - t[n - 1]))
        F = p[:, None] * (F + W[n - 1, :, None] * z_prev[None, :])
        z_n = Y[n] - U[n] @ F
        return (z_n, F), z_n

    _, z_rest = jax.lax.scan(step, (z0, F0), jnp.arange(1, t.shape[0]))
    return jnp.concatenate([z0[None, :], z_rest])


def _celerite_dot_tril(t, c, U, W, d, z):
    """Compute L diag(sqrt(d)) z for prior sampling.

    Parameters
    ----------
    t : (N,)
    c, U, W : from factorization
    d : (N,) diagonal from factorization
    z : (N,) standard normal draw

    Returns
    -------
    y : (N,) a draw from N(0, K)  (caller adds the mean)
    """
    x = jnp.sqrt(d) * z
    J = U.shape[1]
    y0 = x[0]
    F0 = jnp.zeros(J)

    def step(carry, n):
        F = carry
        p = jnp.exp(-c * (t[n] - t[n - 1]))
        F = p * (F + W[n - 1] * x[n - 1])
        y_n = x[n] + U[n] @ F
        return F, y_n

    _, y_rest = jax.lax.scan(step, F0, jnp.arange(1, t.shape[0]))
    return jnp.concatenate([jnp.array([y0]), y_rest])


def _default_log_prior(theta_arr, bounds):
    """Soft uniform prior within bounds (sigmoid barriers)."""
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    k = 100.0
    log_lo = jnp.sum(jax.nn.log_sigmoid(k * (theta_arr - lo)))
    log_hi = jnp.sum(jax.nn.log_sigmoid(k * (hi - theta_arr)))
    log_vol = jnp.sum(jnp.log(hi - lo))
    return log_lo + log_hi - log_vol


# =====================================================================
# Term classes
# =====================================================================

class CeleriteTerm:
    """Base class for celerite kernel terms.

    Subclasses implement ``get_coefficients``, ``get_params``,
    ``_build_matrices_fn``, and ``_build_eval_fn``.
    """

    param_names = ()

    @property
    def J(self):
        raise NotImplementedError

    @property
    def n_params(self):
        return len(self.param_names)

    @property
    def default_bounds(self):
        raise NotImplementedError

    def get_coefficients(self):
        """Return ``(ar, cr, ac, bc, cc, dc)`` as numpy arrays."""
        raise NotImplementedError

    def get_params(self):
        """Return a dict mapping param_names to current values."""
        raise NotImplementedError

    def get_value(self, tau):
        """Evaluate kernel at time lags *tau* (numpy)."""
        tau = np.abs(np.asarray(tau, dtype=float))
        ar, cr, ac, bc, cc, dc = self.get_coefficients()
        k = np.zeros_like(tau)
        for j in range(len(ar)):
            k += ar[j] * np.exp(-cr[j] * tau)
        for j in range(len(ac)):
            k += np.exp(-cc[j] * tau) * (
                ac[j] * np.cos(dc[j] * tau) + bc[j] * np.sin(dc[j] * tau))
        return k

    def get_psd(self, omega):
        """Power spectral density at angular frequencies *omega* (numpy)."""
        omega = np.asarray(omega, dtype=float)
        ar, cr, ac, bc, cc, dc = self.get_coefficients()
        w2 = omega ** 2
        psd = np.zeros_like(omega)
        for j in range(len(ar)):
            psd += ar[j] * cr[j] / (cr[j] ** 2 + w2)
        for j in range(len(ac)):
            d2pc2 = cc[j] ** 2 + dc[j] ** 2
            psd += (
                (ac[j] * cc[j] + bc[j] * dc[j]) * d2pc2
                + (ac[j] * cc[j] - bc[j] * dc[j]) * w2
            ) / (w2 ** 2 + 2 * (cc[j] ** 2 - dc[j] ** 2) * w2 + d2pc2 ** 2)
        return psd * np.sqrt(2.0 / np.pi)

    def _build_matrices_fn(self):
        """Return JAX-traceable ``(theta, t) -> (c, a_diag, U, V)``."""
        raise NotImplementedError

    def _build_eval_fn(self):
        """Return JAX-traceable ``(theta, tau) -> k(tau)``."""
        raise NotImplementedError

    def __add__(self, other):
        if not isinstance(other, CeleriteTerm):
            return NotImplemented
        return TermSum(self, other)

    def __radd__(self, other):
        if isinstance(other, CeleriteTerm):
            return TermSum(other, self)
        return NotImplemented


class RealTerm(CeleriteTerm):
    r"""Single real exponential: :math:`k(\tau) = a\,e^{-c\,\tau}`.

    Parameters
    ----------
    a : float
        Amplitude (positive for valid PSD).
    c : float
        Decay rate (positive).
    """

    param_names = ("a", "c")

    def __init__(self, *, a, c):
        self.a = float(a)
        self.c = float(c)

    @property
    def J(self):
        return 1

    @property
    def default_bounds(self):
        return {"a": (1e-8, 10.0), "c": (1e-4, 100.0)}

    def get_coefficients(self):
        return (np.array([self.a]), np.array([self.c]),
                np.array([]), np.array([]), np.array([]), np.array([]))

    def get_params(self):
        return {"a": self.a, "c": self.c}

    def _build_matrices_fn(self):
        def fn(theta, t):
            a, c = theta[0], theta[1]
            N = t.shape[0]
            c_vec = jnp.array([c])
            a_diag = a
            U = jnp.broadcast_to(jnp.array([a]), (N, 1))
            V = jnp.ones((N, 1))
            return c_vec, a_diag, U, V
        return fn

    def _build_eval_fn(self):
        def fn(theta, tau):
            a, c = theta[0], theta[1]
            return a * jnp.exp(-c * jnp.abs(tau))
        return fn


class ComplexTerm(CeleriteTerm):
    r"""Damped sinusoid: :math:`k(\tau) = e^{-c\tau}[a\cos(d\tau) + b\sin(d\tau)]`.

    Parameters
    ----------
    a, b : float
        Cosine and sine amplitudes.
    c : float
        Decay rate (positive).
    d : float
        Oscillation frequency.
    """

    param_names = ("a", "b", "c", "d")

    def __init__(self, *, a, b, c, d):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)

    @property
    def J(self):
        return 2

    @property
    def default_bounds(self):
        return {"a": (-10.0, 10.0), "b": (-10.0, 10.0),
                "c": (1e-4, 100.0), "d": (1e-4, 100.0)}

    def get_coefficients(self):
        return (np.array([]), np.array([]),
                np.array([self.a]), np.array([self.b]),
                np.array([self.c]), np.array([self.d]))

    def get_params(self):
        return {"a": self.a, "b": self.b, "c": self.c, "d": self.d}

    def _build_matrices_fn(self):
        def fn(theta, t):
            a, b, c, d = theta[0], theta[1], theta[2], theta[3]
            arg = d * t
            cos_arg = jnp.cos(arg)
            sin_arg = jnp.sin(arg)
            c_vec = jnp.array([c, c])
            a_diag = a
            U = jnp.column_stack([a * cos_arg + b * sin_arg,
                                  a * sin_arg - b * cos_arg])
            V = jnp.column_stack([cos_arg, sin_arg])
            return c_vec, a_diag, U, V
        return fn

    def _build_eval_fn(self):
        def fn(theta, tau):
            a, b, c, d = theta[0], theta[1], theta[2], theta[3]
            tau = jnp.abs(tau)
            return jnp.exp(-c * tau) * (a * jnp.cos(d * tau)
                                        + b * jnp.sin(d * tau))
        return fn


def _sho_to_S0_w0_Q(theta, param_names):
    """Convert SHO parameter vector to (S0, w0, Q)."""
    if param_names == ("sigma", "rho", "Q"):
        sigma, rho, Q = theta[0], theta[1], theta[2]
        w0 = 2 * jnp.pi / rho
        S0 = sigma ** 2 / (w0 * Q)
    elif param_names == ("sigma", "rho", "tau"):
        sigma, rho, tau = theta[0], theta[1], theta[2]
        w0 = 2 * jnp.pi / rho
        Q = tau * w0 / 2
        S0 = sigma ** 2 / (w0 * Q)
    else:
        S0, w0, Q = theta[0], theta[1], theta[2]
    return S0, w0, Q


def _sho_matrices_underdamped(S0, w0, Q, t):
    f = jnp.sqrt(4 * Q ** 2 - 1)
    a_c = S0 * w0 * Q
    b_c = S0 * w0 * Q / f
    c_c = w0 / (2 * Q)
    d_c = c_c * f
    arg = d_c * t
    cos_arg = jnp.cos(arg)
    sin_arg = jnp.sin(arg)
    c_vec = jnp.array([c_c, c_c])
    a_diag = a_c
    U = jnp.column_stack([a_c * cos_arg + b_c * sin_arg,
                          a_c * sin_arg - b_c * cos_arg])
    V = jnp.column_stack([cos_arg, sin_arg])
    return c_vec, a_diag, U, V


def _sho_matrices_overdamped(S0, w0, Q, t):
    N = t.shape[0]
    f = jnp.sqrt(1 - 4 * Q ** 2)
    a_r1 = 0.5 * S0 * w0 * Q * (1 + 1 / f)
    a_r2 = 0.5 * S0 * w0 * Q * (1 - 1 / f)
    c_r1 = w0 * (1 - f) / (2 * Q)
    c_r2 = w0 * (1 + f) / (2 * Q)
    c_vec = jnp.array([c_r1, c_r2])
    a_diag = a_r1 + a_r2
    U = jnp.broadcast_to(jnp.array([a_r1, a_r2]), (N, 2))
    V = jnp.ones((N, 2))
    return c_vec, a_diag, U, V


def _sho_eval_underdamped(S0, w0, Q, tau):
    f = jnp.sqrt(4 * Q ** 2 - 1)
    a_c = S0 * w0 * Q
    b_c = S0 * w0 * Q / f
    c_c = w0 / (2 * Q)
    d_c = c_c * f
    return jnp.exp(-c_c * tau) * (a_c * jnp.cos(d_c * tau)
                                   + b_c * jnp.sin(d_c * tau))


def _sho_eval_overdamped(S0, w0, Q, tau):
    f = jnp.sqrt(1 - 4 * Q ** 2)
    a_r1 = 0.5 * S0 * w0 * Q * (1 + 1 / f)
    a_r2 = 0.5 * S0 * w0 * Q * (1 - 1 / f)
    c_r1 = w0 * (1 - f) / (2 * Q)
    c_r2 = w0 * (1 + f) / (2 * Q)
    return a_r1 * jnp.exp(-c_r1 * tau) + a_r2 * jnp.exp(-c_r2 * tau)


class SHOTerm(CeleriteTerm):
    r"""Stochastically-driven harmonic oscillator.

    .. math::

        S(\omega) = \sqrt{2/\pi}\,
        \frac{S_0\,\omega_0^4}{(\omega^2 - \omega_0^2)^2
        + \omega_0^2\omega^2/Q^2}

    Accepts three parameterization styles (pass exactly one set):

    * ``(sigma, rho, Q)`` — ``\sigma^2 = S_0 \omega_0 Q``,
      ``\rho = 2\pi/\omega_0``
    * ``(sigma, rho, tau)`` — ``\tau = 2Q/\omega_0``
    * ``(S0, w0, Q)`` — direct

    Parameters
    ----------
    sigma, rho, tau, S0, w0, Q : float
    """

    def __init__(self, *, sigma=None, rho=None, tau=None,
                 S0=None, w0=None, Q=None):
        if w0 is None:
            if rho is None:
                raise ValueError("Provide either rho or w0")
            w0 = 2 * np.pi / rho
        if Q is None:
            if tau is None:
                raise ValueError("Provide either Q or tau")
            Q = tau * w0 / 2
        if S0 is None:
            if sigma is None:
                raise ValueError("Provide either sigma or S0")
            S0 = sigma ** 2 / (w0 * Q)

        self.S0 = float(S0)
        self.w0 = float(w0)
        self.Q = float(Q)
        self.sigma = float(sigma if sigma is not None else np.sqrt(S0 * w0 * Q))
        self.rho = float(rho if rho is not None else 2 * np.pi / w0)

        if sigma is not None and rho is not None:
            if tau is not None:
                self.param_names = ("sigma", "rho", "tau")
                self._tau = float(tau)
            else:
                self.param_names = ("sigma", "rho", "Q")
        else:
            self.param_names = ("S0", "w0", "Q")

    @property
    def J(self):
        return 2

    @property
    def default_bounds(self):
        if self.param_names == ("sigma", "rho", "Q"):
            return {"sigma": (1e-6, 1.0), "rho": (0.1, 100.0),
                    "Q": (0.1, 50.0)}
        elif self.param_names == ("sigma", "rho", "tau"):
            return {"sigma": (1e-6, 1.0), "rho": (0.1, 100.0),
                    "tau": (0.1, 200.0)}
        return {"S0": (1e-12, 10.0), "w0": (0.01, 100.0), "Q": (0.1, 50.0)}

    def get_coefficients(self):
        S0, w0, Q = self.S0, self.w0, self.Q
        if Q >= 0.5:
            f = np.sqrt(4 * Q ** 2 - 1)
            return (np.array([]), np.array([]),
                    np.array([S0 * w0 * Q]),
                    np.array([S0 * w0 * Q / f]),
                    np.array([w0 / (2 * Q)]),
                    np.array([w0 * f / (2 * Q)]))
        f = np.sqrt(1 - 4 * Q ** 2)
        return (np.array([0.5 * S0 * w0 * Q * (1 + 1 / f),
                          0.5 * S0 * w0 * Q * (1 - 1 / f)]),
                np.array([w0 * (1 - f) / (2 * Q),
                          w0 * (1 + f) / (2 * Q)]),
                np.array([]), np.array([]), np.array([]), np.array([]))

    def get_params(self):
        if self.param_names == ("sigma", "rho", "Q"):
            return {"sigma": self.sigma, "rho": self.rho, "Q": self.Q}
        elif self.param_names == ("sigma", "rho", "tau"):
            return {"sigma": self.sigma, "rho": self.rho, "tau": self._tau}
        return {"S0": self.S0, "w0": self.w0, "Q": self.Q}

    def _build_matrices_fn(self):
        pnames = self.param_names

        def fn(theta, t):
            S0, w0, Q = _sho_to_S0_w0_Q(theta, pnames)
            return jax.lax.cond(
                Q >= 0.5,
                lambda: _sho_matrices_underdamped(S0, w0, Q, t),
                lambda: _sho_matrices_overdamped(S0, w0, Q, t))

        return fn

    def _build_eval_fn(self):
        pnames = self.param_names

        def fn(theta, tau):
            S0, w0, Q = _sho_to_S0_w0_Q(theta, pnames)
            tau = jnp.abs(tau)
            return jax.lax.cond(
                Q >= 0.5,
                lambda: _sho_eval_underdamped(S0, w0, Q, tau),
                lambda: _sho_eval_overdamped(S0, w0, Q, tau))

        return fn


class RotationTerm(CeleriteTerm):
    r"""Stellar rotation kernel — two coupled SHOs at period *P* and *P/2*.

    Primary mode (period *P*):

    .. math::

        Q_1 = \tfrac{1}{2} + Q_0 + \Delta Q, \quad
        \omega_1 = \frac{4\pi Q_1}{P\sqrt{4Q_1^2 - 1}}, \quad
        S_1 = \frac{\sigma^2}{(1+f)\,\omega_1 Q_1}

    Secondary mode (period *P/2*):

    .. math::

        Q_2 = \tfrac{1}{2} + Q_0, \quad
        \omega_2 = \frac{8\pi Q_2}{P\sqrt{4Q_2^2 - 1}}, \quad
        S_2 = \frac{f\,\sigma^2}{(1+f)\,\omega_2 Q_2}

    Both modes are always underdamped (Q > 0.5 by construction).

    Parameters
    ----------
    sigma : float
        Overall variability amplitude.
    period : float
        Rotation period.
    Q0 : float
        Baseline quality factor offset (Q0 > 0).
    dQ : float
        Extra quality factor for the primary mode (dQ > 0).
    f : float
        Fractional amplitude of the secondary (half-period) mode.
    """

    param_names = ("sigma", "period", "Q0", "dQ", "f")

    def __init__(self, *, sigma, period, Q0, dQ, f):
        self.sigma = float(sigma)
        self.period = float(period)
        self.Q0 = float(Q0)
        self.dQ = float(dQ)
        self.f = float(f)

    @property
    def J(self):
        return 4

    @property
    def default_bounds(self):
        return {"sigma": (1e-6, 1.0), "period": (0.1, 100.0),
                "Q0": (0.01, 50.0), "dQ": (0.01, 50.0),
                "f": (0.01, 10.0)}

    def _sho_params(self):
        """Return (S0_1, w0_1, Q_1, S0_2, w0_2, Q_2)."""
        Q1 = 0.5 + self.Q0 + self.dQ
        Q2 = 0.5 + self.Q0
        w1 = 4 * np.pi * Q1 / (self.period * np.sqrt(4 * Q1 ** 2 - 1))
        w2 = 8 * np.pi * Q2 / (self.period * np.sqrt(4 * Q2 ** 2 - 1))
        S1 = self.sigma ** 2 / ((1 + self.f) * w1 * Q1)
        S2 = self.f * self.sigma ** 2 / ((1 + self.f) * w2 * Q2)
        return S1, w1, Q1, S2, w2, Q2

    def get_coefficients(self):
        S1, w1, Q1, S2, w2, Q2 = self._sho_params()
        ar, cr, ac, bc, cc, dc = [], [], [], [], [], []
        for S0, w0, Q in [(S1, w1, Q1), (S2, w2, Q2)]:
            f = np.sqrt(4 * Q ** 2 - 1)
            ac.append(S0 * w0 * Q)
            bc.append(S0 * w0 * Q / f)
            cc.append(w0 / (2 * Q))
            dc.append(w0 * f / (2 * Q))
        return (np.array(ar), np.array(cr),
                np.array(ac), np.array(bc), np.array(cc), np.array(dc))

    def get_params(self):
        return {"sigma": self.sigma, "period": self.period,
                "Q0": self.Q0, "dQ": self.dQ, "f": self.f}

    def _build_matrices_fn(self):
        def fn(theta, t):
            sigma = theta[0]
            period = theta[1]
            Q0 = theta[2]
            dQ = theta[3]
            f_mix = theta[4]

            Q1 = 0.5 + Q0 + dQ
            Q2 = 0.5 + Q0
            w1 = 4 * jnp.pi * Q1 / (period * jnp.sqrt(4 * Q1 ** 2 - 1))
            w2 = 8 * jnp.pi * Q2 / (period * jnp.sqrt(4 * Q2 ** 2 - 1))
            S1 = sigma ** 2 / ((1 + f_mix) * w1 * Q1)
            S2 = f_mix * sigma ** 2 / ((1 + f_mix) * w2 * Q2)

            c1, a1, U1, V1 = _sho_matrices_underdamped(S1, w1, Q1, t)
            c2, a2, U2, V2 = _sho_matrices_underdamped(S2, w2, Q2, t)

            return (jnp.concatenate([c1, c2]), a1 + a2,
                    jnp.concatenate([U1, U2], axis=1),
                    jnp.concatenate([V1, V2], axis=1))

        return fn

    def _build_eval_fn(self):
        def fn(theta, tau):
            sigma = theta[0]
            period = theta[1]
            Q0 = theta[2]
            dQ = theta[3]
            f_mix = theta[4]

            Q1 = 0.5 + Q0 + dQ
            Q2 = 0.5 + Q0
            w1 = 4 * jnp.pi * Q1 / (period * jnp.sqrt(4 * Q1 ** 2 - 1))
            w2 = 8 * jnp.pi * Q2 / (period * jnp.sqrt(4 * Q2 ** 2 - 1))
            S1 = sigma ** 2 / ((1 + f_mix) * w1 * Q1)
            S2 = f_mix * sigma ** 2 / ((1 + f_mix) * w2 * Q2)

            tau = jnp.abs(tau)
            return (_sho_eval_underdamped(S1, w1, Q1, tau)
                    + _sho_eval_underdamped(S2, w2, Q2, tau))

        return fn


class Matern32Term(CeleriteTerm):
    r"""Matern-3/2 kernel approximation via a celerite complex term.

    Exact in the limit ``eps -> 0``:

    .. math::

        k(\tau) = \sigma^2\,\bigl(1 + \sqrt{3}\,\tau/\rho\bigr)\,
        e^{-\sqrt{3}\,\tau/\rho}

    Parameters
    ----------
    sigma : float
        Amplitude.
    rho : float
        Correlation length-scale.
    eps : float
        Approximation parameter (default 0.01).
    """

    param_names = ("sigma", "rho")

    def __init__(self, *, sigma, rho, eps=0.01):
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.eps = float(eps)

    @property
    def J(self):
        return 2

    @property
    def default_bounds(self):
        return {"sigma": (1e-6, 1.0), "rho": (0.1, 200.0)}

    def get_coefficients(self):
        w0 = np.sqrt(3) / self.rho
        S0 = self.sigma ** 2 / w0
        return (np.array([]), np.array([]),
                np.array([w0 * S0]),
                np.array([w0 ** 2 * S0 / self.eps]),
                np.array([w0]),
                np.array([self.eps]))

    def get_params(self):
        return {"sigma": self.sigma, "rho": self.rho}

    def _build_matrices_fn(self):
        eps = self.eps

        def fn(theta, t):
            sigma, rho = theta[0], theta[1]
            w0 = jnp.sqrt(3.0) / rho
            S0 = sigma ** 2 / w0
            ac = w0 * S0
            bc = w0 ** 2 * S0 / eps
            cc = w0
            dc = eps

            arg = dc * t
            cos_arg = jnp.cos(arg)
            sin_arg = jnp.sin(arg)
            c_vec = jnp.array([cc, cc])
            a_diag = ac
            U = jnp.column_stack([ac * cos_arg + bc * sin_arg,
                                  ac * sin_arg - bc * cos_arg])
            V = jnp.column_stack([cos_arg, sin_arg])
            return c_vec, a_diag, U, V

        return fn

    def _build_eval_fn(self):
        eps = self.eps

        def fn(theta, tau):
            sigma, rho = theta[0], theta[1]
            w0 = jnp.sqrt(3.0) / rho
            S0 = sigma ** 2 / w0
            ac = w0 * S0
            bc = w0 ** 2 * S0 / eps
            tau = jnp.abs(tau)
            return jnp.exp(-w0 * tau) * (ac * jnp.cos(eps * tau)
                                          + bc * jnp.sin(eps * tau))

        return fn


class TermSum(CeleriteTerm):
    """Sum of celerite terms (built via ``term1 + term2``)."""

    def __init__(self, term1, term2):
        self.terms = []
        for t in (term1, term2):
            if isinstance(t, TermSum):
                self.terms.extend(t.terms)
            else:
                self.terms.append(t)

        names = []
        self._param_slices = []
        offset = 0
        for i, term in enumerate(self.terms):
            for name in term.param_names:
                names.append(f"{name}_{i}")
            self._param_slices.append(slice(offset, offset + term.n_params))
            offset += term.n_params
        self.param_names = tuple(names)

    @property
    def J(self):
        return sum(t.J for t in self.terms)

    @property
    def default_bounds(self):
        bounds = {}
        for i, term in enumerate(self.terms):
            for name, val in term.default_bounds.items():
                bounds[f"{name}_{i}"] = val
        return bounds

    def get_coefficients(self):
        ar, cr, ac, bc, cc, dc = [], [], [], [], [], []
        for term in self.terms:
            t_ar, t_cr, t_ac, t_bc, t_cc, t_dc = term.get_coefficients()
            ar.extend(t_ar)
            cr.extend(t_cr)
            ac.extend(t_ac)
            bc.extend(t_bc)
            cc.extend(t_cc)
            dc.extend(t_dc)
        return (np.array(ar), np.array(cr), np.array(ac),
                np.array(bc), np.array(cc), np.array(dc))

    def get_params(self):
        result = {}
        for i, term in enumerate(self.terms):
            for name, val in term.get_params().items():
                result[f"{name}_{i}"] = val
        return result

    def _build_matrices_fn(self):
        fns = [t._build_matrices_fn() for t in self.terms]
        slices = list(self._param_slices)

        def fn(theta, t):
            c_parts, a_diag, U_parts, V_parts = [], 0.0, [], []
            for sub_fn, sl in zip(fns, slices):
                c_i, a_i, U_i, V_i = sub_fn(theta[sl], t)
                c_parts.append(c_i)
                a_diag = a_diag + a_i
                U_parts.append(U_i)
                V_parts.append(V_i)
            return (jnp.concatenate(c_parts), a_diag,
                    jnp.concatenate(U_parts, axis=1),
                    jnp.concatenate(V_parts, axis=1))

        return fn

    def _build_eval_fn(self):
        fns = [t._build_eval_fn() for t in self.terms]
        slices = list(self._param_slices)

        def fn(theta, tau):
            result = jnp.zeros_like(tau)
            for sub_fn, sl in zip(fns, slices):
                result = result + sub_fn(theta[sl], tau)
            return result

        return fn


# =====================================================================
# CeleriteGPSolver
# =====================================================================

class CeleriteGPSolver:
    """O(N) Gaussian Process solver using celerite kernels.

    Parameters
    ----------
    data : TimeSeriesData
        Observations (times must be sortable; duplicates not allowed).
    kernel : CeleriteTerm
        Celerite kernel (single term or sum of terms).
    mean : float or callable or None
        Constant mean, mean function, or None (uses data mean).
    fit_sigma_n : bool
        Include white-noise amplitude as a free parameter.
    bounds : dict or None
        Parameter bounds ``{name: (lo, hi)}``.  Supports ``log_``-prefixed
        keys for log10-space sampling (same convention as ``GPSolver``).
    log_prior : callable or None
        Custom log-prior ``f(theta) -> scalar``.  If None, uses soft
        uniform within bounds.
    save_dir : str or None
        Directory for auto-saving fit results.
    """

    def __init__(self, data, kernel, mean=None, fit_sigma_n=False,
                 bounds=None, log_prior=None, save_dir=None):
        from .observations import TimeSeriesData

        if not isinstance(data, TimeSeriesData):
            raise TypeError("data must be a TimeSeriesData instance")
        if not isinstance(kernel, CeleriteTerm):
            raise TypeError("kernel must be a CeleriteTerm instance")

        self.data = data
        self.kernel = kernel

        order = np.argsort(data.x)
        self.x = jnp.asarray(data.x[order], dtype=jnp.float64)
        self.y = jnp.asarray(data.y[order], dtype=jnp.float64)
        self.yerr = jnp.asarray(data.yerr[order], dtype=jnp.float64)
        self.N = len(self.x)

        if mean is None:
            self.mean_val = float(jnp.mean(self.y))
            self.mean_func = lambda t: self.mean_val
        elif callable(mean):
            self.mean_func = mean
            self.mean_val = float(mean(self.x[0]))
        else:
            self.mean_val = float(mean)
            self.mean_func = lambda t: self.mean_val

        self.fit_sigma_n = fit_sigma_n
        self._n_kernel = kernel.n_params

        _base_keys = list(kernel.param_names)
        if fit_sigma_n:
            _base_keys.append("sigma_n")

        self._log_param_map = {}
        if isinstance(bounds, dict):
            for k in bounds:
                if k.startswith("log_"):
                    self._log_param_map[k] = k[4:]

        _phys_to_log = {v: k for k, v in self._log_param_map.items()}
        self.param_keys = tuple(_phys_to_log.get(k, k) for k in _base_keys)
        self.n_params = len(self.param_keys)

        _default = dict(kernel.default_bounds)
        _default["sigma_n"] = (1e-6, 0.1)

        if bounds is None:
            self.bounds = jnp.array(
                [_default[k] for k in _base_keys], dtype=jnp.float64)
        elif isinstance(bounds, dict):
            self.bounds = jnp.array(
                [bounds.get(pk, _default.get(bk, (-1e10, 1e10)))
                 for pk, bk in zip(self.param_keys, _base_keys)],
                dtype=jnp.float64)
        else:
            self.bounds = jnp.asarray(bounds, dtype=jnp.float64)

        params = kernel.get_params()
        theta0_list = [float(params[k]) for k in kernel.param_names]
        if fit_sigma_n:
            theta0_list.append(float(np.median(data.yerr)))

        self.theta0 = jnp.array([
            np.log10(v) if _base_keys[i] in _phys_to_log
            else v
            for i, v in enumerate(theta0_list)
        ], dtype=jnp.float64)

        self._build_transform()
        self._matrices_fn = kernel._build_matrices_fn()
        self._eval_fn = kernel._build_eval_fn()

        self._custom_log_prior = log_prior
        self._build_logposterior()

        self.map_estimate = None
        self._current_theta = self.theta0.copy()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_transform(self):
        if not self._log_param_map:
            self._to_physical = lambda x: x
            return
        keys = list(self.param_keys)
        log_indices = jnp.array(
            [keys.index(k) for k in self._log_param_map], dtype=jnp.int32)

        @jax.jit
        def to_physical(theta_arr):
            return theta_arr.at[log_indices].set(
                10.0 ** theta_arr[log_indices])

        self._to_physical = to_physical

    def _build_logposterior(self):
        matrices_fn = self._matrices_fn
        x, y, yerr = self.x, self.y, self.yerr
        mean_val = self.mean_val
        bounds = self.bounds
        fit_sn = self.fit_sigma_n
        n_kernel = self._n_kernel
        to_phys = self._to_physical
        custom_prior = self._custom_log_prior
        N = self.N

        @jax.jit
        def log_posterior(theta_arr):
            theta = to_phys(theta_arr)
            theta_k = theta[:n_kernel]
            sigma_n = jax.lax.cond(
                fit_sn, lambda: theta[n_kernel], lambda: 0.0)

            c, a_diag, U, V = matrices_fn(theta_k, x)
            a = jnp.full(N, a_diag) + yerr ** 2 + sigma_n ** 2 + 1e-8

            d, W = _celerite_factor(x, c, a, U, V)
            resid = y - mean_val
            z = _celerite_solve_lower(x, c, U, W, resid)
            norm = jnp.sum(z ** 2 / d)
            log_det = jnp.sum(jnp.log(d))

            ll = -0.5 * (norm + log_det + N * jnp.log(2 * jnp.pi))
            lp = (custom_prior(theta_arr) if custom_prior is not None
                  else _default_log_prior(theta_arr, bounds))
            return ll + lp

        @jax.jit
        def neg_log_posterior(theta_arr):
            return -log_posterior(theta_arr)

        self.log_posterior = log_posterior
        self.neg_log_posterior = neg_log_posterior
        self.grad_log_posterior = jax.jit(jax.grad(log_posterior))
        self.grad_neg_log_posterior = jax.jit(jax.grad(neg_log_posterior))

        @jax.jit
        def _log_prior_fn(theta_arr):
            return (custom_prior(theta_arr) if custom_prior is not None
                    else _default_log_prior(theta_arr, bounds))

        @jax.jit
        def _log_likelihood_fn(theta_arr):
            theta = to_phys(theta_arr)
            theta_k = theta[:n_kernel]
            sigma_n = jax.lax.cond(
                fit_sn, lambda: theta[n_kernel], lambda: 0.0)
            c, a_diag, U, V = matrices_fn(theta_k, x)
            a = jnp.full(N, a_diag) + yerr ** 2 + sigma_n ** 2 + 1e-8
            d, W = _celerite_factor(x, c, a, U, V)
            resid = y - mean_val
            z = _celerite_solve_lower(x, c, U, W, resid)
            norm = jnp.sum(z ** 2 / d)
            log_det = jnp.sum(jnp.log(d))
            return -0.5 * (norm + log_det + N * jnp.log(2 * jnp.pi))

        self.log_prior_fn = _log_prior_fn
        self.log_likelihood_fn = _log_likelihood_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_jax(self, recompute=True):
        """Pre-compile and warm up all JIT functions.

        Returns *self* for chaining.
        """
        theta0 = self.theta0
        t0 = _time.time()
        jax.block_until_ready(self.log_posterior(theta0))
        jax.block_until_ready(self.neg_log_posterior(theta0))
        jax.block_until_ready(self.grad_log_posterior(theta0))
        jax.block_until_ready(self.grad_neg_log_posterior(theta0))
        print(f"JAX celerite solver compiled in "
              f"{np.round(_time.time() - t0, 2)}s  (J={self.kernel.J})")
        if recompute:
            t0 = _time.time()
            jax.block_until_ready(self.log_posterior(theta0))
            jax.block_until_ready(self.grad_log_posterior(theta0))
            print(f"JAX celerite solver recompute in "
                  f"{np.round(_time.time() - t0, 2)}s")
        return self

    def log_likelihood(self, theta=None):
        """Marginal log-likelihood at *theta* (or current params)."""
        if theta is None:
            theta = self._current_theta
        else:
            theta = jnp.asarray(theta, dtype=jnp.float64)
        return float(self.log_likelihood_fn(theta))

    def update_params(self, params):
        """Update kernel parameters from a dict and rebuild covariance.

        Parameters
        ----------
        params : dict
            Mapping of param names to new values (physical units).
        """
        theta_list = list(np.asarray(self._current_theta))
        _base_keys = list(self.kernel.param_names)
        if self.fit_sigma_n:
            _base_keys.append("sigma_n")
        for k, v in params.items():
            if k in self.param_keys:
                idx = list(self.param_keys).index(k)
                theta_list[idx] = float(v)
            elif k in _base_keys:
                idx = _base_keys.index(k)
                phys_key = _base_keys[idx]
                mapped = {v2: k2 for k2, v2 in self._log_param_map.items()}
                if phys_key in mapped:
                    idx = list(self.param_keys).index(mapped[phys_key])
                    theta_list[idx] = np.log10(float(v))
                else:
                    theta_list[idx] = float(v)
        self._current_theta = jnp.array(theta_list, dtype=jnp.float64)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, xpred, return_cov=False):
        """Predictive mean and variance at new input locations.

        Parameters
        ----------
        xpred : array_like, shape (M,)
        return_cov : bool
            If True, return full (M, M) covariance instead of diagonal.

        Returns
        -------
        mu_pred : ndarray, shape (M,)
        var_pred : ndarray, shape (M,) or (M, M)
        """
        xpred = jnp.asarray(xpred, dtype=jnp.float64)
        M = len(xpred)

        theta = self._to_physical(self._current_theta)
        theta_k = theta[:self._n_kernel]
        sigma_n = float(theta[self._n_kernel]) if self.fit_sigma_n else 0.0

        c, a_diag, U, V = self._matrices_fn(theta_k, self.x)
        a = jnp.full(self.N, a_diag) + self.yerr ** 2 + sigma_n ** 2 + 1e-8

        d, W = _celerite_factor(self.x, c, a, U, V)
        resid = self.y - self.mean_val
        z = _celerite_solve_lower(self.x, c, U, W, resid)
        alpha = _celerite_solve_upper(self.x, c, U, W, z / d)

        lag_cross = jnp.abs(xpred[:, None] - self.x[None, :]).ravel()
        K_star = self._eval_fn(theta_k, lag_cross).reshape(M, self.N)

        mu_prior = self.mean_func(xpred)
        if jnp.isscalar(mu_prior):
            mu_prior = jnp.full(M, mu_prior)
        mu_pred = mu_prior + K_star @ alpha

        V_lower = _celerite_solve_lower_matrix(
            self.x, c, U, W, K_star.T)

        if return_cov:
            lag_pred = jnp.abs(xpred[:, None] - xpred[None, :]).ravel()
            K_pred = self._eval_fn(theta_k, lag_pred).reshape(M, M)
            V_scaled = V_lower / jnp.sqrt(d[:, None])
            cov_pred = K_pred - V_scaled.T @ V_scaled
            return np.asarray(mu_pred), np.asarray(cov_pred)

        k0 = float(a_diag)
        var_pred = k0 - jnp.sum(V_lower ** 2 / d[:, None], axis=0)
        return np.asarray(mu_pred), np.asarray(var_pred)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_prior(self, xpred, n_samples=1, rng=None):
        """Draw samples from the GP prior."""
        xpred = jnp.asarray(xpred, dtype=jnp.float64)
        if rng is None:
            rng = np.random.default_rng()

        theta = self._to_physical(self._current_theta)
        theta_k = theta[:self._n_kernel]

        c, a_diag, U, V = self._matrices_fn(theta_k, xpred)
        a = jnp.full(len(xpred), a_diag) + 1e-10

        d, W = _celerite_factor(xpred, c, a, U, V)

        mu = self.mean_func(xpred)
        if jnp.isscalar(mu):
            mu = jnp.full(len(xpred), mu)
        mu = np.asarray(mu)

        samples = np.empty((n_samples, len(xpred)))
        for i in range(n_samples):
            z = jnp.asarray(rng.standard_normal(len(xpred)))
            y = _celerite_dot_tril(xpred, c, U, W, d, z)
            samples[i] = mu + np.asarray(y)
        return samples

    def sample_posterior(self, xpred, n_samples=1, rng=None):
        """Draw samples from the GP posterior."""
        if rng is None:
            rng = np.random.default_rng()
        mu_pred, cov_pred = self.predict(xpred, return_cov=True)
        cov_pred = cov_pred + 1e-10 * np.eye(len(xpred))
        return rng.multivariate_normal(mu_pred, cov_pred, size=n_samples)

    # ------------------------------------------------------------------
    # Kernel and ACF evaluation
    # ------------------------------------------------------------------

    def compute_kernel(self, tlags):
        """Evaluate the kernel at given time lags (uses current params)."""
        tlags = jnp.asarray(tlags, dtype=jnp.float64)
        theta = self._to_physical(self._current_theta)
        theta_k = theta[:self._n_kernel]
        return np.asarray(self._eval_fn(theta_k, jnp.abs(tlags)))

    def compute_acf(self, tlags=None, n_bins=50, normalize=True):
        """Empirical autocorrelation function of the data.

        Returns
        -------
        lag_centers : ndarray
        acf : ndarray
        """
        if tlags is not None:
            tlags = np.asarray(tlags, dtype=np.float64)
            max_lag = float(tlags[-1])
            n_bins = len(tlags) - 1
        else:
            max_lag = self.data.baseline / 2.0

        lag_centers, acf = self.data.compute_acf(
            n_bins=n_bins, max_lag=max_lag)
        if not normalize:
            var = np.var(self.data.y)
            acf = acf * var
        return lag_centers, acf

    # ------------------------------------------------------------------
    # MAP estimation
    # ------------------------------------------------------------------

    def _resolve_keys(self, keys):
        if keys is None:
            return (list(range(self.n_params)), [], jnp.array([]))
        pk_set = set(self.param_keys)
        normalized = []
        for k in keys:
            if k in pk_set:
                normalized.append(k)
            elif k.startswith("log_") and k[4:] in pk_set:
                normalized.append(k[4:])
            elif f"log_{k}" in pk_set:
                normalized.append(f"log_{k}")
            else:
                raise ValueError(
                    f"Unknown key '{k}'. Valid: {self.param_keys}")
            keys = normalized
        free_idx = [i for i, k in enumerate(self.param_keys) if k in keys]
        fixed_idx = [i for i, k in enumerate(self.param_keys) if k not in keys]
        fixed_vals = (self._current_theta[jnp.array(fixed_idx)]
                      if fixed_idx else jnp.array([]))
        return free_idx, fixed_idx, fixed_vals

    @staticmethod
    def _theta_from_free(free_vals, free_idx, fixed_idx, fixed_vals):
        n = len(free_idx) + len(fixed_idx)
        theta = jnp.zeros(n)
        theta = theta.at[jnp.array(free_idx)].set(free_vals)
        if len(fixed_idx) > 0:
            theta = theta.at[jnp.array(fixed_idx)].set(fixed_vals)
        return theta

    def fit_map(self, theta0=None, keys=None, method="L-BFGS-B",
                maxiter=500, ftol=0, gtol=1e-8, disp=False,
                nopt=1, ncore=None, rng=None):
        """Find the maximum a posteriori (MAP) estimate.

        Parameters
        ----------
        theta0 : dict or array_like or None
            Starting point.  Dict keys matching ``param_keys`` override
            individual entries; remaining keys are inferred as free
            parameters when *keys* is None.
        keys : list of str or None
            Free parameters.  If None, all are free.
        method : str
            Scipy optimizer method.
        maxiter : int
        ftol, gtol : float
        disp : bool
        nopt : int
            Number of multi-start trials (> 1 triggers parallel starts).
        ncore : int or None
        rng : numpy.random.Generator or None

        Returns
        -------
        theta_dict : dict
        result : scipy.optimize.OptimizeResult
        """
        if nopt > 1:
            return self.fit_map_parallel(
                nopt=nopt, ncore=ncore, keys=keys, method=method,
                maxiter=maxiter, ftol=ftol, gtol=gtol, disp=disp, rng=rng)

        from scipy.optimize import minimize

        if theta0 is None:
            theta0_arr = self._current_theta.copy()
        elif isinstance(theta0, dict):
            theta0_arr = self._current_theta.copy()
            dict_keys_in_params = []
            for k, v in theta0.items():
                if k in self.param_keys:
                    idx = list(self.param_keys).index(k)
                    theta0_arr = theta0_arr.at[idx].set(float(v))
                    dict_keys_in_params.append(k)
            if keys is None and dict_keys_in_params:
                keys = dict_keys_in_params
        else:
            theta0_arr = jnp.asarray(theta0, dtype=jnp.float64)

        free_idx, fixed_idx, fixed_vals = self._resolve_keys(keys)

        free0 = theta0_arr[jnp.array(free_idx)]
        free_bounds = self.bounds[jnp.array(free_idx)]
        blo, bhi = free_bounds[:, 0], free_bounds[:, 1]
        brange = bhi - blo

        u0 = np.asarray((free0 - blo) / brange, dtype=np.float64)

        log_post = self.log_posterior
        _theta_from_free = self._theta_from_free

        @jax.jit
        def neg_logpost_u(u_arr):
            free_theta = blo + u_arr * brange
            theta_full = _theta_from_free(
                free_theta, free_idx, fixed_idx, fixed_vals)
            return -log_post(theta_full)

        vg_fn = jax.jit(jax.value_and_grad(neg_logpost_u))
        jax.block_until_ready(vg_fn(jnp.array(u0, dtype=jnp.float64)))

        _gradient_free = method.lower() in ("nelder-mead", "cobyla", "powell")

        if _gradient_free:
            def objective(u_np):
                val, _ = vg_fn(jnp.array(u_np, dtype=jnp.float64))
                v = float(val)
                return 1e30 if not np.isfinite(v) else v
            _kw = dict(method=method,
                       options={"maxiter": maxiter, "xatol": ftol,
                                "fatol": ftol, "disp": disp})
        else:
            def objective(u_np):
                val, grad = vg_fn(jnp.array(u_np, dtype=jnp.float64))
                v = float(val)
                g = np.asarray(grad, dtype=np.float64)
                if not np.isfinite(v):
                    return 1e30, np.zeros_like(g)
                if not np.all(np.isfinite(g)):
                    return v, np.zeros_like(g)
                return v, g
            _kw = dict(jac=True, method=method,
                       bounds=[(0.0, 1.0)] * len(free_idx),
                       options={"maxiter": maxiter, "ftol": ftol,
                                "gtol": gtol, "disp": disp})

        result = minimize(objective, u0, **_kw)

        free_best = blo + jnp.array(result.x, dtype=jnp.float64) * brange
        theta_best = self._theta_from_free(
            free_best, free_idx, fixed_idx, fixed_vals)

        self._current_theta = theta_best
        self.map_estimate = theta_best

        theta_dict = {k: float(theta_best[i])
                      for i, k in enumerate(self.param_keys)}
        self._autosave("map_results.npz", **theta_dict)
        return theta_dict, result

    def fit_map_parallel(self, nopt=10, ncore=None, keys=None,
                         method="L-BFGS-B", maxiter=500, ftol=0,
                         gtol=1e-8, disp=False, rng=None):
        """Multi-start MAP estimation."""
        from concurrent.futures import ThreadPoolExecutor

        if ncore is None:
            ncore = min(nopt, os.cpu_count() or 1)
        if rng is None:
            rng = np.random.default_rng()

        free_idx, _, _ = self._resolve_keys(keys)
        free_bounds = np.asarray(self.bounds[jnp.array(free_idx)])
        blo, bhi = free_bounds[:, 0], free_bounds[:, 1]

        free_keys = [self.param_keys[i] for i in free_idx]
        seeds = rng.integers(0, 2 ** 31, size=nopt)
        starts = []
        for i in range(nopt):
            child = np.random.default_rng(int(seeds[i]))
            u = child.uniform(size=len(free_keys))
            starts.append({k: float(blo[j] + u[j] * (bhi[j] - blo[j]))
                           for j, k in enumerate(free_keys)})

        def _run(theta0_dict):
            return self.fit_map(theta0=theta0_dict, keys=keys,
                                method=method, maxiter=maxiter,
                                ftol=ftol, gtol=gtol, disp=disp)

        with ThreadPoolExecutor(max_workers=ncore) as pool:
            futures = [pool.submit(_run, s) for s in starts]
            results = [f.result() for f in futures]

        results.sort(key=lambda tr: float(tr[1].fun))
        best_theta, best_result = results[0]

        self._current_theta = jnp.array(
            [float(best_theta[k]) for k in self.param_keys],
            dtype=jnp.float64)
        self.map_estimate = self._current_theta
        return best_theta, best_result

    def fit_acf(self, theta0=None, keys=None, tlags=None, n_bins=50,
                method="L-BFGS-B", maxiter=500, ftol=0, gtol=1e-8,
                disp=False, nopt=1, ncore=None, rng=None):
        """Fit the kernel to the empirical ACF via least-squares.

        Parameters
        ----------
        theta0 : dict or array_like or None
        keys : list of str or None
        tlags : array_like or None
        n_bins : int
        method, maxiter, ftol, gtol, disp : optimizer config
        nopt : int
        ncore : int or None
        rng : Generator or None

        Returns
        -------
        theta_dict : dict
        result : scipy.optimize.OptimizeResult
        """
        if nopt > 1:
            return self._fit_acf_parallel(
                nopt=nopt, ncore=ncore, keys=keys, tlags=tlags,
                n_bins=n_bins, method=method, maxiter=maxiter,
                ftol=ftol, gtol=gtol, disp=disp, rng=rng)

        from scipy.optimize import minimize as _minimize

        if tlags is None:
            baseline = float(jnp.max(self.x) - jnp.min(self.x))
            tlags = np.linspace(0, baseline / 2, n_bins + 1)

        lag_centers, acf_data = self.compute_acf(tlags=tlags, normalize=False)
        lag_jax = jnp.asarray(lag_centers)
        acf_jax = jnp.asarray(acf_data)

        n_kernel = self._n_kernel
        kernel_keys = list(self.kernel.param_names)

        if theta0 is None:
            theta0_arr = self._current_theta[:n_kernel]
        elif isinstance(theta0, dict):
            theta0_arr = self._current_theta[:n_kernel].copy()
            dict_keys = []
            for k, v in theta0.items():
                if k in kernel_keys:
                    idx = kernel_keys.index(k)
                    theta0_arr = theta0_arr.at[idx].set(float(v))
                    dict_keys.append(k)
            if keys is None and dict_keys:
                keys = dict_keys
        else:
            theta0_arr = jnp.asarray(theta0, dtype=jnp.float64)

        if keys is None:
            free_idx = list(range(n_kernel))
            fixed_idx, fixed_vals = [], jnp.array([])
        else:
            for k in keys:
                if k not in kernel_keys:
                    raise ValueError(
                        f"Unknown key '{k}'. Valid: {kernel_keys}")
            free_idx = [i for i, k in enumerate(kernel_keys) if k in keys]
            fixed_idx = [i for i, k in enumerate(kernel_keys) if k not in keys]
            fixed_vals = (theta0_arr[jnp.array(fixed_idx)]
                          if fixed_idx else jnp.array([]))

        free0 = theta0_arr[jnp.array(free_idx)]
        bounds_kernel = self.bounds[:n_kernel]
        free_bounds = bounds_kernel[jnp.array(free_idx)]
        blo, bhi = free_bounds[:, 0], free_bounds[:, 1]
        brange = bhi - blo
        u0 = np.asarray((free0 - blo) / brange, dtype=np.float64)

        eval_fn = self._eval_fn
        _theta_from_free = self._theta_from_free

        @jax.jit
        def loss_u(u_arr):
            free_theta = blo + u_arr * brange
            theta_full = _theta_from_free(
                free_theta, free_idx, fixed_idx, fixed_vals)
            K_model = eval_fn(theta_full, lag_jax)
            return jnp.sum((acf_jax - K_model) ** 2)

        vg_fn = jax.jit(jax.value_and_grad(loss_u))
        jax.block_until_ready(vg_fn(jnp.array(u0, dtype=jnp.float64)))

        _gradient_free = method.lower() in ("nelder-mead", "cobyla", "powell")

        if _gradient_free:
            def objective(u_np):
                val, _ = vg_fn(jnp.array(u_np, dtype=jnp.float64))
                v = float(val)
                return 1e30 if not np.isfinite(v) else v
            _kw = dict(method=method,
                       options={"maxiter": maxiter, "xatol": ftol,
                                "fatol": ftol, "disp": disp})
        else:
            def objective(u_np):
                val, grad = vg_fn(jnp.array(u_np, dtype=jnp.float64))
                v, g = float(val), np.asarray(grad, dtype=np.float64)
                if not np.isfinite(v):
                    return 1e30, np.zeros_like(g)
                if not np.all(np.isfinite(g)):
                    return v, np.zeros_like(g)
                return v, g
            _kw = dict(jac=True, method=method,
                       bounds=[(0.0, 1.0)] * len(free_idx),
                       options={"maxiter": maxiter, "ftol": ftol,
                                "gtol": gtol, "disp": disp})

        result = _minimize(objective, u0, **_kw)

        free_best = blo + jnp.array(result.x, dtype=jnp.float64) * brange
        theta_full = self._theta_from_free(
            free_best, free_idx, fixed_idx, fixed_vals)

        self.acf_fit_theta = theta_full
        self._acf_lag_centers = lag_centers
        self._acf_data = acf_data

        theta_dict = {k: float(theta_full[i])
                      for i, k in enumerate(kernel_keys)}
        return theta_dict, result

    def _fit_acf_parallel(self, nopt=10, ncore=None, keys=None,
                          tlags=None, n_bins=50, method="nelder-mead",
                          maxiter=500, ftol=0, gtol=1e-8, disp=False,
                          rng=None):
        from concurrent.futures import ThreadPoolExecutor

        if ncore is None:
            ncore = min(nopt, os.cpu_count() or 1)
        if rng is None:
            rng = np.random.default_rng()

        kernel_keys = list(self.kernel.param_names)
        n_kernel = self._n_kernel
        if keys is None:
            free_keys = list(kernel_keys)
            fb = np.asarray(self.bounds[:n_kernel])
        else:
            free_keys = [k for k in kernel_keys if k in keys]
            idx = [kernel_keys.index(k) for k in free_keys]
            fb = np.asarray(self.bounds[jnp.array(idx)])
        blo, bhi = fb[:, 0], fb[:, 1]

        seeds = rng.integers(0, 2 ** 31, size=nopt)
        starts = []
        for i in range(nopt):
            child = np.random.default_rng(int(seeds[i]))
            u = child.uniform(size=len(free_keys))
            starts.append({k: float(blo[j] + u[j] * (bhi[j] - blo[j]))
                           for j, k in enumerate(free_keys)})

        def _run(theta0_dict):
            return self.fit_acf(theta0=theta0_dict, keys=keys,
                                tlags=tlags, n_bins=n_bins,
                                method=method, maxiter=maxiter,
                                ftol=ftol, gtol=gtol, disp=disp)

        with ThreadPoolExecutor(max_workers=ncore) as pool:
            futures = [pool.submit(_run, s) for s in starts]
            results = [f.result() for f in futures]

        results.sort(key=lambda tr: float(tr[1].fun))
        best_theta, best_result = results[0]
        self.acf_fit_theta = jnp.array(
            [float(best_theta[k]) for k in kernel_keys], dtype=jnp.float64)
        return best_theta, best_result

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_prediction(self, theta=None, n_points=2000, n_sigma=(1, 2),
                        ax=None, data_color="k", model_color="r",
                        show_legend=True, xlim=None, ylim=None,
                        model_label="GP mean", data_label="Data"):
        """Plot GP posterior mean and uncertainty bands over the data."""
        import matplotlib.pyplot as plt

        if theta is not None:
            self._set_theta(theta)

        xpred = np.linspace(float(self.x[0]), float(self.x[-1]), n_points)
        mu, var = self.predict(xpred)
        sigma = np.sqrt(np.maximum(var, 0.0))

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4))

        ax.errorbar(np.asarray(self.x), np.asarray(self.y),
                    yerr=np.asarray(self.yerr),
                    fmt=".", color=data_color, capsize=0, alpha=0.5,
                    label=data_label)
        ax.plot(xpred, mu, color=model_color, lw=1.5, label=model_label)

        alphas = {1: 0.35, 2: 0.18, 3: 0.10}
        for ns in (n_sigma if hasattr(n_sigma, "__iter__") else (n_sigma,)):
            ax.fill_between(xpred, mu - ns * sigma, mu + ns * sigma,
                            color=model_color,
                            alpha=alphas.get(ns, 0.15),
                            label=rf"$\pm{ns}\sigma$")
        ax.set_xlim(xlim or (float(self.x[0]), float(self.x[-1])))
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlabel("Time [days]", fontsize=22)
        ax.set_ylabel("Flux", fontsize=22)
        if show_legend:
            ax.legend()
        return ax

    def plot_acf(self, theta=None, tlags=None, n_bins=50, ax=None,
                 normalize=False, data_color="k", model_color="r",
                 show_legend=True, xlim=None, ylim=None,
                 model_label="Celerite kernel", data_label="Data ACF"):
        """Plot empirical ACF and optionally the celerite kernel."""
        import matplotlib.pyplot as plt

        if tlags is None:
            baseline = float(jnp.max(self.x) - jnp.min(self.x))
            tlags = np.linspace(0, baseline / 2, n_bins + 1)

        lag_centers, acf_data = self.compute_acf(
            tlags=tlags, n_bins=n_bins, normalize=normalize)

        if ax is None:
            _, ax = plt.subplots()

        ax.plot(lag_centers, acf_data, color=data_color, label=data_label)

        if theta is not None:
            self._set_theta(theta)
        lag_fine = np.linspace(0.0, float(tlags[-1]), 300)
        K_model = self.compute_kernel(lag_fine)
        if normalize:
            var = np.var(self.data.y)
            if var > 0:
                K_model = K_model / var
        ax.plot(lag_fine, K_model, color=model_color, label=model_label)

        ax.set_xlim(xlim or (min(tlags), max(tlags)))
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlabel("Time lag [days]", fontsize=22)
        ax.set_ylabel("ACF" if normalize else "Autocovariance", fontsize=22)
        if show_legend:
            ax.legend()
        return ax

    def plot_psd(self, theta=None, n_freq=500, dt_kernel=None, ax=None,
                 data_color="k", model_color="r", show_legend=True,
                 xlim=None, ylim=None,
                 model_label="Celerite PSD", data_label="Data Lomb-Scargle"):
        """Plot empirical PSD and the celerite kernel PSD."""
        import matplotlib.pyplot as plt

        x = np.asarray(self.x)
        resid = np.asarray(self.y) - self.mean_val
        var = float(np.mean(resid ** 2))
        baseline = float(x[-1] - x[0])
        dt_med = float(np.median(np.diff(x)))
        freq_min = 1.0 / baseline
        freq_max = 1.0 / (2.0 * dt_med)

        freqs, psd_data = self.data.compute_psd(
            normalization="psd", n_freq=n_freq,
            freq_min=freq_min, freq_max=freq_max)
        integral = np.trapezoid(psd_data, freqs)
        if integral > 0:
            psd_data = psd_data * var / integral

        if ax is None:
            _, ax = plt.subplots()

        ax.semilogy(freqs, psd_data, color=data_color, lw=0.8,
                    label=data_label)

        if theta is not None:
            self._set_theta(theta)
        omega = 2 * np.pi * freqs
        psd_model = self.kernel.get_psd(omega)
        integral_m = np.trapezoid(psd_model, freqs)
        if integral_m > 0:
            psd_model = psd_model * var / integral_m
        ax.semilogy(freqs, psd_model, color=model_color, lw=1.5,
                    label=model_label)

        ax.set_xlim(xlim or (freq_min, freq_max))
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlabel("Frequency [1/day]", fontsize=22)
        ax.set_ylabel("PSD", fontsize=22)
        if show_legend:
            ax.legend()
        return ax

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _set_theta(self, theta):
        """Set _current_theta from a dict or array."""
        if isinstance(theta, dict):
            self.update_params(theta)
        else:
            self._current_theta = jnp.asarray(theta, dtype=jnp.float64)

    def _result_dict(self, theta_arr):
        return {k: float(theta_arr[i])
                for i, k in enumerate(self.param_keys)}

    def _autosave(self, filename, **arrays):
        if self.save_dir is None:
            return
        path = os.path.join(self.save_dir, filename)
        np.savez(path, **arrays)
        print(f"Saved {filename} -> {path}")
