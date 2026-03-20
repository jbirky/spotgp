"""
distributions.py — Parameter distributions for hierarchical kernel modeling.

When a kernel parameter is a ``ParameterDistribution`` instead of a plain
float, the kernel marginalizes (integrates) over it.  A fixed float is
internally wrapped as a ``DeltaDistribution`` so all code paths are uniform.

The ``as_distribution`` helper performs this wrapping:

>>> as_distribution(0.01)          # DeltaDistribution(0.01)
>>> as_distribution(Uniform(1, 5)) # passes through unchanged
"""

import numpy as np

__all__ = [
    "ParameterDistribution",
    "DeltaDistribution",
    "UniformDistribution",
    "GaussianDistribution",
    "LogNormalDistribution",
    "as_distribution",
    "is_distributed",
]


class ParameterDistribution:
    """
    Base class for a distribution over a scalar kernel parameter.

    Subclasses must implement ``support`` and ``__call__``.
    """

    @property
    def support(self) -> tuple:
        """(min, max) range of the parameter."""
        raise NotImplementedError

    def __call__(self, x: float) -> float:
        """Unnormalized probability density at x."""
        raise NotImplementedError

    @property
    def mean(self) -> float:
        """Mean of the distribution. Default: numerical via quadrature."""
        return self.expectation(lambda x: x)

    def expectation(self, func, n_quad=32):
        """
        Compute E[func(x)] under this distribution via Gauss-Legendre
        quadrature over ``self.support``.

        Parameters
        ----------
        func : callable
            Function to take the expectation of.
        n_quad : int
            Number of quadrature points (default 32).

        Returns
        -------
        float
        """
        lo, hi = self.support
        nodes, weights = np.polynomial.legendre.leggauss(n_quad)
        x = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * weights
        pdf_vals = np.array([self(float(xi)) for xi in x])
        func_vals = np.array([func(float(xi)) for xi in x])
        norm = np.sum(pdf_vals * w)
        if norm == 0:
            return 0.0
        return float(np.sum(pdf_vals * func_vals * w) / norm)

    def sample(self, n, rng=None):
        """
        Draw n samples via inverse-CDF on the quadrature grid.
        For quick prototyping; not intended for MCMC.
        """
        if rng is None:
            rng = np.random.default_rng()
        lo, hi = self.support
        x_grid = np.linspace(lo, hi, 1000)
        pdf = np.array([self(xi) for xi in x_grid])
        cdf = np.cumsum(pdf)
        cdf = cdf / cdf[-1]
        u = rng.uniform(size=n)
        return np.interp(u, cdf, x_grid)

    def sympy_pdf(self):
        """
        Return the sympy expression for the PDF.

        Subclasses should override to provide their analytic form.
        The base implementation returns None (numerical only).

        Returns
        -------
        sympy.Expr or None
        """
        return None

    def get_sympy(self, display=True, status=None, var_name="x"):
        """
        Display the sympy expression for the distribution PDF.

        Parameters
        ----------
        display : bool
            If True (default), render/print the expression.
        status : str or None, optional
            If provided, appended to the class name header in brackets.
        var_name : str
            Name of the variable (default "x").

        Returns
        -------
        dict
            ``{"pdf": expr_or_None}``
        """
        expr = self.sympy_pdf()
        exprs = {"pdf": expr}

        if display:
            try:
                import sympy as sp
            except ImportError:
                raise ImportError(
                    "sympy is required for get_sympy(). "
                    "Install with: pip install sympy")
            rhs = r"\text{[numerical]}" if expr is None else sp.latex(expr)
            status_tag = r" \text{[" + status + r"]}" if status else ""
            header = r"\textbf{" + type(self).__name__ + r"}" + status_tag
            try:
                from IPython.display import display as ipy_display, Math
                ipy_display(Math(header))
                ipy_display(Math(f"p({var_name}) = " + rhs))
            except ImportError:
                status_str = f" [{status}]" if status else ""
                print(f"{type(self).__name__}{status_str}")
                print(f"  $p({var_name}) = {rhs}$")

        return exprs

    def __repr__(self):
        return f"{type(self).__name__}(support={self.support})"


class DeltaDistribution(ParameterDistribution):
    """
    Degenerate distribution at a fixed value (Dirac delta).

    Wraps a plain float so that all code paths can treat parameters
    uniformly as distributions.  ``expectation(func)`` returns
    ``func(value)`` with no quadrature overhead.
    """

    def __init__(self, value):
        self._value = float(value)

    @property
    def support(self):
        return (self._value, self._value)

    def __call__(self, x):
        return 1.0

    @property
    def mean(self):
        return self._value

    def expectation(self, func, n_quad=32):
        return func(self._value)

    def sympy_pdf(self):
        try:
            import sympy as sp
        except ImportError:
            return None
        x = sp.Symbol("x")
        return sp.DiracDelta(x - sp.Float(self._value))

    def __repr__(self):
        return f"DeltaDistribution({self._value})"


class UniformDistribution(ParameterDistribution):
    """
    Uniform distribution over [lo, hi].

    Parameters
    ----------
    lo, hi : float
        Lower and upper bounds.
    """

    def __init__(self, lo, hi):
        self._lo = float(lo)
        self._hi = float(hi)

    @property
    def support(self):
        return (self._lo, self._hi)

    def __call__(self, x):
        return 1.0

    @property
    def mean(self):
        return 0.5 * (self._lo + self._hi)

    def sympy_pdf(self):
        try:
            import sympy as sp
        except ImportError:
            return None
        return sp.Rational(1, 1) / sp.Float(self._hi - self._lo)

    def __repr__(self):
        return f"UniformDistribution({self._lo}, {self._hi})"


class GaussianDistribution(ParameterDistribution):
    """
    Truncated Gaussian distribution.

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation.
    clip_sigma : float
        Number of sigma for truncation (default 4).
    """

    def __init__(self, mu, sigma, clip_sigma=4.0):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.clip_sigma = float(clip_sigma)

    @property
    def support(self):
        lo = self.mu - self.clip_sigma * self.sigma
        hi = self.mu + self.clip_sigma * self.sigma
        return (lo, hi)

    def __call__(self, x):
        return np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)

    @property
    def mean(self):
        return self.mu

    def sympy_pdf(self):
        try:
            import sympy as sp
        except ImportError:
            return None
        x = sp.Symbol("x")
        mu = sp.Float(self.mu)
        sigma = sp.Float(self.sigma)
        return sp.exp(sp.Rational(-1, 2) * ((x - mu) / sigma) ** 2) / (
            sigma * sp.sqrt(2 * sp.pi))

    def __repr__(self):
        return f"GaussianDistribution(mu={self.mu}, sigma={self.sigma})"


class LogNormalDistribution(ParameterDistribution):
    """
    Log-normal distribution (positive-valued parameters).

    If X ~ LogNormal(mu, sigma), then log(X) ~ Normal(mu, sigma).

    Parameters
    ----------
    mu : float
        Mean of log(X).
    sigma : float
        Std of log(X).
    clip_sigma : float
        Truncation in log-space (default 4).
    """

    def __init__(self, mu, sigma, clip_sigma=4.0):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.clip_sigma = float(clip_sigma)

    @property
    def support(self):
        lo = np.exp(self.mu - self.clip_sigma * self.sigma)
        hi = np.exp(self.mu + self.clip_sigma * self.sigma)
        return (lo, hi)

    def __call__(self, x):
        if x <= 0:
            return 0.0
        return np.exp(-0.5 * ((np.log(x) - self.mu) / self.sigma) ** 2) / x

    @property
    def mean(self):
        return np.exp(self.mu + 0.5 * self.sigma ** 2)

    def sympy_pdf(self):
        try:
            import sympy as sp
        except ImportError:
            return None
        x = sp.Symbol("x", positive=True)
        mu = sp.Float(self.mu)
        sigma = sp.Float(self.sigma)
        return sp.exp(sp.Rational(-1, 2) * ((sp.log(x) - mu) / sigma) ** 2) / (
            x * sigma * sp.sqrt(2 * sp.pi))

    def __repr__(self):
        return f"LogNormalDistribution(mu={self.mu}, sigma={self.sigma})"


def as_distribution(value):
    """
    Wrap a value as a ParameterDistribution if it isn't one already.

    Parameters
    ----------
    value : float or ParameterDistribution
        If float, returns a DeltaDistribution.
        If already a ParameterDistribution, returns it unchanged.

    Returns
    -------
    ParameterDistribution
    """
    if isinstance(value, ParameterDistribution):
        return value
    return DeltaDistribution(float(value))


def is_distributed(value):
    """Return True if value is a non-degenerate distribution."""
    return isinstance(value, ParameterDistribution) and not isinstance(value, DeltaDistribution)
