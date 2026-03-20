"""
latitude.py — Latitude distribution functions for starspot models.

LatitudeDistributionFunction defines the probability density p(phi) over
stellar latitude, controlling where spots are placed and how the kernel
is integrated over latitude.
"""
from __future__ import annotations

import numpy as np

__all__ = ["LatitudeDistributionFunction"]


class LatitudeDistributionFunction:
    """
    Base class for starspot latitude distributions.

    Defines the probability density p(phi) over stellar latitude phi and
    the latitude range over which spots are placed.

    The default implementation is a uniform distribution over
    [-pi/2, pi/2].  To define a custom distribution, subclass this class
    and override ``__call__`` and optionally ``lat_range``.

    Examples
    --------
    Equatorial band (spots confined to |phi| < 30 deg):

    >>> class EquatorialBand(LatitudeDistributionFunction):
    ...     @property
    ...     def lat_range(self):
    ...         return (-np.pi / 6, np.pi / 6)
    ...     def __call__(self, phi):
    ...         return 1.0

    Gaussian centred on the equator:

    >>> class GaussianLatitude(LatitudeDistributionFunction):
    ...     def __init__(self, sigma=np.pi / 6):
    ...         self.sigma = sigma
    ...     def __call__(self, phi):
    ...         return np.exp(-0.5 * (phi / self.sigma) ** 2)
    """

    @property
    def lat_range(self) -> tuple:
        """(min, max) latitude in radians."""
        return (-np.pi / 2, np.pi / 2)

    def __call__(self, phi: float) -> float:
        """
        Unnormalized probability density at latitude phi.

        Normalization is handled internally by the kernel integrator.

        Parameters
        ----------
        phi : float
            Stellar latitude [radians].

        Returns
        -------
        float
            Relative probability density at phi.
        """
        return 1.0

    def sympy_pdf(self):
        """
        Return the sympy expression for the latitude PDF p(phi).

        Subclasses should override this to provide their analytic form.
        The base implementation returns 1 (uniform distribution).

        Returns
        -------
        sympy.Expr or None
            Sympy expression for p(phi), or None if no analytic form exists.
        """
        try:
            import sympy as sp
        except ImportError:
            raise ImportError(
                "sympy is required for get_sympy(). "
                "Install with: pip install sympy")
        return sp.Integer(1)

    def get_sympy(self, display=True, status=None):
        """
        Display the sympy expression for the latitude PDF p(phi).

        Requires sympy (``pip install sympy``).

        Parameters
        ----------
        display : bool, optional
            If True (default), render equations as formatted LaTeX in a
            Jupyter notebook (via IPython.display) or print them as LaTeX
            strings in a plain terminal.
        status : str or None, optional
            If provided, appended to the class name header in brackets,
            e.g. ``"default"`` renders as
            ``LatitudeDistributionFunction [default]``.

        Returns
        -------
        dict
            ``{"pdf": expr_or_None}``
        """
        try:
            import sympy as sp
        except ImportError:
            raise ImportError(
                "sympy is required for get_sympy(). "
                "Install with: pip install sympy")

        expr = self.sympy_pdf()
        exprs = {"pdf": expr}

        if display:
            rhs = r"\text{[numerical]}" if expr is None else sp.latex(expr)
            status_tag = r" \text{[" + status + r"]}" if status else ""
            header = r"\textbf{" + type(self).__name__ + r"}" + status_tag
            try:
                from IPython.display import display as ipy_display, Math
                ipy_display(Math(header))
                ipy_display(Math(r"p(\phi) = " + rhs))
            except ImportError:
                status_str = f" [{status}]" if status else ""
                print(f"{type(self).__name__}{status_str}")
                print(f"  $p(\\phi) = {rhs}$")

        return exprs

    def __repr__(self) -> str:
        return (f"{type(self).__name__}("
                f"lat_range=[{self.lat_range[0]:.3f}, {self.lat_range[1]:.3f}])")
