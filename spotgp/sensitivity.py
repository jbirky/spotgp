import numpy as np
from itertools import combinations

__all__ = ["sobol_indices"]


def _saltelli_sample(bounds, n):
    """
    Generate Saltelli (2002) quasi-random sample matrices A, B, AB.

    Parameters
    ----------
    bounds : list of (lo, hi) tuples, length k
    n : base sample size (total evaluations = n * (k + 2))

    Returns
    -------
    A, B : ndarray, shape (n, k)  — base sample matrices
    AB   : ndarray, shape (k, n, k) — AB[i] has column i from B, rest from A
    """
    k = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    # Two independent Sobol-like random matrices (use uniform for simplicity)
    rng = np.random.default_rng()
    A = lo + (hi - lo) * rng.random((n, k))
    B = lo + (hi - lo) * rng.random((n, k))

    AB = np.empty((k, n, k))
    for i in range(k):
        AB[i] = A.copy()
        AB[i, :, i] = B[:, i]

    return A, B, AB


def sobol_indices(func, bounds, n=1024, param_names=None):
    """
    Estimate first-order and total-order Sobol sensitivity indices.

    Uses the Saltelli estimator:
        S_i  = V[E[Y|X_i]] / V[Y]   (first-order)
        ST_i = E[V[Y|X_~i]] / V[Y]  (total-order)

    Parameters
    ----------
    func : callable
        Scalar function f(x) where x has shape (k,).
    bounds : list of (lo, hi) tuples
        Parameter ranges, length k.
    n : int
        Base sample size. Total model evaluations = n * (k + 2).
    param_names : list of str, optional
        Names for each parameter.

    Returns
    -------
    results : dict with keys:
        'S1'     : ndarray (k,) — first-order indices
        'ST'     : ndarray (k,) — total-order indices
        'S1_var' : ndarray (k,) — bootstrap variance of S1
        'ST_var' : ndarray (k,) — bootstrap variance of ST
        'names'  : list of str
    """
    k = len(bounds)
    if param_names is None:
        param_names = [f"x{i}" for i in range(k)]

    A, B, AB = _saltelli_sample(bounds, n)

    # Evaluate model
    fA  = np.array([func(A[j])     for j in range(n)])
    fB  = np.array([func(B[j])     for j in range(n)])
    fAB = np.array([[func(AB[i][j]) for j in range(n)] for i in range(k)])

    # Total variance (Jansen estimator base)
    f0   = 0.5 * (fA.mean() + fB.mean())
    varY = np.var(np.concatenate([fA, fB]), ddof=1)

    # Saltelli (2010) estimators
    S1  = np.empty(k)
    ST  = np.empty(k)
    for i in range(k):
        S1[i]  = np.mean(fB * (fAB[i] - fA)) / varY      # first-order
        ST[i]  = np.mean((fA - fAB[i]) ** 2) / (2 * varY) # total-order

    # Bootstrap variance estimate (200 resamples)
    n_boot = 200
    rng = np.random.default_rng()
    S1_boot = np.empty((n_boot, k))
    ST_boot = np.empty((n_boot, k))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        fA_b  = fA[idx]
        fB_b  = fB[idx]
        fAB_b = fAB[:, idx]
        varY_b = np.var(np.concatenate([fA_b, fB_b]), ddof=1)
        for i in range(k):
            S1_boot[b, i]  = np.mean(fB_b * (fAB_b[i] - fA_b)) / varY_b
            ST_boot[b, i]  = np.mean((fA_b - fAB_b[i]) ** 2) / (2 * varY_b)

    return dict(
        S1=S1,
        ST=ST,
        S1_var=S1_boot.var(axis=0),
        ST_var=ST_boot.var(axis=0),
        names=param_names,
    )
