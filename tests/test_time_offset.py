"""Prediction times are in the frame of the times given to GPSolver.

``TimeSeriesData`` shifts the time axis to start at 0 (``x_offset``).  The
prediction methods must undo that shift, or a light curve that does not
start at t = 0 (any real Kepler/TESS series) is predicted at the wrong times.
Every other test uses data starting at 0, where the shift is zero.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from spotgp import (GPSolver, JitterTerm, KernelSum, SpotEvolutionModel,
                    SpotTerm, TrapezoidSymmetricEnvelope, VisibilityFunction)

T0 = 1234.5          # start time of the shifted copy [days]


def _model():
    env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
    vis = VisibilityFunction(peq=4.0, kappa=0.1, inc=np.pi / 3)
    return SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.01)


@pytest.fixture
def pair():
    """The same light curve on t in [0, 30] and on t in [T0, T0 + 30]."""
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0.0, 30.0, 120))
    y = 0.01 * np.sin(2 * np.pi * x / 4.0) + 1e-3 * rng.standard_normal(x.size)
    yerr = np.full(x.size, 1e-3)
    gp0 = GPSolver(x, y, yerr, _model(), matrix_solver="cholesky_full")
    gp1 = GPSolver(x + T0, y, yerr, _model(), matrix_solver="cholesky_full")
    gp1.input_times = x + T0     # kept for the tests, independent of x_offset
    return gp0, gp1


def test_x_offset_is_first_input_time(pair):
    gp0, gp1 = pair
    assert gp0.x_offset == pytest.approx(float(gp0.x[0] + gp0.x_offset))
    assert gp1.x_offset == pytest.approx(T0 + float(gp0.x_offset))
    np.testing.assert_allclose(np.asarray(gp1.x), np.asarray(gp0.x), atol=1e-9)


@pytest.mark.parametrize("return_cov", [False, True])
def test_predict_uses_input_frame(pair, return_cov):
    gp0, gp1 = pair
    xp = np.linspace(-2.0, 32.0, 25)
    mu0, v0 = gp0.predict(xp, return_cov=return_cov)
    mu1, v1 = gp1.predict(xp + T0, return_cov=return_cov)
    np.testing.assert_allclose(mu1, mu0, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(v1, v0, rtol=1e-8, atol=1e-14)


def test_predict_at_uses_input_frame(pair):
    gp0, gp1 = pair
    xp = np.linspace(-2.0, 32.0, 25)
    mu0, v0 = gp0.predict_at(gp0.theta0, xp)
    mu1, v1 = gp1.predict_at(gp1.theta0, xp + T0)
    np.testing.assert_allclose(mu1, mu0, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(v1, v0, rtol=1e-8, atol=1e-14)


def test_predict_at_reproduces_data_at_input_times(pair):
    # At the training times, a noise-dominated GP mean tracks the data
    # closely; at times shifted by x_offset it would not.
    _, gp1 = pair
    mu, _ = gp1.predict_at(gp1.theta0, gp1.input_times)
    resid = np.asarray(gp1.y) - mu
    assert np.std(resid) < 0.5 * np.std(np.asarray(gp1.y))


def test_composite_kernel_predict_at_uses_input_frame():
    rng = np.random.default_rng(5)
    x = np.sort(rng.uniform(0.0, 20.0, 80))
    y = 0.01 * np.sin(2 * np.pi * x / 4.0) + 1e-3 * rng.standard_normal(x.size)
    kern = lambda: KernelSum(SpotTerm(_model(), prefix="spot"),
                             JitterTerm(sigma_j=1e-3, prefix="jit"))
    gp0 = GPSolver(x, y, 1e-3, kern(), matrix_solver="cholesky_full")
    gp1 = GPSolver(x + T0, y, 1e-3, kern(), matrix_solver="cholesky_full")
    xp = np.linspace(0.0, 20.0, 15) + 0.013    # off the training times
    np.testing.assert_allclose(gp1.predict_at(gp1.theta0, xp + T0)[0],
                               gp0.predict_at(gp0.theta0, xp)[0],
                               rtol=1e-8, atol=1e-12)


def test_sample_lightcurves_grid_in_input_frame(pair):
    _, gp1 = pair
    xpred, samples = gp1.sample_lightcurves(n_samples=2, n_points=50,
                                            source="posterior",
                                            rng=np.random.default_rng(0))
    x_in = np.asarray(gp1.x) + gp1.x_offset
    assert xpred[0] == pytest.approx(x_in[0])
    assert xpred[-1] == pytest.approx(x_in[-1])
    assert samples.shape == (2, 50)


def test_plot_prediction_in_input_frame(pair):
    _, gp1 = pair
    ax = gp1.plot_prediction(n_points=50, show_legend=False)
    lo, hi = ax.get_xlim()
    x_in = np.asarray(gp1.x) + gp1.x_offset
    assert lo == pytest.approx(x_in[0]) and hi == pytest.approx(x_in[-1])
    plt.close(ax.figure)
