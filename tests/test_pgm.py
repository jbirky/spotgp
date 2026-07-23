"""Tests for spotgp.pgm — probabilistic graphical model visualization."""

import numpy as np
import pytest

from spotgp import GPSolver
from spotgp.pgm import (
    PGModelVis, _strip_log_prefix,
    ROTATION_KEYS, ENVELOPE_KEYS, LATITUDE_KEYS, AMPLITUDE_KEYS,
    MULTIBAND_KEYS, NOISE_KEYS,
    PARAM_LABELS, PARAM_DESCRIPTIONS,
)


@pytest.fixture
def solver(default_hparam, synthetic_data):
    x, y, yerr = synthetic_data
    return GPSolver(x, y, yerr, default_hparam, n_lat=8)


class TestStripLogPrefix:
    def test_strips_log_prefix(self):
        assert _strip_log_prefix("log_sigma_k") == "sigma_k"

    def test_leaves_plain_key_untouched(self):
        assert _strip_log_prefix("peq") == "peq"

    def test_only_strips_leading_prefix(self):
        """A key merely containing 'log_' should not be truncated."""
        assert _strip_log_prefix("tau_log_x") == "tau_log_x"


class TestLabelTables:
    """Every categorized parameter must have a label and a description."""

    ALL_KEYS = (ROTATION_KEYS | ENVELOPE_KEYS | LATITUDE_KEYS
                | AMPLITUDE_KEYS | MULTIBAND_KEYS | NOISE_KEYS)

    def test_every_key_has_latex_label(self):
        missing = self.ALL_KEYS - set(PARAM_LABELS)
        assert not missing, f"missing LaTeX labels: {sorted(missing)}"

    def test_every_key_has_description(self):
        missing = self.ALL_KEYS - set(PARAM_DESCRIPTIONS)
        assert not missing, f"missing descriptions: {sorted(missing)}"

    def test_category_sets_are_disjoint(self):
        """A parameter must not fall into two groups at once."""
        sets = [ROTATION_KEYS, ENVELOPE_KEYS, LATITUDE_KEYS,
                AMPLITUDE_KEYS, MULTIBAND_KEYS, NOISE_KEYS]
        for i, a in enumerate(sets):
            for b in sets[i + 1:]:
                assert not (a & b), f"overlap: {sorted(a & b)}"


class TestCategorization:
    def test_rotation_params_detected(self, solver):
        pgm = PGModelVis(solver)
        assert set(pgm.rotation_params) == {"peq", "kappa", "inc"}

    def test_envelope_params_detected(self, solver):
        pgm = PGModelVis(solver)
        assert set(pgm.envelope_params) == {"lspot", "tau_spot"}

    def test_amplitude_params_detected(self, solver):
        pgm = PGModelVis(solver)
        assert pgm.amplitude_params == ["sigma_k"]

    def test_no_noise_param_when_not_fit(self, solver):
        assert PGModelVis(solver).noise_params == []

    def test_noise_param_when_fit_sigma_n(self, default_hparam, synthetic_data):
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, default_hparam, n_lat=8, fit_sigma_n=True)
        assert PGModelVis(gp).noise_params == ["sigma_n"]

    def test_categorization_covers_all_param_keys(self, solver):
        """No free parameter should be silently dropped from the diagram."""
        pgm = PGModelVis(solver)
        grouped = set(pgm.rotation_params + pgm.envelope_params
                      + pgm.latitude_params + pgm.amplitude_params
                      + pgm.multiband_params + pgm.noise_params)
        assert grouped == set(pgm._physical_keys)

    def test_log_space_keys_are_categorized(self, solver):
        """Solvers sampling in log space still group correctly."""
        pgm = PGModelVis(solver)
        pgm._physical_keys = [_strip_log_prefix(k) for k in
                              ("peq", "log_sigma_k", "log_tau_spot")]
        pgm._categorize()
        assert pgm.amplitude_params == ["sigma_k"]
        assert pgm.envelope_params == ["tau_spot"]

    def test_not_multiband_for_plain_solver(self, solver):
        assert PGModelVis(solver).is_multiband is False


class TestGroups:
    def test_group_order(self, solver):
        groups = PGModelVis(solver)._build_groups()
        assert [g["name"] for g in groups] == [
            "rotation", "envelope", "amplitude"]

    def test_amplitude_group_has_no_intermediate(self, solver):
        groups = {g["name"]: g for g in PGModelVis(solver)._build_groups()}
        assert groups["amplitude"]["intermediate"] is None

    def test_rotation_group_targets_kernel(self, solver):
        groups = {g["name"]: g for g in PGModelVis(solver)._build_groups()}
        assert groups["rotation"]["target"] == "K"
        assert groups["rotation"]["intermediate"] == "V"

    def test_noise_group_targets_data(self, default_hparam, synthetic_data):
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, default_hparam, n_lat=8, fit_sigma_n=True)
        groups = {g["name"]: g for g in PGModelVis(gp)._build_groups()}
        assert groups["noise"]["target"] == "y_i"

    def test_empty_groups_are_omitted(self, solver):
        """Latitude nodes are absent when no latitude params are free."""
        names = [g["name"] for g in PGModelVis(solver)._build_groups()]
        assert "latitude" not in names
        assert "multiband" not in names


class TestLayout:
    def test_required_nodes_present(self, solver):
        pgm = PGModelVis(solver)
        pos, _ = pgm._compute_layout(pgm._build_groups())
        for node in ("K", "x_i", "y_i", "sigma_obs"):
            assert node in pos

    def test_every_param_gets_a_position(self, solver):
        pgm = PGModelVis(solver)
        pos, _ = pgm._compute_layout(pgm._build_groups())
        for key in pgm._physical_keys:
            assert key in pos

    def test_layers_are_vertically_ordered(self, solver):
        """Parameters sit above intermediates, above the kernel, above data."""
        pgm = PGModelVis(solver)
        pos, _ = pgm._compute_layout(pgm._build_groups())
        assert pos["peq"][1] > pos["V"][1] > pos["K"][1] > pos["y_i"][1]

    def test_params_within_a_row_do_not_overlap(self, solver):
        pgm = PGModelVis(solver)
        pos, _ = pgm._compute_layout(pgm._build_groups())
        xs = sorted(pos[k][0] for k in pgm._physical_keys)
        assert all(b - a > 0 for a, b in zip(xs, xs[1:]))

    def test_group_center_matches_its_params(self, solver):
        pgm = PGModelVis(solver)
        pos, meta = pgm._compute_layout(pgm._build_groups())
        for gm in meta:
            xs = [pos[p][0] for p in gm["params"]]
            assert gm["center_x"] == pytest.approx(sum(xs) / len(xs))

    def test_intermediate_sits_above_its_group_center(self, solver):
        pgm = PGModelVis(solver)
        pos, meta = pgm._compute_layout(pgm._build_groups())
        for gm in meta:
            if gm.get("intermediate"):
                assert pos[gm["intermediate"]][0] == pytest.approx(
                    gm["center_x"])


class TestRender:
    """Rendering requires the optional 'daft' dependency."""

    def test_returns_figure(self, solver):
        pytest.importorskip("daft")
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        assert isinstance(PGModelVis(solver).render(), Figure)

    def test_render_with_legend(self, solver):
        pytest.importorskip("daft")
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        fig = PGModelVis(solver).render(show_legend=True)
        assert isinstance(fig, Figure)

    def test_render_honors_dpi(self, solver):
        pytest.importorskip("daft")
        import matplotlib
        matplotlib.use("Agg")
        fig = PGModelVis(solver).render(dpi=200)
        assert fig.dpi == pytest.approx(200)

    def test_render_with_sigma_n(self, default_hparam, synthetic_data):
        pytest.importorskip("daft")
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, default_hparam, n_lat=8, fit_sigma_n=True)
        assert isinstance(PGModelVis(gp).render(), Figure)


class TestSolverPlotPGM:
    """GPSolver.plot_pgm is a thin delegator to PGModelVis(self).render()."""

    def test_plot_pgm_returns_figure(self, solver):
        pytest.importorskip("daft")
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        assert isinstance(solver.plot_pgm(), Figure)

    def test_plot_pgm_forwards_kwargs(self, solver):
        pytest.importorskip("daft")
        import matplotlib
        matplotlib.use("Agg")
        fig = solver.plot_pgm(dpi=200, show_legend=True)
        assert fig.dpi == pytest.approx(200)
