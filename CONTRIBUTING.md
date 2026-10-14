# Contributing to `spotgp`

Thank you for your interest in `spotgp`. Bug reports, feature requests,
documentation fixes, and new physical components (envelopes, visibility
functions, latitude distributions, kernel terms) are all welcome.

## Reporting problems and asking for help

- **Bugs and feature requests:** open an issue at
  <https://github.com/jbirky/spotgp/issues>. For bugs, please include the
  `spotgp` and `jax` versions (`python -c "import spotgp, jax; print(spotgp.__version__, jax.__version__)"`),
  your operating system, a minimal script that reproduces the problem, and the
  full traceback.
- **Usage questions:** open an issue with the `question` label, or start a
  thread under <https://github.com/jbirky/spotgp/discussions>. The
  [tutorials](https://spotgp.readthedocs.io) cover most common workflows and are
  the best place to start.
- **Security or private concerns:** email the maintainer (see `pyproject.toml`).

## Development setup

```bash
git clone https://github.com/jbirky/spotgp.git
cd spotgp
conda create -n spotgp-dev python=3.12 && conda activate spotgp-dev
pip install -e ".[pgm,jaxopt,docs]"
pip install pytest pytest-cov
```

Run the test suite (CPU is sufficient; the tests set `JAX_PLATFORMS=cpu`):

```bash
JAX_PLATFORMS=cpu pytest tests/ -v
```

Build the documentation locally:

```bash
cd docs && make html   # output in docs/_build/html
```

## Making changes

1. Fork the repository and create a branch from `main`
   (`git checkout -b feature/my-change`).
2. Keep changes focused. If you are adding a new physical component, follow the
   extension contracts in the
   [custom components](https://spotgp.readthedocs.io/en/latest/custom_components.html)
   page: subclass the relevant base class (`EnvelopeFunction`,
   `VisibilityFunction`, `LatitudeDistributionFunction`, or `Term`), keep all
   numerical code in `jax.numpy` so it stays differentiable, and where a
   closed-form expression exists, provide a `get_sympy()` method so it can be
   checked symbolically.
3. Add tests under `tests/` for any new behaviour. For a new kernel component,
   the minimum is (a) agreement between any analytic override and the numerical
   fallback (see `EnvelopeFunction.check_functions()`), and (b) agreement
   between `AnalyticKernel` and `NumericalKernel` for a representative parameter
   set.
4. Update or add a tutorial under `docs/tutorials/` if the change is
   user-facing, and add a line to the API listing in `docs/api/` if you add a
   public class or function.
5. Make sure `pytest` passes on CPU and that the docs build without warnings.
6. Open a pull request against `main` describing the motivation and the change.
   Continuous integration runs the test suite on Linux and macOS for Python
   3.10–3.12; please address any failures.

## Style

- Follow the conventions of the surrounding code (NumPy-style docstrings,
  `jax.numpy` for anything that may be traced, explicit parameter names rather
  than positional hyperparameter vectors in public APIs).
- Public classes and functions should be exported from `spotgp/__init__.py` and
  listed in `__all__`.
- Keep commit messages short and descriptive.

## Code of conduct

Please be respectful and constructive in issues, discussions, and reviews. We
follow the spirit of the
[Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/);
unacceptable behaviour can be reported to the maintainer.

## Licence

By contributing you agree that your contributions will be licensed under the
MIT licence that covers the project (see `LICENSE`).
