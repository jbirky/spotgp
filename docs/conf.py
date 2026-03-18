"""Sphinx configuration for spotgp documentation."""

import os
import sys

# Add source directories to path
sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))
sys.path.insert(0, os.path.abspath(".."))

# Mock imports for packages that are heavy / unavailable on RTD
autodoc_mock_imports = ["jax", "jaxopt", "blackjax"]

# -- Project information -----------------------------------------------------

project = "spotgp"
copyright = "2026, Jessica Birky"
author = "Jessica Birky"
version = "0.1.0"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for autodoc -----------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
highlight_language = "python3"
pygments_style     = "friendly"

html_theme_options = {
    "repository_url": "https://github.com/jbirky/spotgp",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_source_button": True,
    "use_edit_page_button": True,
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "logo": {
        "image_light": "_static/spotgp_logo_light.png",
        "image_dark": "_static/spotgp_logo_dark.png",
    },
}

# -- nbsphinx ----------------------------------------------------------------

# Embed the notebook filename as a hidden meta tag so JS can build the
# download link without hard-coding paths.
nbsphinx_prolog = r"""
.. raw:: html

    <meta name="notebook-source" content="{{ env.docname.split('/')|last }}.ipynb">
"""

html_js_files = ["notebook_download.js"]

# -- Intersphinx mapping -----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}
