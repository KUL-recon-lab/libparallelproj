# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from sphinx_gallery.sorting import FileNameSortKey

# Anchor all build-directory paths to conf.py's own location so sphinx-build
# can be invoked from any working directory (e.g. the project root on RTD).
_BUILD_DOXYGEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_build_doxygen")

# Add the build directory to the path so we can import the Python module
sys.path.insert(0, _BUILD_DOXYGEN)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "libparallelproj"
# copyright = "2026, parallelproj team"
# author = "parallelproj team"


def _resolve_release() -> str:
    # single source of truth: the VERSION file at the repo root
    version_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "VERSION"
    )
    if os.path.exists(version_file):
        with open(version_file, encoding="utf-8") as handle:
            version = handle.read().strip()
        if version:
            return version

    return "0.0.0-unknown"


release = _resolve_release()

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "breathe",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "examples/README.rst"]
bibtex_bibfiles = ["refs.bib"]

sphinx_gallery_conf = {
    "examples_dirs": ["examples"],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": r"[\\/]\d{2,3}_.*\.py$",
    "ignore_pattern": r"(^|[\\/])utils\.py$",
    "plot_gallery": True,
    "within_subsection_order": FileNameSortKey,
}

# -- Breathe configuration ---------------------------------------------------
breathe_projects = {"parallelproj": os.path.join(_BUILD_DOXYGEN, "docs", "xml")}
breathe_default_project = "parallelproj"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_logo = "_static/logo.png"

# Theme options
html_theme_options = {
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view", "edit"],
}

suppress_warnings = ["config.cache"]
