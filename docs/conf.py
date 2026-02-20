# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
import sys
from typing import Optional

# Add the build directory to the path so we can import the Python module
sys.path.insert(0, os.path.abspath("_build_doxygen"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "libparallelproj"
# copyright = "2026, parallelproj team"
# author = "parallelproj team"


def _read_version_from_cmake_config(config_path: str) -> Optional[str]:
    if not os.path.exists(config_path):
        return None

    with open(config_path, encoding="utf-8") as handle:
        content = handle.read()

    match = re.search(r"set\(PACKAGE_VERSION\s+\"([^\"]+)\"\)", content)
    if match:
        return match.group(1)

    return None


def _resolve_release() -> str:
    env_release = os.environ.get("PARALLELPROJ_RELEASE")
    if env_release:
        return env_release

    cmake_config_version = os.path.abspath(
        os.path.join("_build_doxygen", "parallelprojConfigVersion.cmake")
    )
    cmake_release = _read_version_from_cmake_config(cmake_config_version)
    if cmake_release:
        return cmake_release

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
}

# -- Breathe configuration ---------------------------------------------------
breathe_projects = {"parallelproj": "_build_doxygen/docs/xml"}
breathe_default_project = "parallelproj"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"

# Theme options
html_theme_options = {
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view", "edit"],
}
