# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the build directory to the path so we can import the Python module
sys.path.insert(0, os.path.abspath("_build_doxygen"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "libparallelproj"
copyright = "2026, parallelproj team"
author = "parallelproj team"
release = "2.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "breathe",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

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
