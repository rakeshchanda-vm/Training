# Configuration file for the Sphinx documentation builder.
import os
import sys
from pathlib import Path

# Add the project root to sys.path
# From docs/source/, ../../ takes you to the root directory where 'andromeda' package resides.
sys.path.insert(0, os.path.abspath("../../"))
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Andromeda"
copyright = "2025, Jefferson Nelsson"
author = "Jefferson Nelsson"
release = "v1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


# Suppress warnings from external library docstrings
suppress_warnings = [
    "autosummary",
    "autodoc.import_object",
    "docutils",
]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # If you use Google or NumPy style docstrings
    "sphinx_copybutton",
]
templates_path = ["_templates"]
exclude_patterns = []

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_permalinks_icon = '<span>#</span>'
html_theme = "shibuya"
html_logo = "_static/vm.png"
html_baseurl = "https://docs.valuemomentum.studio/andromeda/"
html_copy_source = True
html_theme_options = {
    "navigation_with_keys": True,
    "accent_color": "green",
    "page_layout": "default",
    "show_ai_links": True,
    "open_in_chatgpt": True,
    "open_in_claude": True,
    "open_in_perplexity": True,
    "color_mode": "dark",
    "github_url": "https://github.com/AI-Emerging-Tech/ET-Agentify",
    "nav_links": [
        {
            "title": "AI & Emerging Tech",
            "url": "https://www.valuemomentum.studio",
            "external": True
        },
        {
            "title": "Andromeda Documentation",
            "url": "https://docs.valuemomentum.studio/andromeda"
        },
    ]
}

html_static_path = ["_static"]

# html_css_files = [
#     "custom.css",
# ]
