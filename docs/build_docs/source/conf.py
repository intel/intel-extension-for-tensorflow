# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../itex/'))
from datetime import datetime
from setuptools_scm import get_version
# See https://pypi.org/project/setuptools-scm/#programmatic-usage
version = get_version(root='../..', relative_to=__file__)
release = version

with open("version.txt", "w") as f:
    f.write(version)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Intel® Extension for TensorFlow*'
author = 'Intel Corporation'
copyright = u'2022-' + str(datetime.now().year) + u' ' + author



# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        'recommonmark',
        'sphinx_markdown_tables',
        'sphinx.ext.coverage',
        'sphinx.ext.autosummary',
        'sphinx_md',
        #'autoapi.extension',
        'sphinx.ext.napoleon',
        'sphinx.ext.githubpages'
        ]


templates_path = ['_templates']

source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'

exclude_patterns = ['build', 'itex', '.github', 'test', 'third_party']


pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    #'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': -1,
    'includehidden': True,
    'titles_only': False
}

html_static_path = ['_static']

def setup(app):
   app.add_css_file("custom.css")
