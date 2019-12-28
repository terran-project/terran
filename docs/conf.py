import pkg_resources
import sys

project = 'Terran'
copyright = '2019, Agustín Azzinnari'
author = 'Agustín Azzinnari'

try:
    release = pkg_resources.get_distribution('terran').version
except pkg_resources.DistributionNotFound:
    print('Terran must be installed to build the documentation.')
    sys.exit(1)

if 'dev' in release:
    # Trim everything after `dev`, if present.
    release = ''.join(release.partition('dev')[:2])

# The short X.Y version.
version = '.'.join(release.split('.')[:2])

master_doc = 'index'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
]

autoclass_content = 'both'

# Mock dependencies that have C-extensions.
autodoc_mock_imports = [
    'cairo',
    'lycon'
]

templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']
