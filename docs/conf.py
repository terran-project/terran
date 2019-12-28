project = 'Terran'
copyright = '2019, Agustín Azzinnari'
author = 'Agustín Azzinnari'

release = '0.0.1'

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
