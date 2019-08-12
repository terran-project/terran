from setuptools import setup, find_packages

setup(
    name='terran',
    version='0.0.1',

    author='Agustín Azzinnari',
    author_email='me@nagitsu.com',
    url='https://github.com/nagitsu/terran',

    packages=find_packages(),
    install_requires=[
        'click',
    ],

    entry_points="""
        [console_scripts]
        terran=terran:cli
    """,

    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
