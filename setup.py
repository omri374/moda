"""Setup for moda (trending topics detection)."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools
import versioneer

INSTALL_REQUIRES = [
    'numpy',
    'pandas',
    'statsmodels',
    'matplotlib'
]
TEST_REQUIRES = [
    # testing and coverage
    'pytest',
    # to be able to run `python setup.py checkdocs`
    'collective.checkdocs', 'pygments'
]

with open('README.rst') as f:
    README = f.read()

setuptools.setup(
    author="Omri Mendels",
    author_email="omri.mendels@microsoft.com",
    name='moda',
    license="MIT",
    description='Trending topics detection',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    long_description=README,
    url='https://github.com/morsh/social-posts-pipeline',
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires=">=3.5",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'test': TEST_REQUIRES + INSTALL_REQUIRES
    },
    # dependency_links=DEPENDENCY_LINKS,
    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
    ],
)
