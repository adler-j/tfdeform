"""Setup script for tfdeform.

Installation command::

    pip install [--user] [-e] .
"""

from __future__ import print_function, absolute_import

from setuptools import setup, find_packages

setup(
    name='tfdeform',
    version='0.1.0',
    url='https://github.com/adler-j/tfdeform',

    author='Jonas Adler',
    author_email='jonasadl@kth.se',

    packages=find_packages(exclude=['*test*']),
    package_dir={'tfdeform': 'tfdeform'}
)
