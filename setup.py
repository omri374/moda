from setuptools import setup, find_packages

# read the contents of the README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    # print(long_description)

setup(
    name='moda',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='0.3.0',
    packages=find_packages(include=['moda.dataprep',
                                    'moda.dataprep.*',
                                    'moda.evaluators',
                                    'moda.evaluators.*',
                                    'moda.models',
                                    'moda.models.*', ]),
    url='https://www.github.com/omri374/moda',
    license='MIT',
    author='Omri Mendels',
    author_email='omri.mendels@microsoft.com',
    description='Tools for analyzing trending topics',
    install_requires=['numpy', 'pandas', 'stldecompose', 'statsmodels==0.10.2', 'comet_ml', 'requests', 'matplotlib',
                      'pytest',
                      'scikit_learn==0.23.0', 'pytest-cov']

)
