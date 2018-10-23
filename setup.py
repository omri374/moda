from setuptools import setup

setup(
    name='moda',
    version='0.1.4',
    packages=['moda.dataprep',
              'moda.evaluators',
              'moda.models'
              ],
    url='https://www.github.com/omri374/moda',
    license='MIT',
    author='Omri Mendels',
    author_email='omri.mendels@microsoft.com',
    description='Tools for analyzing trending topics',
    install_requires=['numpy', 'pandas', 'stldecompose', 'statsmodels', 'comet_ml', 'requests', 'matplotlib', 'pytest',
                      'pytest-cov']

)
