from setuptools import setup

setup(
    name='moda',
    version='0.0.5',
    packages=['moda.dataprep',
              'moda.evaluators', 'moda.evaluators.metrics',
              'moda.models', 'moda.models.twitter', 'moda.models.ma_seasonal', 'moda.models.stl'
              ],
    url='https://www.github.com/omri374/moda',
    license='MIT',
    author='Omri Mendels',
    author_email='omri.mendels@microsoft.com',
    description='Tools for analyzing trending topics',
    install_requires=['numpy', 'pandas', 'stldecompose', 'statsmodels', 'requests', 'matplotlib','pytest','pytest-cov']

)
