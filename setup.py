from setuptools import setup

setup(
    name='moda',
    version='0.0.1',
    packages=['moda.test', 'moda.models', 'moda.models.twitter', 'moda.models.ma_seasonal',
              'moda.models.keras-anomaly-detection.demo', 'moda.models.keras-anomaly-detection.demo.ecg_demo',
              'moda.models.keras-anomaly-detection.demo.credit_card_demo',
              'moda.models.keras-anomaly-detection.keras_anomaly_detection',
              'moda.models.keras-anomaly-detection.keras_anomaly_detection.library', 'moda.dataprep',
              'moda.exploration', 'moda.exploration.mappings'],
    url='https://www.github.com/omri374/moda',
    license='MIT',
    author='Omri Mendels',
    author_email='omri.mendels@microsoft.com',
    description='Tools for analyzing trending topics', install_requires=['numpy', 'pandas', 'pytest']

)
