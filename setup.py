from setuptools import setup

setup(
    name="fdavg",
    version="0.1",
    packages=['fdavg', 'fdavg.metrics', 'fdavg.models', 'fdavg.strategies'],
    package_dir={
        'fdavg': 'FedL-Sync-FDA',
        'fdavg.strategies': 'FedL-Sync-FDA/strategies',
        'fdavg.models': 'FedL-Sync-FDA/models',
        'fdavg.metrics': 'FedL-Sync-FDA/metrics'
    },
    install_requires=[
        'tensorflow', 'numpy'
    ],
)

