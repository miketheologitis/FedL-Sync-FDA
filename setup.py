from setuptools import setup

setup(
    name="fda",
    version="0.1",
    packages=['fda', 'fda.metrics', 'fda.models', 'fda.strategies'],
    package_dir={
        'fda': 'FedL-Sync-FDA',
        'fda.strategies': 'FedL-Sync-FDA/strategies',
        'fda.models': 'FedL-Sync-FDA/models',
        'fda.metrics': 'FedL-Sync-FDA/metrics'
    },
    install_requires=[
        'tensorflow', 'numpy'
    ],
)

