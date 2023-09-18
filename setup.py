from setuptools import setup, find_namespace_packages

setup(
    name="fda",
    version="0.1",
    packages=find_namespace_packages(
        include=["FedL-Sync-FDA.strategies", "FedL-Sync-FDA.models", "FedL-Sync-FDA.metrics"]
    ),
    install_requires=[
        'tensorflow', 'numpy'
    ],
)