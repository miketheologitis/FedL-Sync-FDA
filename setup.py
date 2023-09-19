from setuptools import setup

setup(
    name="fdavg",
    version="0.2",
    packages=['fdavg', 'fdavg.metrics', 'fdavg.models', 'fdavg.strategies'],
    package_dir={
        'fdavg': 'FdAvg',
        'fdavg.strategies': 'FdAvg/strategies',
        'fdavg.models': 'FdAvg/models',
        'fdavg.metrics': 'FdAvg/metrics'
    },
    install_requires=[
        'tensorflow', 'numpy'
    ],
)

