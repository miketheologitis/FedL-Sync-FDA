from setuptools import setup

setup(
    name="fdavg",
    version="0.42",
    packages=['fdavg', 'fdavg.metrics', 'fdavg.models', 'fdavg.strategies'],
    install_requires=[
        'tensorflow', 'numpy'
    ],
)

