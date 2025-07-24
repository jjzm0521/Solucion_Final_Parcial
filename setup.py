"""Package installer"""

from setuptools import find_packages, setup  # type: ignore

setup(
    name="homework",
    version="0.1",
    packages=find_packages(),
    install_requires=[
    "numpy==1.26.4"
    "matplotlib==3.8.4"
    "scipy==1.13.1"

    ],
)
