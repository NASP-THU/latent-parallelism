from setuptools import find_packages, setup

setup(
    name="unite",
    version="0.1.0",
    description="UNITE: Latent Parallelism for video diffusion inference",
    packages=find_packages(include=["wan", "wan.*"]),
    python_requires=">=3.10",
)
