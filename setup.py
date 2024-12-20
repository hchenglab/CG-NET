from setuptools import find_packages, setup

setup(
    name="cgnet",
    version="1.0",
    description="A physics-informed cluster graph neural network (CG-NET)",
    url="https://github.com/hchenglab/CG-NET",
    packages=find_packages(),
    include_package_data=True,
)