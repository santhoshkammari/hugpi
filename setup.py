from setuptools import setup, find_packages

setup(
    name="hugpi",
    version="0.2.0",  # Incremented the version number
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
    ]
)