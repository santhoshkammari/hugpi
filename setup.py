from setuptools import setup, find_packages

setup(
    name="hugpi",
    version="0.1.6",  # Incremented the version number
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
    ]
)