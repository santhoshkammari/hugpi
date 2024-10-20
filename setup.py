from setuptools import setup, find_packages

setup(
    name="hugpi",
    version="0.1.7",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # List your core dependencies here
        "numpy",
        "pandas",
    ],
    extras_require={
        "features": [
            # List dependencies for the 'features' module here
            "scikit-learn",
            "matplotlib",
        ],
    },
)