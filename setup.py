from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Read README.md for long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="tsunami",
    version="0.1.0",
    author="Deniz Akdemir",
    author_email="denizakdemir@example.com",
    description="TSUNAMI: A comprehensive tabular transformer architecture for advanced survival analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/denizakdemir/TSUNAMI",
    packages=find_packages(exclude=["tests*", "*.tests", "*.venv"]),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    test_suite="source.tests",
)
