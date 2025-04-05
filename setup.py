from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Read README.md for long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="enhanced_deephit",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive tabular transformer architecture for advanced survival analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/TSUNAMI",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)