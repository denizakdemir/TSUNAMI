# Contributing to TSUNAMI

Thank you for your interest in contributing to TSUNAMI! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## How to Contribute

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes
4. Run the tests to ensure your changes don't break existing functionality
5. Submit a pull request

## Development Setup

1. Clone the repository:

```bash
git clone https://github.com/denizakdemir/TSUNAMI.git
cd TSUNAMI
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

3. Install development dependencies:

```bash
pip install pytest flake8
```

## Testing

Run the tests using pytest:

```bash
pytest enhanced_deephit/tests/
```

You can run specific tests:

```bash
pytest enhanced_deephit/tests/test_model.py
pytest enhanced_deephit/tests/test_data_processor.py::test_missing_data_handling
```

## Code Style

This project follows PEP 8 style guidelines with a few exceptions (see `.flake8` file). Please ensure your code adheres to these standards.

Run the linter to check your code:

```bash
flake8 enhanced_deephit/
```

## Documentation Style

- Use NumPy docstring format for all functions and classes
- Include type information in the docstrings
- For PyTorch tensors, specify the tensor shapes in the format `[batch_size, ...]`

## Pull Request Process

1. Update the README.md or documentation with details of changes if appropriate
2. The pull request will be reviewed by maintainers
3. Once approved, the pull request will be merged

## Release Process

Releases are managed by the project maintainers.