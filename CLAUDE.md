# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test Commands

- Run all tests: `pytest enhanced_deephit/tests/`
- Run a single test file: `pytest enhanced_deephit/tests/test_data_processor.py`
- Run a single test function: `pytest enhanced_deephit/tests/test_data_processor.py::test_missing_data_handling`
- Install in development mode: `pip install -e .`
- Run examples: `python enhanced_deephit/examples/single_risk_example.py`
- Lint code: `flake8 enhanced_deephit/`

## Code Style Guidelines

- **Imports**: Standard library first, third-party second, local modules third with clear separation
- **Docstrings**: NumPy style with Parameters/Returns sections, including type descriptions
- **Types**: Use type hints (from typing import Dict, List, Tuple, Optional, etc.)
- **Naming**: 
  - snake_case for variables, functions, methods
  - CamelCase for classes
  - ALL_CAPS for constants
- **Error Handling**: Use ValueError for parameter validation, wrap numerical operations with nan_to_num
- **Line Length**: ~100 characters
- **Class Structure**: Public methods first, private methods (prefixed with _) second
- **Tensor Documentation**: Include tensor shapes in docstrings [batch_size, ...]

## Important Guidelines

- Do NOT use target variables (time, event, time_bin) as predictors to avoid data leakage
- When working with categorical variables, ensure original labels are maintained for visualization
- Always include sample_weights support in model training and evaluation functions
- Ensure categorical variables are properly grouped in importance calculations
- Maintain clear separation between features and targets in DataProcessor operations