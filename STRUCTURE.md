# TSUNAMI Project Structure

This document outlines the organization of the TSUNAMI project codebase.

## Directory Structure

```
source/
├── __init__.py             # Package initialization
├── data/                   # Data processing modules
│   ├── __init__.py
│   └── processing.py       # Data processing utilities
├── models/                 # Model architecture components
│   ├── __init__.py
│   ├── encoder.py          # Transformer-based encoder
│   ├── model.py            # Main model implementation
│   └── tasks/              # Task-specific modules
│       ├── __init__.py
│       ├── base.py         # Base task implementations
│       ├── standard.py     # Standard prediction heads
│       └── survival.py     # Survival analysis heads
├── simulation/             # Simulation utilities
│   ├── __init__.py
│   ├── data_generation.py  # Synthetic data generation
│   ├── missing_data_analysis.py  # Missing data utilities
│   └── scenario_analysis.py      # Simulation scenarios
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── test_*.py           # Test modules
│   └── utils/              # Test utilities
│       ├── __init__.py
│       ├── test_debug.py   # Debug utilities
│       └── test_imports.py # Import testing
└── visualization/          # Visualization modules
    ├── __init__.py
    ├── feature_effects.py  # Feature effect plots
    ├── importance/         # Feature importance tools
    │   ├── __init__.py
    │   └── importance.py   # Importance calculation
    ├── survival_plots.py   # Survival curve plotting
    └── uncertainty_plots.py # Uncertainty visualization
```

## Key Components

### Data Processing

- `DataProcessor`: Handles feature preprocessing, normalization, and categorical feature encoding

### Models

- `EnhancedDeepHit`: Main model class that implements the TSUNAMI architecture
- `TabularTransformer`: Transformer encoder for tabular data with feature interactions
- Task Heads:
  - `SingleRiskHead`: Survival analysis for single risk scenarios
  - `CompetingRisksHead`: Survival analysis with competing risks
  - `ClassificationHead`: Binary and multi-class classification
  - `RegressionHead`: Regression tasks
  - `CountDataHead`: Count data modeling

### Visualization

- `PermutationImportance`: Feature importance using permutation approach
- `ShapImportance`: SHAP-based feature importance
- `IntegratedGradients`: Gradient-based importance
- `plot_survival_curves`: Generates survival probability curves

### Simulation

- `generate_synthetic_data`: Creates synthetic survival data
- `create_missing_pattern`: Simulates missing data scenarios
- `run_scenario_analysis`: Tests model performance under various scenarios
