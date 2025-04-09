# TSUNAMI

**Tabular SUrviVal aNAlytics with Multiple targets and Imputation**

TSUNAMI is a comprehensive tabular transformer architecture for advanced survival analysis, extending the DeepHit model with modern deep learning techniques. This framework can simultaneously handle survival analysis, competing risks, classification, regression, and other prediction tasks within a unified multi-task learning framework.

## Features

- **Tabular Transformer Architecture**: Efficiently processes tabular data with attention mechanisms
- **Missing Data Handling**: Robust methods for handling incomplete data
- **Categorical Feature Processing**: Native categorical data embedding
- **Multi-Task Framework**: Supports diverse target types:
  - Survival analysis (single risk)
  - Competing risks (multiple event types)
  - Classification (binary and multi-class)
  - Regression (continuous outcomes)
  - Count data (Poisson and binomial)
- **Masked Loss Functionality**: Handles incomplete target data
- **Variational Methods**: Uncertainty quantification
- **Visualization Capabilities**: Comprehensive visualization for all supported target types
- **Simulation Utilities**: Generate synthetic data for experimentation and validation

## Installation

Clone the repository:

```bash
git clone https://github.com/denizakdemir/TSUNAMI.git
cd TSUNAMI
```

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
python -m pip install -e .
```

## Usage

Here's a basic example of using TSUNAMI for survival analysis:

```python
import torch
from source.data.processing import DataProcessor
from source.models import EnhancedDeepHit
from source.models.tasks.survival import SingleRiskHead, CompetingRisksHead
from source.simulation.data_generation import generate_survival_data

# Generate synthetic data for demonstration
data, target, num_bins = generate_survival_data(
    n_samples=1000, 
    n_features=20, 
    include_categorical=True
)

# Initialize data processor
processor = DataProcessor(
    num_impute_strategy='mean',
    cat_impute_strategy='most_frequent',
    normalize='robust',
    create_missing_indicators=True
)

# Fit processor on training data
processor.fit(data)

# Transform data
processed_data = processor.transform(data)

# Initialize survival task head
survival_head = SingleRiskHead(
    name='survival',
    input_dim=128,
    num_time_bins=num_bins,
    alpha_rank=0.1
)

# Initialize model
model = EnhancedDeepHit(
    num_continuous=len(continuous_cols),
    targets=[survival_head],
    cat_feat_info=cat_feat_info,  # Categorical feature information
    encoder_dim=128,
    encoder_depth=4,
    encoder_heads=8
)

# Train model
history = model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=1e-3,
    num_epochs=50,
    patience=5
)

# Predict
predictions = model.predict(test_continuous, categorical=test_categorical)
```

See the examples directory for more detailed usage examples:
- `source/examples/single_risk_example.py`: Single-risk survival analysis
- `source/examples/competing_risks_example.py`: Competing risks survival analysis
- `source/examples/multi_task_example.py`: Multi-task learning with survival, classification, and regression
- `source/examples/importance/feature_importance_example.py`: Feature importance calculation
- `source/examples/visualization/survival_visualization_example.py`: Visualization tools

## Project Structure

```
source/
├── checkpoints/        # Model checkpoint storage
├── data/               # Data processing utilities
├── examples/           # Example usage scripts
│   ├── importance/     # Feature importance examples
│   └── visualization/  # Visualization examples
├── models/             # Model architecture components
│   └── tasks/          # Task-specific heads
├── simulation/         # Synthetic data generation
├── tests/              # Unit tests
│   └── utils/          # Test utilities
└── visualization/      # Visualization utilities
    └── importance/     # Feature importance tools
```

For more details about the project structure, see [STRUCTURE.md](STRUCTURE.md).

## Architecture

TSUNAMI consists of several core components:

1. **Data Preprocessing**: Handles missing data, categorical encoding, and feature normalization
2. **Tabular Transformer Encoder**: Processes tabular data with self-attention mechanisms
3. **Multi-Task Learning Framework**: Coordinates different prediction tasks
4. **Task-Specific Heads**: Specialized components for each prediction task
5. **Masked Loss Mechanism**: Handles incomplete target data
6. **Variational Component**: Provides uncertainty quantification
7. **Visualization Module**: Tools for interpreting model results
8. **Simulation Utilities**: Generate synthetic data for experimentation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use TSUNAMI in your research, please cite:

```
@misc{tsunami2023,
  author = {Deniz Akdemir},
  title = {TSUNAMI: Tabular SUrviVal aNAlytics with Multiple targets and Imputation},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/denizakdemir/TSUNAMI}}
}
```

## Acknowledgments

- The original DeepHit paper by Lee et al. (2018)
- All contributors to this project
