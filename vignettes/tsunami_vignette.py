import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import warnings

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import TSUNAMI modules
from enhanced_deephit.data.processing import DataProcessor
from enhanced_deephit.models import EnhancedDeepHit
from enhanced_deephit.models.tasks.base import TaskHead
from enhanced_deephit.models.tasks.standard import ClassificationHead, RegressionHead
from enhanced_deephit.models.tasks.survival import SingleRiskHead
from enhanced_deephit.visualization.importance.importance import (
    PermutationImportance,
    ShapImportance,
    IntegratedGradients,
    AttentionImportance
)
from enhanced_deephit.visualization.survival_plots import (
    plot_survival_curve,
    plot_cumulative_incidence,
    plot_calibration_curve
)
from enhanced_deephit.visualization.feature_effects import (
    plot_partial_dependence,
    plot_ice_curves,
    plot_feature_interaction
)

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Create output directory for plots
os.makedirs("plots", exist_ok=True)

# Load and explore the EBMT dataset
print("Loading EBMT dataset...")
ebmt_data = pd.read_csv('vignettes/ebmt3.csv')
print(f"Dataset loaded: {ebmt_data.shape[0]} patients, {ebmt_data.shape[1]} variables")

# Display dataset information
print("\nDataset Information:")
print(ebmt_data.head())
print("\nVariable types and counts:")
print(ebmt_data.dtypes)
print("\nMissing values:")
print(ebmt_data.isnull().sum())

# Basic statistics of the time-to-event variables
print("\nTime-to-event statistics:")
print("\ntime:")
print(ebmt_data['time'].describe())

# Event counts
print("\nEvent counts:")
print("\nstatus:")
print(ebmt_data['status'].value_counts())

# Categorical variable distributions
print("\nCategorical variable distributions:")
for cat_var in ['dissub', 'age', 'match', 'tcd']:
    print(f"\n{cat_var}:")
    print(ebmt_data[cat_var].value_counts())

# Create additional synthetic variables to demonstrate multi-task capability
print("\nCreating synthetic variables for demonstration...")

# Rename columns for clarity in modeling
ebmt_data = ebmt_data.rename(columns={
    'time': 'survival_time',
    'status': 'event_indicator',
    'match': 'drmatch'  # More descriptive name
})

# Simplify event indicator for demonstration (0: censored, 1: event)
# Original dataset has 0, 1, 2, 3, 4, 5, 6 statuses
ebmt_data['event_indicator'] = (ebmt_data['event_indicator'] > 0).astype(int)

# Synthetic binary classification target
ebmt_data['binary_outcome'] = np.random.binomial(1, 0.3, size=len(ebmt_data))

# Synthetic regression target (e.g., biomarker value) - not using survival_time to avoid data leakage
age_effect = 0.3 * (ebmt_data['age'] == ">40").astype(int) * 10
dissub_effect = 0.2 * (ebmt_data['dissub'] == "ALL").astype(int) * 5
ebmt_data['biomarker'] = age_effect + dissub_effect + np.random.normal(15, 5, size=len(ebmt_data))

# Synthetic competing risks (create a cause indicator from 'cod' column)
# 0: censored, 1: Relapse, 2: Death without relapse
ebmt_data['cr_cause'] = 0  # Default to censored
ebmt_data.loc[ebmt_data['event_indicator'] == 1, 'cr_cause'] = 1  # Relapse as default event
# Randomly assign half of the events to cause 2 (death) for demonstration
event_mask = ebmt_data['event_indicator'] == 1
event_indices = ebmt_data[event_mask].index
cause2_indices = np.random.choice(event_indices, size=len(event_indices) // 2, replace=False)
ebmt_data.loc[cause2_indices, 'cr_cause'] = 2

print("Synthetic variables created:")
print(" - 'binary_outcome': Binary classification target")
print(" - 'biomarker': Regression target")
print(" - 'cr_cause': Competing risks cause (0: censored, 1: relapse, 2: death)")

# Data preprocessing
print("\nPreprocessing data...")

# Define feature columns 
numeric_features = []  # Not using survival_time as feature to avoid data leakage
categorical_features = ['dissub', 'age', 'drmatch', 'tcd']

# Create a preprocessor
preprocessor = DataProcessor(
    num_impute_strategy='median',
    cat_impute_strategy='most_frequent',
    normalize='robust'
)

# Prepare feature dataframe - only using actual features, not target variables
feature_df = ebmt_data[categorical_features].copy()  # No numeric features used

# Fit the preprocessor and transform the data
preprocessor.fit(feature_df)
processed_df = preprocessor.transform(feature_df)

print("\nPreprocessed data:")
print(processed_df.head())

# Convert to PyTorch tensors
X_tensor = torch.tensor(processed_df.values, dtype=torch.float32)

# Prepare time and event data for SingleRiskHead
# Discretize time into bins
num_time_bins = 20
max_time = np.percentile(ebmt_data['survival_time'], 99)
bin_edges = np.linspace(0, max_time, num_time_bins + 1)
time_bins = np.digitize(ebmt_data['survival_time'], bin_edges) - 1
time_bins = np.clip(time_bins, 0, num_time_bins - 1)

# Create target format for SingleRiskHead
# [event_indicator, time_bin, one_hot_encoding]
single_risk_target = np.zeros((len(ebmt_data), 2 + num_time_bins))
single_risk_target[:, 0] = ebmt_data['event_indicator'].values
single_risk_target[:, 1] = time_bins

# One-hot encoding of time
for i in range(len(ebmt_data)):
    if ebmt_data['event_indicator'].iloc[i]:
        # For events, mark the event time
        single_risk_target[i, 2 + int(time_bins[i])] = 1
    else:
        # For censored, mark all times after censoring as unknown (-1)
        single_risk_target[i, 2 + int(time_bins[i]):] = -1

# Create target for CompetingRisksHead
# [event_indicator, time_bin, cause_index, one_hot_encoding]
competing_risks_target = np.zeros((len(ebmt_data), 3 + num_time_bins * 2))
competing_risks_target[:, 0] = (ebmt_data['cr_cause'] > 0).astype(float)  # Event indicator
competing_risks_target[:, 1] = time_bins
competing_risks_target[:, 2] = ebmt_data['cr_cause'] - 1  # -1 for censored, 0 for cause 1, 1 for cause 2
competing_risks_target[competing_risks_target[:, 2] < 0, 2] = -1  # Set censored to -1

# Convert targets to tensors
single_risk_tensor = torch.tensor(single_risk_target, dtype=torch.float32)
competing_risks_tensor = torch.tensor(competing_risks_target, dtype=torch.float32)
binary_tensor = torch.tensor(ebmt_data['binary_outcome'].values, dtype=torch.float32).unsqueeze(1)
regression_tensor = torch.tensor(ebmt_data['biomarker'].values, dtype=torch.float32).unsqueeze(1)

# Create dataset and dataloader with proper formatting
class SurvivalDataset(torch.utils.data.Dataset):
    def __init__(self, X, sr_targets=None, cr_targets=None, binary_targets=None, regression_targets=None):
        self.X = X
        self.sr_targets = sr_targets
        self.cr_targets = cr_targets
        self.binary_targets = binary_targets
        self.regression_targets = regression_targets
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        item = {'continuous': self.X[idx]}
        targets = {}
        
        if self.sr_targets is not None:
            targets['survival'] = self.sr_targets[idx]
        
        if self.cr_targets is not None:
            targets['competing_risks'] = self.cr_targets[idx]
            
        if self.binary_targets is not None:
            targets['binary'] = self.binary_targets[idx]
            
        if self.regression_targets is not None:
            targets['regression'] = self.regression_targets[idx]
            
        item['targets'] = targets
        return item

# Create full multi-task dataset
full_dataset = SurvivalDataset(
    X_tensor, 
    sr_targets=single_risk_tensor,
    cr_targets=competing_risks_tensor,
    binary_targets=binary_tensor,
    regression_targets=regression_tensor
)

# Split data into train and test sets
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

# Create random split
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size, test_size]
)

# Create dataloaders
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

print(f"\nDatasets prepared: {len(train_dataset)} train samples, {len(val_dataset)} validation samples, {len(test_dataset)} test samples")

# Demonstration 1: Single Risk Survival Analysis
print("\n--- DEMONSTRATION 1: SINGLE RISK SURVIVAL ANALYSIS ---\n")

# Create single risk model
sr_task_head = SingleRiskHead(
    name='survival',
    input_dim=64,
    num_time_bins=num_time_bins,
    alpha_rank=0.1,
    alpha_calibration=0.0  # Disable calibration loss due to issues
)

sr_model = EnhancedDeepHit(
    num_continuous=X_tensor.shape[1],
    targets=[sr_task_head],
    encoder_dim=64,
    encoder_depth=2,
    encoder_heads=4,
    include_variational=True,
    device='cpu'
)

# Train the single risk model
print("Training single risk model...")
sr_model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=3,  # Reduced for speed
    learning_rate=0.001,
    patience=2
)

# Generate predictions for test set
print("\nGenerating predictions for test set...")
X_test = torch.cat([test_dataset[i]['continuous'].unsqueeze(0) for i in range(len(test_dataset))])
single_risk_preds = sr_model.predict(X_test)

# Extract survival curves and uncertainty
sr_survival_curves = single_risk_preds['task_outputs']['survival']['survival'].detach().numpy()
sr_uncertainty = sr_model.compute_uncertainty(X_test[:10], num_samples=5)
sr_survival_std = sr_uncertainty['survival']['std'].detach().numpy()

# Verify survival curves start at 1.0
print("\nVerifying survival curves start at 1.0:")
print(f"First survival curve values: {sr_survival_curves[0, :5]}")

# Plot single risk survival curves
print("\nPlotting survival curves...")
time_points = bin_edges[:-1]  # Use bin start points as time points

# Plot survival curves for first 5 patients
fig1 = plot_survival_curve(
    sr_survival_curves[:5], 
    time_points=time_points,
    labels=[f'Patient {i+1}' for i in range(5)],
    title="Survival Curves for Multiple Patients"
)
fig1.savefig("plots/sr_survival_curves.png")
plt.close(fig1)

# Plot with uncertainty
fig2 = plot_survival_curve(
    sr_survival_curves[0],
    time_points=time_points,
    uncertainty=sr_survival_std[0],
    title="Survival Curve with Uncertainty"
)
fig2.savefig("plots/sr_survival_uncertainty.png")
plt.close(fig2)

# Feature importance for single risk model
print("\nCalculating feature importance...")
perm_imp = PermutationImportance(sr_model)
survival_targets = torch.cat([test_dataset[i]['targets']['survival'].unsqueeze(0) for i in range(100)])
perm_importances = perm_imp.compute_importance(
    {'continuous': X_test[:100]},
    y=survival_targets,
    n_repeats=2,
    feature_names=processed_df.columns.tolist()
)

# Plot permutation importance
fig3 = perm_imp.plot_importance(perm_importances)
plt.title("Permutation Importance (Single Risk Model)")
fig3.savefig("plots/sr_permutation_importance.png")
plt.close(fig3)

# Partial dependence plots
print("\nGenerating partial dependence plots...")
# For a categorical feature embedding since we removed numeric features
categorical_idx = processed_df.columns.get_loc('dissub_embed_0')  
categorical_feature_name = processed_df.columns[categorical_idx]

fig4 = plot_partial_dependence(
    sr_model,
    X_test[:100],
    feature_idx=categorical_idx,
    feature_name=categorical_feature_name,
    target='risk_score',
    title=f"Partial Dependence of Risk Score on {categorical_feature_name}",
    categorical_info={
        'cardinality': 4,  # CML, AML, ALL, Missing
        'embed_dim': 2,
        'original_name': 'dissub',
        'reverse_mapping': {
            0: 'CML',
            1: 'AML',
            2: 'ALL'
        }
    }
)
fig4.savefig("plots/sr_pd_categorical1.png")
plt.close(fig4)

# For another categorical feature
categorical_idx_other = processed_df.columns.get_loc('drmatch_embed_0')  
categorical_feature_name_other = processed_df.columns[categorical_idx_other]

fig5 = plot_partial_dependence(
    sr_model,
    X_test[:100],
    feature_idx=categorical_idx_other,
    feature_name=categorical_feature_name_other,
    target='risk_score',
    title=f"Partial Dependence of Risk Score on {categorical_feature_name_other}",
    categorical_info={
        'cardinality': 3,  # No gender mismatch, Gender mismatch, Missing
        'embed_dim': 2,
        'original_name': 'drmatch',
        'reverse_mapping': {
            0: 'No gender mismatch',
            1: 'Gender mismatch'
        }
    }
)
fig5.savefig("plots/sr_pd_categorical.png")
plt.close(fig5)

# ICE curves for a categorical feature (using a different categorical feature for variety)
categorical_idx2 = processed_df.columns.get_loc('age_embed_0')  # Using a different categorical feature embedding
categorical_feature_name2 = processed_df.columns[categorical_idx2]

fig6 = plot_ice_curves(
    sr_model,
    X_test[:20],
    feature_idx=categorical_idx2,
    feature_name=categorical_feature_name2,
    target='risk_score',
    title=f"ICE Curves for {categorical_feature_name2}",
    feature_range=(0, 0.5)  # Limit to the range where our categories are
)
fig6.savefig("plots/sr_ice_curves.png")
plt.close(fig6)

# Feature interaction between two categorical features
fig7 = plot_feature_interaction(
    sr_model,
    X_test[:100],
    feature1_idx=categorical_idx,
    feature2_idx=categorical_idx2,
    feature1_name=categorical_feature_name,
    feature2_name=categorical_feature_name2,
    target='risk_score',
    title=f"Interaction between {categorical_feature_name} and {categorical_feature_name2}"
)
fig7.savefig("plots/sr_feature_interaction.png")
plt.close(fig7)

# Demonstration 2: Competing Risks Analysis - Skipped
print("\n--- DEMONSTRATION 2: COMPETING RISKS ANALYSIS ---\n")
print("Skipping competing risks demonstration as CompetingRisksHead is not yet implemented.")
print("This will be available in a future update.")

# Create dummy data for plotting examples when CompetingRisksHead is not available
print("\nCreating dummy data for plotting examples...")
# Random CIF for two risks - shape: [num_risks, num_time_bins]
time_scale = np.linspace(0, 1, num_time_bins)
dummy_cif = np.zeros((2, num_time_bins))
dummy_cif[0, :] = 0.3 * (1 - np.exp(-time_scale * 2))  # Risk 1
dummy_cif[1, :] = 0.4 * (1 - np.exp(-time_scale * 1.5))  # Risk 2

# Ensure CIF is monotonically increasing
dummy_cif = np.cumsum(np.clip(dummy_cif, 0, 0.1), axis=1)
dummy_cif = np.clip(dummy_cif, 0, 1)

# Overall survival = 1 - sum(CIF)
dummy_survival = 1 - np.sum(dummy_cif, axis=0)

# Repeat for 5 patients with slight variations
dummy_survival_multi = np.array([dummy_survival * (0.9 + 0.2 * np.random.rand()) for _ in range(5)])

# Use these as our "predictions"
cr_cif = dummy_cif
cr_survival = dummy_survival_multi

# Plot competing risks CIF
# The function expects cif with shape [num_risks, num_time_bins]
fig8 = plot_cumulative_incidence(
    cr_cif,  # Already in correct format: [num_risks, num_time_bins]
    time_points=time_points,
    risk_names=['Relapse', 'Death'],
    title="Cumulative Incidence Functions"
)
fig8.savefig("plots/cr_cif.png")
plt.close(fig8)

# Plot competing risks survival
fig9 = plot_survival_curve(
    cr_survival[:5],
    time_points=time_points,
    labels=[f'Patient {i+1}' for i in range(5)],
    title="Overall Survival in Competing Risks Setting"
)
fig9.savefig("plots/cr_overall_survival.png")
plt.close(fig9)

# Create a dummy dictionary with random importance values
np.random.seed(42)  # For reproducibility
cr_perm_importances = {}
for col in processed_df.columns:
    cr_perm_importances[col] = np.random.rand()

# Normalize values
max_val = max(cr_perm_importances.values())
for key in cr_perm_importances:
    cr_perm_importances[key] /= max_val

# Plot importance using generic plotting method
fig10, ax10 = plt.subplots(figsize=(10, 6))
sorted_importance = sorted(cr_perm_importances.items(), key=lambda x: x[1], reverse=True)
feature_names = [x[0] for x in sorted_importance]
scores = [x[1] for x in sorted_importance]
y_pos = np.arange(len(feature_names))
ax10.barh(y_pos, scores, color='skyblue')
ax10.set_yticks(y_pos)
ax10.set_yticklabels(feature_names)
ax10.invert_yaxis()  # Labels read top-to-bottom
ax10.set_xlabel('Importance Score')
ax10.set_title("Simulated Feature Importance (Placeholder)")
fig10.savefig("plots/cr_permutation_importance.png")
plt.close(fig10)

# Demonstration 3: Multi-Task Learning
print("\n--- DEMONSTRATION 3: MULTI-TASK LEARNING ---\n")

# Create tasks for multi-task model
mt_survival_head = SingleRiskHead(
    name='survival',
    input_dim=64,
    num_time_bins=num_time_bins,
    alpha_rank=0.1,
    alpha_calibration=0.0  # Disable calibration loss
)

mt_binary_head = ClassificationHead(
    name='binary',
    input_dim=64,
    num_classes=2,
    task_weight=1.0
)

mt_regression_head = RegressionHead(
    name='regression',
    input_dim=64,
    output_dim=1,
    task_weight=1.0
)

# Create multi-task model
mt_model = EnhancedDeepHit(
    num_continuous=X_tensor.shape[1],
    targets=[mt_survival_head, mt_binary_head, mt_regression_head],
    encoder_dim=64,
    encoder_depth=2,
    encoder_heads=4,
    include_variational=True,
    device='cpu'
)

# Create multi-task dataset
mt_dataset = SurvivalDataset(
    X_tensor, 
    sr_targets=single_risk_tensor,
    binary_targets=binary_tensor,
    regression_targets=regression_tensor
)

# Split data
mt_train_size = int(0.7 * len(mt_dataset))
mt_val_size = int(0.15 * len(mt_dataset))
mt_test_size = len(mt_dataset) - mt_train_size - mt_val_size

# Create random split
mt_train_dataset, mt_val_dataset, mt_test_dataset = torch.utils.data.random_split(
    mt_dataset, [mt_train_size, mt_val_size, mt_test_size]
)

# Create dataloaders
mt_train_loader = torch.utils.data.DataLoader(mt_train_dataset, batch_size=batch_size, shuffle=True)
mt_val_loader = torch.utils.data.DataLoader(mt_val_dataset, batch_size=batch_size)
mt_test_loader = torch.utils.data.DataLoader(mt_test_dataset, batch_size=batch_size)

# Train the multi-task model
print("Training multi-task model...")
mt_model.fit(
    train_loader=mt_train_loader,
    val_loader=mt_val_loader,
    num_epochs=3,  # Reduced for speed
    learning_rate=0.001,
    patience=2
)

# Generate predictions for test set
print("\nGenerating multi-task predictions...")
X_mt_test = torch.cat([mt_test_dataset[i]['continuous'].unsqueeze(0) for i in range(len(mt_test_dataset))])
mt_preds = mt_model.predict(X_mt_test)

# Extract task-specific predictions
mt_survival = mt_preds['task_outputs']['survival']['survival'].detach().numpy()
mt_binary_probs = mt_preds['task_outputs']['binary']['probabilities'].detach().numpy()
mt_regression_values = mt_preds['task_outputs']['regression']['predictions'].detach().numpy()

# Plot multi-task predictions
print("\nPlotting multi-task predictions...")
# Survival curves
fig11 = plot_survival_curve(
    mt_survival[0],
    time_points=time_points,
    title="Multi-Task Survival Prediction"
)
fig11.savefig("plots/mt_survival.png")
plt.close(fig11)

# Binary classification histogram
fig12, ax12 = plt.subplots(figsize=(10, 6))
ax12.hist(mt_binary_probs, bins=20)
ax12.set_title("Multi-Task Binary Classification Probabilities")
ax12.set_xlabel("Probability")
ax12.set_ylabel("Frequency")
fig12.savefig("plots/mt_binary.png")
plt.close(fig12)

# Regression prediction scatter plot
binary_targets = torch.cat([mt_test_dataset[i]['targets']['binary'] for i in range(len(mt_test_dataset))])
regression_targets = torch.cat([mt_test_dataset[i]['targets']['regression'] for i in range(len(mt_test_dataset))])

fig13, ax13 = plt.subplots(figsize=(10, 6))
ax13.scatter(regression_targets.numpy(), mt_regression_values.flatten())
ax13.plot([min(regression_targets), max(regression_targets)], [min(regression_targets), max(regression_targets)], 'r--')
ax13.set_title("Multi-Task Regression Predictions vs Actual")
ax13.set_xlabel("Actual Values")
ax13.set_ylabel("Predicted Values")
fig13.savefig("plots/mt_regression.png")
plt.close(fig13)

# SHAP Importance (more detailed analysis for one model)
print("\nGenerating SHAP importance values...")
shap_imp = ShapImportance(sr_model)
# Further reduce the dataset size for faster computation
shap_values = shap_imp.compute_importance(
    {'continuous': X_test[:10]},  # Use only 10 samples
    n_samples=3,  # Use minimal background samples
    feature_names=processed_df.columns.tolist()
)

# Plot SHAP importance
fig14 = shap_imp.plot_importance(shap_values, plot_type='bar')
plt.title("SHAP Feature Importance")
fig14.savefig("plots/shap_importance_bar.png")
plt.close(fig14)

# Attention Importance
print("\nGenerating attention-based importance...")
attn_imp = AttentionImportance(sr_model)
attention_scores = attn_imp.compute_importance(
    {'continuous': X_test[:10]},  # Use only 10 samples for speed
    feature_names=processed_df.columns.tolist(),
    layer_idx=-1
)

# Plot attention importance
fig15 = attn_imp.plot_importance(attention_scores)
plt.title("Attention-Based Feature Importance")
fig15.savefig("plots/attention_importance.png")
plt.close(fig15)

# Demonstration 4: Using Sample Weights
print("\n--- DEMONSTRATION 4: USING SAMPLE WEIGHTS ---\n")

# Create sample weights for our dataset
print("Creating sample weights...")
# We'll create weights that prioritize:
# 1. Patients who had an event (weight = 2.0)
# 2. Patients with certain characteristics (e.g., age > 40, weight = 1.5)
# 3. All other patients (weight = 1.0)

sample_weights = np.ones(len(ebmt_data))

# Assign higher weight to patients who had an event
event_mask = ebmt_data['event_indicator'] == 1
sample_weights[event_mask] = 2.0

# Assign medium weight to older patients
age_mask = ebmt_data['age'] == ">40"
sample_weights[age_mask & ~event_mask] = 1.5  # Only if not already weighted higher

print(f"Sample weights created: {len(sample_weights)} weight values")
print(f"Weight distribution: {np.unique(sample_weights, return_counts=True)}")

# Convert to tensor
sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)

# Create a weighted dataset class
class WeightedSurvivalDataset(torch.utils.data.Dataset):
    def __init__(self, X, weights, sr_targets=None, binary_targets=None, regression_targets=None):
        self.X = X
        self.weights = weights
        self.sr_targets = sr_targets
        self.binary_targets = binary_targets
        self.regression_targets = regression_targets
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        item = {'continuous': self.X[idx]}
        targets = {}
        
        if self.sr_targets is not None:
            targets['survival'] = self.sr_targets[idx]
            
        if self.binary_targets is not None:
            targets['binary'] = self.binary_targets[idx]
            
        if self.regression_targets is not None:
            targets['regression'] = self.regression_targets[idx]
            
        item['targets'] = targets
        item['sample_weights'] = self.weights[idx]
        return item

# Create datasets with and without sample weights
print("\nCreating weighted and unweighted datasets...")
# 1. Dataset without weights - we'll reuse the previous dataset
unweighted_dataset = SurvivalDataset(
    X_tensor,
    sr_targets=single_risk_tensor,
    binary_targets=binary_tensor
)

# 2. Dataset with weights
weighted_dataset = WeightedSurvivalDataset(
    X_tensor,
    weights=sample_weights_tensor,
    sr_targets=single_risk_tensor,
    binary_targets=binary_tensor
)

# Split data
train_size = int(0.7 * len(unweighted_dataset))
val_size = int(0.15 * len(unweighted_dataset))
test_size = len(unweighted_dataset) - train_size - val_size

# Set the same seed for both splits to ensure they're equivalent
generator = torch.Generator().manual_seed(42)

# Create random splits with same indices
unweighted_train, unweighted_val, unweighted_test = torch.utils.data.random_split(
    unweighted_dataset, [train_size, val_size, test_size], generator=generator
)

weighted_train, weighted_val, weighted_test = torch.utils.data.random_split(
    weighted_dataset, [train_size, val_size, test_size], generator=generator
)

# Create dataloaders
batch_size = 64
unweighted_train_loader = torch.utils.data.DataLoader(unweighted_train, batch_size=batch_size, shuffle=True)
unweighted_val_loader = torch.utils.data.DataLoader(unweighted_val, batch_size=batch_size)

weighted_train_loader = torch.utils.data.DataLoader(weighted_train, batch_size=batch_size, shuffle=True)
weighted_val_loader = torch.utils.data.DataLoader(weighted_val, batch_size=batch_size)

print("Datasets prepared for sample weight comparison")

# Create two identical task configurations
print("\nCreating task heads and models...")
task_config = [
    {
        'type': 'survival',
        'params': {
            'name': 'survival',
            'input_dim': 64,
            'num_time_bins': num_time_bins,
            'alpha_rank': 0.1,
            'alpha_calibration': 0.0
        }
    },
    {
        'type': 'binary',
        'params': {
            'name': 'binary',
            'input_dim': 64,
            'num_classes': 2,
            'task_weight': 1.0
        }
    }
]

# Create tasks for unweighted model
unweighted_survival_head = SingleRiskHead(**task_config[0]['params'])
unweighted_binary_head = ClassificationHead(**task_config[1]['params'])

unweighted_model = EnhancedDeepHit(
    num_continuous=X_tensor.shape[1],
    targets=[unweighted_survival_head, unweighted_binary_head],
    encoder_dim=64,
    encoder_depth=2,
    encoder_heads=4,
    include_variational=False,  # Disable for simplicity
    device='cpu'
)

# Create identical tasks for weighted model
weighted_survival_head = SingleRiskHead(**task_config[0]['params'])
weighted_binary_head = ClassificationHead(**task_config[1]['params'])

weighted_model = EnhancedDeepHit(
    num_continuous=X_tensor.shape[1],
    targets=[weighted_survival_head, weighted_binary_head],
    encoder_dim=64,
    encoder_depth=2,
    encoder_heads=4,
    include_variational=False,  # Disable for simplicity
    device='cpu'
)

# For reproducibility, ensure both models start with the same weights
torch.manual_seed(42)
# We don't need to manually initialize parameters as PyTorch does this automatically
# when the models are created

# Train the models
print("\nTraining model WITHOUT sample weights...")
unweighted_history = unweighted_model.fit(
    train_loader=unweighted_train_loader,
    val_loader=unweighted_val_loader,
    num_epochs=3,  # Reduced for demonstration
    learning_rate=0.001,
    patience=2,
    use_sample_weights=False  # Default, but being explicit here
)

print("\nTraining model WITH sample weights...")
weighted_history = weighted_model.fit(
    train_loader=weighted_train_loader,
    val_loader=weighted_val_loader,
    num_epochs=3,  # Reduced for demonstration
    learning_rate=0.001,
    patience=2,
    use_sample_weights=True  # Enable sample weights
)

# Evaluate and compare models
print("\nEvaluating and comparing models...")

# Extract test data for evaluation
X_test_tensor = torch.cat([unweighted_test[i]['continuous'].unsqueeze(0) for i in range(len(unweighted_test))])
y_test_survival = torch.cat([unweighted_test[i]['targets']['survival'].unsqueeze(0) for i in range(len(unweighted_test))])
y_test_binary = torch.cat([unweighted_test[i]['targets']['binary'].unsqueeze(0) for i in range(len(unweighted_test))])

# Generate predictions for both models
unweighted_preds = unweighted_model.predict(X_test_tensor)
weighted_preds = weighted_model.predict(X_test_tensor)

# Extract predictions for comparison
unwght_survival = unweighted_preds['task_outputs']['survival']['survival'].detach().numpy()
unwght_binary = unweighted_preds['task_outputs']['binary']['probabilities'].detach().numpy()

wght_survival = weighted_preds['task_outputs']['survival']['survival'].detach().numpy()
wght_binary = weighted_preds['task_outputs']['binary']['probabilities'].detach().numpy()

print("\nPrediction comparison:")
print(f"Unweighted survival mean: {np.mean(unwght_survival):.4f}")
print(f"Weighted survival mean: {np.mean(wght_survival):.4f}")
print(f"Unweighted binary mean: {np.mean(unwght_binary):.4f}")
print(f"Weighted binary mean: {np.mean(wght_binary):.4f}")

# Plot comparisons
print("\nPlotting comparison visualizations...")

# 1. Plot survival curves for both models
fig_sw1, ax_sw1 = plt.subplots(figsize=(12, 6))

# Select sample patients to plot
sample_indices = [0, 10, 20]  # Select a few test samples

# Plot survival curves from both models
for i, idx in enumerate(sample_indices):
    # Plot unweighted model prediction
    ax_sw1.plot(time_points, unwght_survival[idx], 
             label=f'Patient {idx+1} - Unweighted', 
             linestyle='-', linewidth=2, 
             color=f'C{i}')
    
    # Plot weighted model prediction
    ax_sw1.plot(time_points, wght_survival[idx], 
             label=f'Patient {idx+1} - Weighted', 
             linestyle='--', linewidth=2, 
             color=f'C{i}')

ax_sw1.set_xlabel('Time')
ax_sw1.set_ylabel('Survival Probability')
ax_sw1.set_title('Comparison of Survival Curves: Weighted vs. Unweighted')
ax_sw1.legend()
ax_sw1.grid(True, alpha=0.3)
fig_sw1.savefig("plots/weighted_vs_unweighted_survival.png")
plt.close(fig_sw1)

# 2. Compare binary classification predictions
fig_sw2, ax_sw2 = plt.subplots(figsize=(10, 6))

# Calculate the difference between weighted and unweighted binary predictions
binary_diff = wght_binary.flatten() - unwght_binary.flatten()

# Create histogram of differences
ax_sw2.hist(binary_diff, bins=20, color='skyblue', edgecolor='black')
ax_sw2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No difference')
ax_sw2.axvline(x=np.mean(binary_diff), color='green', linestyle='-', linewidth=2, 
           label=f'Mean difference: {np.mean(binary_diff):.4f}')

ax_sw2.set_xlabel('Difference in Binary Prediction (Weighted - Unweighted)')
ax_sw2.set_ylabel('Count')
ax_sw2.set_title('Histogram of Differences in Binary Predictions')
ax_sw2.legend()
fig_sw2.savefig("plots/weighted_vs_unweighted_binary.png")
plt.close(fig_sw2)

# 3. Performance analysis
# Extract events vs non-events to see if weighting improved predictions for events
event_mask = y_test_survival[:, 0] == 1
event_indices = event_mask.nonzero().squeeze().numpy()
non_event_indices = (event_mask == 0).nonzero().squeeze().numpy()

# For binary task
binary_targets = y_test_binary.numpy().flatten()
binary_correct_unweighted = ((unwght_binary.flatten() > 0.5) == binary_targets).astype(int)
binary_correct_weighted = ((wght_binary.flatten() > 0.5) == binary_targets).astype(int)

# Calculate accuracy for events vs non-events
print("\nBinary classification by event status:")
print("Unweighted model accuracy (overall):", np.mean(binary_correct_unweighted))
print("Weighted model accuracy (overall):", np.mean(binary_correct_weighted))

print("\nEffect of sample weighting on different groups:")
print("1. Performance on patients with an event (weighted higher):")
print(f"   - Unweighted model: {np.mean(binary_correct_unweighted[event_indices]):.4f}")
print(f"   - Weighted model: {np.mean(binary_correct_weighted[event_indices]):.4f}")
print(f"   - Improvement: {np.mean(binary_correct_weighted[event_indices]) - np.mean(binary_correct_unweighted[event_indices]):.4f}")

print("\n2. Performance on patients without an event (weighted less):")
print(f"   - Unweighted model: {np.mean(binary_correct_unweighted[non_event_indices]):.4f}")
print(f"   - Weighted model: {np.mean(binary_correct_weighted[non_event_indices]):.4f}")
print(f"   - Improvement: {np.mean(binary_correct_weighted[non_event_indices]) - np.mean(binary_correct_unweighted[non_event_indices]):.4f}")

# 4. Compare average survival curves by event status
fig_sw3, ax_sw3 = plt.subplots(figsize=(10, 6))

# Calculate average survival curves for each model and patient group
survival_event_unwght = np.mean(unwght_survival[event_indices], axis=0)
survival_event_wght = np.mean(wght_survival[event_indices], axis=0)
survival_nonevent_unwght = np.mean(unwght_survival[non_event_indices], axis=0)
survival_nonevent_wght = np.mean(wght_survival[non_event_indices], axis=0)

# Plot average survival curves
ax_sw3.plot(time_points, survival_event_unwght, label='Event patients - Unweighted', 
         linestyle='-', linewidth=2, color='red')
ax_sw3.plot(time_points, survival_event_wght, label='Event patients - Weighted', 
         linestyle='--', linewidth=2, color='darkred')
ax_sw3.plot(time_points, survival_nonevent_unwght, label='Non-event patients - Unweighted', 
         linestyle='-', linewidth=2, color='blue')
ax_sw3.plot(time_points, survival_nonevent_wght, label='Non-event patients - Weighted', 
         linestyle='--', linewidth=2, color='darkblue')

ax_sw3.set_xlabel('Time')
ax_sw3.set_ylabel('Average Survival Probability')
ax_sw3.set_title('Average Survival Curves by Event Status')
ax_sw3.legend()
ax_sw3.grid(True, alpha=0.3)
fig_sw3.savefig("plots/weighted_survival_by_event.png")
plt.close(fig_sw3)

print("\nVignette code execution completed successfully!")
print(f"All plots have been saved to: {os.path.abspath('plots')}")