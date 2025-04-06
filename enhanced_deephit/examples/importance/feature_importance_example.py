import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from enhanced_deephit.data.processing import DataProcessor
from enhanced_deephit.models import EnhancedDeepHit
from enhanced_deephit.models.tasks.survival import SingleRiskHead
from enhanced_deephit.visualization.importance.importance import (
    PermutationImportance,
    ShapImportance,
    IntegratedGradients,
    AttentionImportance
)

def generate_synthetic_data(n_samples=500, n_features=5, seed=42):
    """Generate synthetic survival data with known important features."""
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate survival times with first and third features being most important
    risk_scores = 2 * X[:, 0] + 1.5 * X[:, 2]
    survival_times = np.random.exponential(scale=np.exp(-risk_scores))
    censoring_times = np.random.exponential(scale=2)
    
    # Determine observed time and event indicator
    times = np.minimum(survival_times, censoring_times)
    events = (survival_times <= censoring_times).astype(np.float32)
    
    # Discretize time into bins
    num_bins = 10
    max_time = np.percentile(times, 99)
    bin_edges = np.linspace(0, max_time, num_bins + 1)
    time_bins = np.digitize(times, bin_edges) - 1
    time_bins = np.clip(time_bins, 0, num_bins - 1)
    
    # Create dataframe
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['time'] = times
    df['event'] = events
    df['time_bin'] = time_bins
    
    return df, num_bins, time_bins, events

def prepare_data(df, num_bins, time_bins, events):
    """Prepare data for the model."""
    # Create data processor
    processor = DataProcessor(
        num_impute_strategy='mean',
        normalize='robust'
    )
    processor.fit(df)
    
    # Process features
    df_processed = processor.transform(df)
    
    # Extract features and convert to tensor
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X_tensor = torch.tensor(df_processed[feature_cols].values, dtype=torch.float32)
    
    # Create target format for the model
    # [event_indicator, time_bin, one_hot_encoding]
    target = np.zeros((len(df), 2 + num_bins))
    target[:, 0] = events
    target[:, 1] = time_bins
    
    # One-hot encoding of time
    for i in range(len(df)):
        if events[i]:
            # For events, mark the event time
            target[i, 2 + int(time_bins[i])] = 1
        else:
            # For censored, mark all times after censoring as unknown (-1)
            target[i, 2 + int(time_bins[i]):] = -1
    
    # Convert to tensor
    target_tensor = torch.tensor(target, dtype=torch.float32)
    
    return X_tensor, target_tensor, processor

def create_and_train_model(X_tensor, target_tensor, num_bins):
    """Create and train the DeepHit model."""
    # Create dataset and dataloader with proper formatting
    class SurvivalDataset(torch.utils.data.Dataset):
        def __init__(self, X, targets):
            self.X = X
            self.targets = targets
            
        def __len__(self):
            return len(self.X)
            
        def __getitem__(self, idx):
            return {
                'continuous': self.X[idx],
                'targets': {
                    'survival': self.targets[idx]
                }
            }
    
    # Create the dataset and dataloader
    dataset = SurvivalDataset(X_tensor, target_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create single risk model
    task_head = SingleRiskHead(
        name='survival',
        input_dim=64,
        num_time_bins=num_bins,
        alpha_rank=0.1
    )
    
    model = EnhancedDeepHit(
        num_continuous=X_tensor.shape[1],
        targets=[task_head],
        encoder_dim=64,
        encoder_depth=2,
        encoder_heads=4,
        device='cpu'
    )
    
    # Train model
    print("Training model...")
    model.fit(
        train_loader=loader,
        num_epochs=5,
        learning_rate=0.001
    )
    
    return model

def calculate_and_plot_importance_scores(model, X_tensor, target_tensor, feature_names):
    """Calculate and plot feature importance scores."""
    # Permutation Importance
    print("\nCalculating Permutation Importance...")
    perm_imp = PermutationImportance(model)
    inputs = {'continuous': X_tensor}
    targets = {'survival': target_tensor}
    
    perm_importances = perm_imp.compute_importance(
        inputs, 
        targets,
        n_repeats=3,
        feature_names=feature_names
    )
    
    # Plot permutation importance
    fig1 = perm_imp.plot_importance(perm_importances)
    plt.title("Permutation Importance")
    plt.tight_layout()
    
    # Save the plot
    fig1.savefig("permutation_importance.png")
    plt.close(fig1)
    print("Permutation importance calculated and saved as 'permutation_importance.png'")
    
    # SHAP Importance (on a subset for speed)
    print("\nCalculating SHAP Importance...")
    shap_imp = ShapImportance(model)
    
    # Use a smaller subset for SHAP (it can be slow)
    subset_size = min(100, len(X_tensor))
    X_subset = X_tensor[:subset_size]
    inputs_subset = {'continuous': X_subset}
    
    shap_values = shap_imp.compute_importance(
        inputs_subset,
        n_samples=20,  # Reduced for speed
        feature_names=feature_names
    )
    
    # Plot SHAP importance
    fig2 = shap_imp.plot_importance(shap_values)
    plt.title("SHAP Importance")
    plt.tight_layout()
    
    # Save the plot
    fig2.savefig("shap_importance.png")
    plt.close(fig2)
    print("SHAP importance calculated and saved as 'shap_importance.png'")
    
    # Integrated Gradients (for a single sample)
    print("\nCalculating Integrated Gradients...")
    ig_imp = IntegratedGradients(model)
    
    # Use just one sample for IG
    single_sample = {'continuous': X_tensor[0:1]}
    
    attributions = ig_imp.compute_importance(
        single_sample,
        target_class='risk_score',
        feature_names=feature_names,
        n_steps=20  # Reduced for speed
    )
    
    # Plot integrated gradients
    fig3 = ig_imp.plot_importance(attributions)
    plt.title("Integrated Gradients (Single Sample)")
    plt.tight_layout()
    
    # Save the plot
    fig3.savefig("integrated_gradients.png")
    plt.close(fig3)
    print("Integrated gradients calculated and saved as 'integrated_gradients.png'")
    
    # Attention Importance (for a small batch)
    print("\nCalculating Attention Importance...")
    attn_imp = AttentionImportance(model)
    
    # Use a small batch
    small_batch = {'continuous': X_tensor[:5]}
    
    attention_scores = attn_imp.compute_importance(
        small_batch,
        feature_names=feature_names,
        layer_idx=-1  # Use the last transformer layer
    )
    
    # Plot attention importance
    fig4 = attn_imp.plot_importance(attention_scores)
    plt.title("Attention Importance")
    plt.tight_layout()
    
    # Save the plot
    fig4.savefig("attention_importance.png")
    plt.close(fig4)
    print("Attention importance calculated and saved as 'attention_importance.png'")

def main():
    print("Feature Importance Example")
    print("=========================\n")
    
    # Step 1: Generate synthetic data
    print("Generating synthetic data...")
    df, num_bins, time_bins, events = generate_synthetic_data(n_samples=500, n_features=5)
    print(f"Generated data with {len(df)} samples and {df.shape[1]-3} features")
    
    # Step 2: Prepare data for the model
    print("\nPreparing data...")
    X_tensor, target_tensor, processor = prepare_data(df, num_bins, time_bins, events)
    
    # Feature names
    feature_names = [f'feature_{i}' for i in range(X_tensor.shape[1])]
    print(f"Features: {feature_names}")
    
    # Step 3: Create and train model
    model = create_and_train_model(X_tensor, target_tensor, num_bins)
    
    # Step 4: Calculate and plot importance scores
    calculate_and_plot_importance_scores(model, X_tensor, target_tensor, feature_names)
    
    print("\nFeature importance example completed successfully.")

if __name__ == "__main__":
    main()