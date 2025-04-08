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
from enhanced_deephit.visualization.feature_effects import plot_partial_dependence

def generate_synthetic_data(n_samples=500, n_features=5, include_categorical=True, include_weights=True, seed=42):
    """Generate synthetic survival data with known important features."""
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate continuous features
    X_continuous = np.random.randn(n_samples, n_features)
    
    # Create a dataframe
    df = pd.DataFrame(X_continuous, columns=[f'continuous_{i}' for i in range(n_features)])
    
    # Add categorical features if requested
    if include_categorical:
        # Create a categorical feature with 3 categories
        categories = ['Low', 'Medium', 'High']
        df['cat_risk'] = np.random.choice(categories, size=n_samples)
        
        # Create another categorical feature with 2 categories
        df['cat_group'] = np.random.choice(['Group A', 'Group B'], size=n_samples)
        
        # Make categorical features have an effect on survival
        risk_modifier = {'Low': -0.5, 'Medium': 0.0, 'High': 1.0}
        group_modifier = {'Group A': 0.0, 'Group B': 0.5}
        
        cat_risk_effect = np.array([risk_modifier[val] for val in df['cat_risk']])
        cat_group_effect = np.array([group_modifier[val] for val in df['cat_group']])
        
        # Generate survival times with continuous and categorical features being important
        risk_scores = (
            2 * X_continuous[:, 0] + 
            1.5 * X_continuous[:, 2] + 
            cat_risk_effect +
            cat_group_effect
        )
    else:
        # Without categorical features, use only continuous ones
        risk_scores = 2 * X_continuous[:, 0] + 1.5 * X_continuous[:, 2]
    
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
    
    # Add time-related columns
    df['time'] = times
    df['event'] = events
    df['time_bin'] = time_bins
    
    # Generate sample weights if requested
    sample_weights = None
    if include_weights:
        # Generate weights based on a combination of features to demonstrate importance
        # For example, give more weight to samples with higher values of feature_0
        weights = np.exp(X_continuous[:, 0]) / np.sum(np.exp(X_continuous[:, 0]))
        weights = weights * n_samples  # Scale to have average weight of 1.0
        df['sample_weight'] = weights
        sample_weights = weights
    
    return df, num_bins, time_bins, events, sample_weights

def prepare_data(df, num_bins, time_bins, events, sample_weights=None):
    """Prepare data for the model."""
    # Create data processor
    processor = DataProcessor(
        num_impute_strategy='mean',
        normalize='robust',
        cat_embed_dim='auto',  # Automatically determine embedding dimensions
        cat_impute_strategy='most_frequent'
    )
    
    # Identify feature columns and exclude target and sample weight columns
    feature_cols = [col for col in df.columns if col.startswith('continuous_') or col.startswith('cat_')]
    
    # Fit the processor on only the feature columns
    processor.fit(df[feature_cols])
    
    # Process features
    df_processed = processor.transform(df[feature_cols])
    
    # Extract features and convert to tensor
    X_tensor = torch.tensor(df_processed.values, dtype=torch.float32)
    
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
    
    # Convert sample weights to tensor if provided
    weights_tensor = None
    if sample_weights is not None:
        weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)
    
    # Get information about categorical features
    cat_feature_info = []
    for col, info in processor.cat_embed_info.items():
        # Store information about each categorical variable
        cat_feature_info.append({
            'name': col,
            'original_name': col,
            'cardinality': info['cardinality'],
            'embed_dim': info['embed_dim'],
            'reverse_mapping': info.get('reverse_mapping', {})
        })
    
    return X_tensor, target_tensor, processor, cat_feature_info, weights_tensor

def create_and_train_model(X_tensor, target_tensor, num_bins, weights_tensor=None):
    """Create and train the DeepHit model."""
    # Create dataset and dataloader with proper formatting
    class SurvivalDataset(torch.utils.data.Dataset):
        def __init__(self, X, targets, weights=None):
            self.X = X
            self.targets = targets
            self.weights = weights
            
        def __len__(self):
            return len(self.X)
            
        def __getitem__(self, idx):
            item = {
                'continuous': self.X[idx],
                'targets': {
                    'survival': self.targets[idx]
                }
            }
            
            if self.weights is not None:
                item['sample_weights'] = self.weights[idx]
                
            return item
    
    # Create the dataset and dataloader
    dataset = SurvivalDataset(X_tensor, target_tensor, weights_tensor)
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

def calculate_and_plot_importance_scores(model, X_tensor, target_tensor, feature_names, cat_feature_info, weights_tensor=None):
    """Calculate and plot feature importance scores."""
    # Permutation Importance
    print("\nCalculating Permutation Importance...")
    perm_imp = PermutationImportance(model)
    inputs = {'continuous': X_tensor}
    targets = {'survival': target_tensor}
    
    # Without sample weights
    perm_importances = perm_imp.compute_importance(
        inputs, 
        targets,
        n_repeats=3,
        feature_names=feature_names,
        cat_feature_info=cat_feature_info,
        use_original_names=True
    )
    
    # With sample weights (if provided)
    perm_importances_weighted = None
    if weights_tensor is not None:
        print("\nCalculating Weighted Permutation Importance...")
        perm_importances_weighted = perm_imp.compute_importance(
            inputs, 
            targets,
            n_repeats=3,
            feature_names=feature_names,
            cat_feature_info=cat_feature_info,
            use_original_names=True,
            sample_weights=weights_tensor
        )
    
    # Plot permutation importance
    fig1 = perm_imp.plot_importance(perm_importances)
    plt.title("Permutation Importance (with Original Variable Names)")
    plt.tight_layout()
    
    # Save the plot
    fig1.savefig("permutation_importance.png")
    plt.close(fig1)
    print("Permutation importance calculated and saved as 'permutation_importance.png'")
    
    # Plot weighted permutation importance if available
    if perm_importances_weighted is not None:
        fig1b = perm_imp.plot_importance(perm_importances_weighted)
        plt.title("Weighted Permutation Importance (with Original Variable Names)")
        plt.tight_layout()
        
        # Save the plot
        fig1b.savefig("weighted_permutation_importance.png")
        plt.close(fig1b)
        print("Weighted permutation importance calculated and saved as 'weighted_permutation_importance.png'")
    
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
    
    # Plot categorical feature effects
    print("\nPlotting Categorical Feature Effects...")
    
    # Find embedding dimensions for categorical features to plot their effects
    for cat_info in cat_feature_info:
        # Only proceed if we have mapping information
        if 'reverse_mapping' in cat_info and cat_info['reverse_mapping']:
            # Find the index of the first embedding dimension for this categorical feature
            # in the feature names list
            embed_prefix = f"{cat_info['name']}_embed_"
            cat_feature_indices = []
            
            for i, name in enumerate(feature_names):
                if name.startswith(embed_prefix):
                    cat_feature_indices.append(i)
            
            if cat_feature_indices:
                # Use the first embedding dimension for plotting
                cat_idx = cat_feature_indices[0]
                
                # Plot partial dependence for this categorical variable
                fig_cat = plot_partial_dependence(
                    model=model,
                    X=X_tensor,
                    feature_idx=cat_idx,
                    feature_name=cat_info['name'],
                    target='risk_score',
                    categorical_info=cat_info,
                    n_points=len(cat_info['reverse_mapping']),
                    figsize=(10, 8),
                    title=f"Effect of {cat_info['original_name']} on Risk Score",
                    sample_weights=weights_tensor
                )
                
                # Save the plot
                fig_cat.savefig(f"categorical_effect_{cat_info['name']}.png")
                plt.close(fig_cat)
                print(f"Categorical effect plot for {cat_info['name']} saved as 'categorical_effect_{cat_info['name']}.png'")
    
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
    print("Feature Importance Example with Sample Weights")
    print("=============================================\n")
    
    # Step 1: Generate synthetic data
    print("Generating synthetic data...")
    df, num_bins, time_bins, events, sample_weights = generate_synthetic_data(
        n_samples=500, 
        n_features=5, 
        include_categorical=True, 
        include_weights=True
    )
    print(f"Generated data with {len(df)} samples, {len([c for c in df.columns if c.startswith('continuous_')])} continuous features, and {len([c for c in df.columns if c.startswith('cat_')])} categorical features")
    
    if sample_weights is not None:
        print(f"Generated sample weights with range [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")
    
    # Step 2: Prepare data for the model
    print("\nPreparing data...")
    X_tensor, target_tensor, processor, cat_feature_info, weights_tensor = prepare_data(df, num_bins, time_bins, events, sample_weights)
    
    # Get feature names from the processor
    feature_names = processor.get_feature_names_out()
    print(f"Processed features ({len(feature_names)}): {feature_names}")
    
    # Print categorical variable information
    print("\nCategorical feature information:")
    for cat_info in cat_feature_info:
        print(f"  - {cat_info['original_name']}: {len(cat_info.get('reverse_mapping', {}))} categories")
        if 'reverse_mapping' in cat_info:
            print(f"    Categories: {list(cat_info['reverse_mapping'].values())}")
    
    # Step 3: Create and train model
    model = create_and_train_model(X_tensor, target_tensor, num_bins, weights_tensor)
    
    # Step 4: Calculate and plot importance scores
    calculate_and_plot_importance_scores(model, X_tensor, target_tensor, feature_names, cat_feature_info, weights_tensor)
    
    print("\nFeature importance example completed successfully with categorical variable and sample weights support.")

if __name__ == "__main__":
    main()