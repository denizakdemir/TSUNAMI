import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from data.processing import DataProcessor
from models import EnhancedDeepHit
from models.tasks.survival import SingleRiskHead
from models.tasks.standard import ClassificationHead, RegressionHead
from simulation.data_generation import generate_survival_data, add_missing_values


def generate_multi_task_data(n_samples=1000, n_features=20, censoring_rate=0.3):
    """Generate synthetic data for multi-task learning"""
    # Generate survival data first
    data, survival_target, num_bins = generate_survival_data(
        n_samples=n_samples,
        n_features=n_features,
        censoring_rate=censoring_rate,
        include_categorical=True,
        missing_rate=0.1
    )
    
    # Extract features for generating additional targets
    feature_cols = [col for col in data.columns if col.startswith('feature_')]
    X = data[feature_cols].values
    
    # Extract actual number of features from the data
    n_features_actual = X.shape[1]
    
    # Generate coefficients for other tasks
    beta_class = np.random.uniform(-1, 1, n_features_actual)
    beta_reg = np.random.uniform(-1, 1, n_features_actual)
    
    # Generate binary classification outcome
    logits_class = np.dot(X, beta_class)
    # Clip to avoid overflow in exp function
    logits_class_clipped = np.clip(logits_class, -10, 10)
    prob_class = 1 / (1 + np.exp(-logits_class_clipped))
    class_label = (prob_class > 0.5).astype(np.float32)
    
    # Generate regression outcome
    reg_value = np.dot(X, beta_reg) + np.random.normal(0, 0.5, n_samples)
    # Replace any potential NaNs
    reg_value = np.nan_to_num(reg_value, nan=0.0)
    
    # Add new targets to dataframe
    data['class_label'] = class_label
    data['reg_value'] = reg_value
    
    # Add masks for multi-task learning (simulate missing targets)
    survival_mask = np.random.random(n_samples) < 0.9  # 90% have survival data
    class_mask = np.random.random(n_samples) < 0.8     # 80% have classification data
    reg_mask = np.random.random(n_samples) < 0.85      # 85% have regression data
    
    # Encode classification target
    class_target = class_label.reshape(-1, 1)
    
    # Set targets for samples without classification data to zeros
    class_target[~class_mask] = 0
    
    # Encode regression target
    reg_target = reg_value.reshape(-1, 1)
    
    # Set targets for samples without regression data to zeros
    reg_target[~reg_mask] = 0
    
    # Set targets for samples without survival data to zeros
    survival_target[~survival_mask] = 0
    
    # Create masks for training
    masks = {
        'survival': survival_mask,
        'classification': class_mask,
        'regression': reg_mask
    }
    
    return data, {
        'survival': survival_target,
        'classification': class_target,
        'regression': reg_target
    }, masks, num_bins


class MultiTaskDataset(Dataset):
    """Dataset for multi-task learning"""
    
    def __init__(self, df, targets, masks, processor=None):
        """
        Initialize dataset
        
        Parameters
        ----------
        df : pd.DataFrame
            Input features
            
        targets : dict
            Dictionary mapping task names to target arrays
            
        masks : dict
            Dictionary mapping task names to boolean masks
            
        processor : DataProcessor, optional
            Data processor for transforming features
        """
        self.targets = {
            name: torch.tensor(target, dtype=torch.float32)
            for name, target in targets.items()
        }
        
        self.masks = {
            name: torch.tensor(mask, dtype=torch.float32)
            for name, mask in masks.items()
        }
        
        if processor is not None:
            # Transform features
            self.df_processed = processor.transform(df)
            
            # Get categorical feature info for the encoder
            self.cat_feat_info = []
            for col, info in processor.cat_embed_info.items():
                self.cat_feat_info.append({
                    'name': col,
                    'cardinality': info['cardinality'],
                    'embed_dim': info['embed_dim']
                })
        else:
            # Use raw features
            self.df_processed = df
            self.cat_feat_info = []
        
        # Extract continuous and categorical features
        self.continuous_cols = [col for col in self.df_processed.columns 
                                if not col.startswith('cat_') and 
                                not col.endswith('_missing')]
        
        self.missing_indicator_cols = [col for col in self.df_processed.columns 
                                       if col.endswith('_missing')]
        
        # Convert to tensors
        self.continuous = torch.tensor(
            self.df_processed[self.continuous_cols].values, 
            dtype=torch.float32
        )
        
        if self.missing_indicator_cols:
            self.missing_mask = torch.tensor(
                self.df_processed[self.missing_indicator_cols].values,
                dtype=torch.float32
            )
        else:
            self.missing_mask = None
    
    def __len__(self):
        return len(self.continuous)
    
    def __getitem__(self, idx):
        item = {
            'continuous': self.continuous[idx],
            'targets': {
                name: target[idx]
                for name, target in self.targets.items()
            },
            'masks': {
                name: mask[idx]
                for name, mask in self.masks.items()
            }
        }
        
        # Let's skip the missing mask for simplicity in this example
        # In a real implementation, you would want to make sure the mask
        # is properly shaped to match what the model expects
        
        return item


def main():
    # Generate synthetic data
    np.random.seed(42)
    data, targets, masks, num_bins = generate_multi_task_data(
        n_samples=500, n_features=10
    )
    
    # Split data into train, validation, and test sets
    indices = np.arange(len(data))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42
    )
    
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.2, random_state=42
    )
    
    # Split data and targets
    train_data = data.iloc[train_idx].reset_index(drop=True)
    val_data = data.iloc[val_idx].reset_index(drop=True)
    test_data = data.iloc[test_idx].reset_index(drop=True)
    
    train_targets = {
        name: target[train_idx]
        for name, target in targets.items()
    }
    
    val_targets = {
        name: target[val_idx]
        for name, target in targets.items()
    }
    
    test_targets = {
        name: target[test_idx]
        for name, target in targets.items()
    }
    
    train_masks = {
        name: mask[train_idx]
        for name, mask in masks.items()
    }
    
    val_masks = {
        name: mask[val_idx]
        for name, mask in masks.items()
    }
    
    test_masks = {
        name: mask[test_idx]
        for name, mask in masks.items()
    }
    
    # Initialize data processor
    processor = DataProcessor(
        num_impute_strategy='mean',
        cat_impute_strategy='most_frequent',
        normalize='robust',
        time_features=None,
        cat_embed_dim='auto',
        create_missing_indicators=True
    )
    
    # Fit processor on training data
    processor.fit(train_data)
    
    # Create datasets
    train_dataset = MultiTaskDataset(train_data, train_targets, train_masks, processor)
    val_dataset = MultiTaskDataset(val_data, val_targets, val_masks, processor)
    test_dataset = MultiTaskDataset(test_data, test_targets, test_masks, processor)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )
    
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize task heads
    survival_head = SingleRiskHead(
        name='survival',
        input_dim=128,
        num_time_bins=num_bins,
        alpha_rank=0.1,
        task_weight=1.0
    )
    
    classification_head = ClassificationHead(
        name='classification',
        input_dim=128,
        num_classes=2,
        task_weight=1.0
    )
    
    regression_head = RegressionHead(
        name='regression',
        input_dim=128,
        output_dim=1,
        loss_type='mse',
        task_weight=1.0
    )
    
    # Initialize model
    model = EnhancedDeepHit(
        num_continuous=len(train_dataset.continuous_cols),
        targets=[survival_head, classification_head, regression_head],
        cat_feat_info=train_dataset.cat_feat_info,
        encoder_dim=128,
        encoder_depth=4,
        encoder_heads=8,
        encoder_feature_interaction=True,
        include_variational=False,  # Disable variational component to avoid NaN issues
        device=device
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-3,
        weight_decay=1e-4,
        num_epochs=5,  # Reduced epochs for faster testing
        patience=2
    )
    
    # Evaluate on test set
    test_loss, test_metrics = model.evaluate(test_loader)
    print(f"\nTest Loss: {test_loss:.4f}")
    print("Test Metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig('multi_task_training_history.png')
    
    # Save model
    model_path = 'model_outputs/multi_task_model'
    os.makedirs('model_outputs', exist_ok=True)
    model.save(model_path, save_processor=True, processor=processor)
    print(f"Model saved to '{model_path}.pt'")

    
    # Generate predictions for a sample
    sample_idx = 0
    sample = test_dataset[sample_idx]
    
    continuous = sample['continuous'].unsqueeze(0).to(device)
    
    # Get predictions with uncertainty
    print("\nComputing predictions with uncertainty...")
    uncertainty = model.compute_uncertainty(
        continuous, num_samples=10  # Reduced samples for faster testing
    )
    
    # Extract survival predictions
    survival_uncertainty = uncertainty.get('survival', {})
    if 'mean' in survival_uncertainty and 'std' in survival_uncertainty:
        survival_mean = survival_uncertainty['mean'].squeeze().cpu().numpy()
        survival_std = survival_uncertainty['std'].squeeze().cpu().numpy()
        
        # Plot survival curve with uncertainty
        time_points = np.arange(num_bins)
        plt.figure(figsize=(10, 6))
        plt.step(time_points, survival_mean, where='post', label='Predicted Survival')
        plt.fill_between(
            time_points, 
            np.clip(survival_mean - 2 * survival_std, 0, 1), 
            np.clip(survival_mean + 2 * survival_std, 0, 1),
            alpha=0.3, 
            label='95% Confidence Interval'
        )
        plt.xlabel('Time Bin')
        plt.ylabel('Survival Probability')
        plt.title('Survival Curve with Uncertainty')
        plt.grid(True)
        plt.legend()
        plt.savefig('multi_task_survival_uncertainty.png')
    
    # Extract classification predictions
    class_uncertainty = uncertainty.get('classification', {})
    if 'mean' in class_uncertainty and 'std' in class_uncertainty:
        class_mean = class_uncertainty['mean'].squeeze().cpu().numpy()
        class_std = class_uncertainty['std'].squeeze().cpu().numpy()
        
        # Print classification prediction with uncertainty
        print(f"\nClassification prediction: {class_mean:.4f} ± {2*class_std:.4f}")
    
    # Extract regression predictions
    reg_uncertainty = uncertainty.get('regression', {})
    if 'mean' in reg_uncertainty and 'std' in reg_uncertainty:
        reg_mean = reg_uncertainty['mean'].squeeze().cpu().numpy()
        reg_std = reg_uncertainty['std'].squeeze().cpu().numpy()
        
        # Print regression prediction with uncertainty
        print(f"Regression prediction: {reg_mean:.4f} ± {2*reg_std:.4f}")
    
    print("Done!")


if __name__ == "__main__":
    main()