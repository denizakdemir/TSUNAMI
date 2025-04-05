import os
import sys
import unittest
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, Dataset

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
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


class TestFeatureImportance(unittest.TestCase):
    """Test cases for feature importance methods."""
    
    def setUp(self):
        """Set up test environment with a simple model and data."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Generate simple synthetic data
        n_samples = 100
        n_features = 5
        
        # Features (5 continuous features)
        self.X = np.random.randn(n_samples, n_features)
        
        # Generate survival times with first and third features being most important
        risk_scores = 2 * self.X[:, 0] + 1.5 * self.X[:, 2]
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
        self.df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(n_features)])
        self.df['time'] = times
        self.df['event'] = events
        self.df['time_bin'] = time_bins
        
        # Create target format for the model
        # [event_indicator, time_bin, one_hot_encoding]
        self.target = np.zeros((n_samples, 2 + num_bins))
        self.target[:, 0] = events
        self.target[:, 1] = time_bins
        
        # One-hot encoding of time
        for i in range(n_samples):
            if events[i]:
                # For events, mark the event time
                self.target[i, 2 + time_bins[i]] = 1
            else:
                # For censored, mark all times after censoring as unknown (-1)
                self.target[i, 2 + time_bins[i]:] = -1
        
        # Create data processor
        self.processor = DataProcessor(
            num_impute_strategy='mean',
            normalize='robust'
        )
        self.processor.fit(self.df)
        
        # Process features
        self.df_processed = self.processor.transform(self.df)
        
        # Convert to tensors
        self.X_tensor = torch.tensor(
            self.df_processed[[f'feature_{i}' for i in range(n_features)]].values, 
            dtype=torch.float32
        )
        self.target_tensor = torch.tensor(self.target, dtype=torch.float32)
        
        # Create dataset and dataloader with proper formatting
        # The model expects a dictionary with 'continuous' and 'targets' keys
        class SurvivalDataset(Dataset):
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
        
        # Create the custom dataset
        custom_dataset = SurvivalDataset(self.X_tensor, self.target_tensor)
        self.loader = DataLoader(custom_dataset, batch_size=32)
        
        # Create model
        self.task_head = SingleRiskHead(
            name='survival',
            input_dim=64,
            num_time_bins=num_bins,
            alpha_rank=0.1
        )
        
        self.model = EnhancedDeepHit(
            num_continuous=n_features,
            targets=[self.task_head],
            encoder_dim=64,
            encoder_depth=2,
            encoder_heads=4,
            device='cpu'
        )
        
        # Train model for a few epochs
        self.model.fit(
            train_loader=self.loader,
            num_epochs=2,
            learning_rate=0.001
        )
    
    def test_permutation_importance(self):
        """Test permutation importance calculator."""
        # Initialize permutation importance calculator
        perm_imp = PermutationImportance(self.model)
        
        # Calculate importances
        inputs = {'continuous': self.X_tensor}
        targets = {'survival': self.target_tensor}
        importances = perm_imp.compute_importance(
            inputs, 
            targets,
            n_repeats=2,
            feature_names=[f'feature_{i}' for i in range(5)]
        )
        
        # Check that importances are calculated for all features
        self.assertEqual(len(importances), 5)
        
        # Check that feature_0 and feature_2 have higher importance
        # (these were the most important in data generation)
        feature_0_importance = importances.get('feature_0', 0)
        feature_2_importance = importances.get('feature_2', 0)
        feature_1_importance = importances.get('feature_1', 0)
        
        # These features should have higher importance than others
        self.assertGreater(feature_0_importance, feature_1_importance)
        self.assertGreater(feature_2_importance, feature_1_importance)
        
        # Check that importances can be plotted
        fig = perm_imp.plot_importance(importances)
        self.assertIsNotNone(fig)
    
    def test_shap_importance(self):
        """Test SHAP importance calculator."""
        # Initialize SHAP importance calculator
        shap_imp = ShapImportance(self.model)
        
        # Calculate importances
        inputs = {'continuous': self.X_tensor}
        shap_values = shap_imp.compute_importance(
            inputs,
            n_samples=10,
            feature_names=[f'feature_{i}' for i in range(5)]
        )
        
        # Check that values are returned for all features
        self.assertEqual(len(shap_values), 5)
        
        # Check that SHAP values can be plotted
        fig = shap_imp.plot_importance(shap_values)
        self.assertIsNotNone(fig)
        
        # Check that feature_0 and feature_2 have higher absolute SHAP values
        # (these were the most important in data generation)
        abs_shap_values = {name: np.abs(value).mean() for name, value in shap_values.items()}
        feature_0_importance = abs_shap_values.get('feature_0', 0)
        feature_2_importance = abs_shap_values.get('feature_2', 0)
        feature_1_importance = abs_shap_values.get('feature_1', 0)
        
        # These features should have higher importance than others
        self.assertGreater(feature_0_importance, feature_1_importance)
        self.assertGreater(feature_2_importance, feature_1_importance)
    
    def test_integrated_gradients(self):
        """Test integrated gradients calculator."""
        # Initialize integrated gradients calculator
        ig_imp = IntegratedGradients(self.model)
        
        # Calculate importances
        inputs = {'continuous': self.X_tensor[0:1]}  # Single sample
        attributions = ig_imp.compute_importance(
            inputs,
            target_class='risk_score',
            feature_names=[f'feature_{i}' for i in range(5)],
            n_steps=10
        )
        
        # Check that attributions are calculated for all features
        self.assertEqual(len(attributions), 5)
        
        # Check that attributions can be plotted
        fig = ig_imp.plot_importance(attributions)
        self.assertIsNotNone(fig)
    
    def test_attention_importance(self):
        """Test attention-based importance calculator."""
        # Initialize attention importance calculator
        attn_imp = AttentionImportance(self.model)
        
        # Calculate importances
        inputs = {'continuous': self.X_tensor[0:5]}  # First 5 samples
        attention_scores = attn_imp.compute_importance(
            inputs,
            feature_names=[f'feature_{i}' for i in range(5)],
            layer_idx=1  # Use the last transformer layer
        )
        
        # Check that importance scores are calculated for all features
        self.assertEqual(len(attention_scores), 5)
        
        # Check that attention scores can be plotted
        fig = attn_imp.plot_importance(attention_scores)
        self.assertIsNotNone(fig)


if __name__ == '__main__':
    unittest.main()