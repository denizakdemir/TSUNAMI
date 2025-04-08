import os
import sys
import unittest
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from enhanced_deephit.data.processing import DataProcessor
from enhanced_deephit.models import EnhancedDeepHit
from enhanced_deephit.models.tasks.survival import SingleRiskHead
from enhanced_deephit.visualization.survival_plots import (
    plot_survival_curve,
    plot_cumulative_incidence,
    plot_calibration_curve
)
# Commented out until these functions are implemented
# from enhanced_deephit.visualization.uncertainty_plots import (
#     plot_prediction_intervals,
#     plot_uncertainty_heatmap
# )
from enhanced_deephit.visualization.feature_effects import (
    plot_partial_dependence,
    plot_ice_curves,
    plot_feature_interaction
)


class TestVisualizationFunctions(unittest.TestCase):
    """Test cases for visualization functions."""
    
    def setUp(self):
        """Set up test environment with models and data."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Generate synthetic data for single risk
        n_samples = 100
        n_features = 5
        
        # Features
        self.X = np.random.randn(n_samples, n_features)
        
        # Generate survival times
        risk_scores = 2 * self.X[:, 0] + 1.5 * self.X[:, 2]
        survival_times = np.random.exponential(scale=np.exp(-risk_scores))
        censoring_times = np.random.exponential(scale=2)
        
        # Determine observed time and event indicator
        times = np.minimum(survival_times, censoring_times)
        events = (survival_times <= censoring_times).astype(np.float32)
        
        # Discretize time into bins
        self.num_bins = 10
        max_time = np.percentile(times, 99)
        bin_edges = np.linspace(0, max_time, self.num_bins + 1)
        time_bins = np.digitize(times, bin_edges) - 1
        time_bins = np.clip(time_bins, 0, self.num_bins - 1)
        
        # Create dataframe
        self.df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(n_features)])
        self.df['time'] = times
        self.df['event'] = events
        self.df['time_bin'] = time_bins
        
        # Create target format for the model
        # [event_indicator, time_bin, one_hot_encoding]
        self.target = np.zeros((n_samples, 2 + self.num_bins))
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
        
        # Generate synthetic data for competing risks
        # Create two causes (0 and 1)
        causes = np.random.choice([0, 1], size=n_samples)
        
        # For competing risks, the target format is different
        # [event_indicator, time_bin, cause_index, one_hot_encoding]
        self.cr_target = np.zeros((n_samples, 3 + 2 * self.num_bins))
        self.cr_target[:, 0] = events
        self.cr_target[:, 1] = time_bins
        
        # Set cause index (-1 for censored)
        self.cr_target[:, 2] = np.where(events, causes, -1)
        
        # Create data processor
        self.processor = DataProcessor(
            num_impute_strategy='mean',
            normalize='robust'
        )
        
        # Only use feature columns for training, not target columns (time, event, time_bin)
        feature_cols = [f'feature_{i}' for i in range(n_features)]
        self.processor.fit(self.df[feature_cols])
        
        # Process features
        self.df_processed = self.processor.transform(self.df[feature_cols])
        
        # Convert to tensors
        self.X_tensor = torch.tensor(self.df_processed.values, dtype=torch.float32)
        self.target_tensor = torch.tensor(self.target, dtype=torch.float32)
        self.cr_target_tensor = torch.tensor(self.cr_target, dtype=torch.float32)
        
        # Create datasets and dataloaders with proper formatting
        # The model expects a dictionary with 'continuous' and 'targets' keys
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
                
        class CompetingRisksDataset(torch.utils.data.Dataset):
            def __init__(self, X, targets):
                self.X = X
                self.targets = targets
                
            def __len__(self):
                return len(self.X)
                
            def __getitem__(self, idx):
                return {
                    'continuous': self.X[idx],
                    'targets': {
                        'competing_risks': self.targets[idx]
                    }
                }
        
        # Create the custom datasets
        survival_dataset = SurvivalDataset(self.X_tensor, self.target_tensor)
        self.loader = DataLoader(survival_dataset, batch_size=32)
        
        cr_dataset = CompetingRisksDataset(self.X_tensor, self.cr_target_tensor)
        self.cr_loader = DataLoader(cr_dataset, batch_size=32)
        
        # Create single risk model
        self.task_head = SingleRiskHead(
            name='survival',
            input_dim=64,
            num_time_bins=self.num_bins,
            alpha_rank=0.1
        )
        
        self.model = EnhancedDeepHit(
            num_continuous=n_features,
            targets=[self.task_head],
            encoder_dim=64,
            encoder_depth=2,
            encoder_heads=4,
            include_variational=True,
            device='cpu'
        )
        
        # Train model for a few epochs
        self.model.fit(
            train_loader=self.loader,
            num_epochs=2,
            learning_rate=0.001
        )
        
        # Add visualization tests for competing risks model
        # self.cr_task_head = CompetingRisksHead(
        #     name='competing_risks',
        #     input_dim=64,
        #     num_time_bins=self.num_bins,
        #     num_risks=2,
        #     alpha_rank=0.1
        # )
        # 
        # self.cr_model = EnhancedDeepHit(
        #     num_continuous=n_features,
        #     targets=[self.cr_task_head],
        #     encoder_dim=64,
        #     encoder_depth=2,
        #     encoder_heads=4,
        #     include_variational=True,
        #     device='cpu'
        # )
        # 
        # # Train model for a few epochs
        # self.cr_model.fit(
        #     train_loader=self.cr_loader,
        #     num_epochs=2,
        #     learning_rate=0.001
        # )
        
        # Generate predictions
        with torch.no_grad():
            self.preds = self.model.predict(self.X_tensor)
            # self.cr_preds = self.cr_model.predict(self.X_tensor)
            
            # Compute uncertainty
            self.uncertainty = self.model.compute_uncertainty(self.X_tensor, num_samples=5)
            # self.cr_uncertainty = self.cr_model.compute_uncertainty(self.X_tensor, num_samples=5)
    
    def test_survival_curve_plot(self):
        """Test survival curve plotting."""
        # Get survival curves from model predictions
        survival_curves = self.preds['task_outputs']['survival']['survival'].numpy()
        
        # Plot for a single patient
        fig1 = plot_survival_curve(
            survival_curves[0], 
            time_points=np.arange(self.num_bins)
        )
        self.assertIsNotNone(fig1)
        plt.close(fig1)
        
        # Plot for multiple patients
        fig2 = plot_survival_curve(
            survival_curves[:5], 
            time_points=np.arange(self.num_bins),
            labels=['Patient 1', 'Patient 2', 'Patient 3', 'Patient 4', 'Patient 5']
        )
        self.assertIsNotNone(fig2)
        plt.close(fig2)
        
        # Plot with uncertainty
        uncertainty_std = self.uncertainty['survival']['std'].numpy()[0]
        fig3 = plot_survival_curve(
            survival_curves[0],
            time_points=np.arange(self.num_bins),
            uncertainty=uncertainty_std
        )
        self.assertIsNotNone(fig3)
        plt.close(fig3)
    
    # def test_cumulative_incidence_plot(self):
    #     """Test cumulative incidence function plotting."""
    #     # Get CIF from competing risks model predictions
    #     cif = self.cr_preds['task_outputs']['competing_risks']['cif'].numpy()
    #     
    #     # Plot for a single patient
    #     fig1 = plot_cumulative_incidence(
    #         cif[0], 
    #         time_points=np.arange(self.num_bins),
    #         risk_names=['Cause 1', 'Cause 2']
    #     )
    #     self.assertIsNotNone(fig1)
    #     plt.close(fig1)
    #     
    #     # Plot with uncertainty
    #     uncertainty_std = self.cr_uncertainty['competing_risks']['std'].numpy()[0]
    #     fig2 = plot_cumulative_incidence(
    #         cif[0],
    #         time_points=np.arange(self.num_bins),
    #         risk_names=['Cause 1', 'Cause 2'],
    #         uncertainty=uncertainty_std
    #     )
    #     self.assertIsNotNone(fig2)
    #     plt.close(fig2)
    
    def test_calibration_curve_plot(self):
        """Test calibration curve plotting."""
        # Get survival probabilities and true event indicators
        survival_probs = self.preds['task_outputs']['survival']['survival'].numpy()
        event_indicators = self.target[:, 0]
        event_times = self.target[:, 1].astype(int)
        
        # Plot calibration curve
        fig = plot_calibration_curve(
            survival_probs,
            event_indicators,
            event_times,
            time_bin=5  # Calibration at time bin 5
        )
        self.assertIsNotNone(fig)
        plt.close(fig)
    
    # def test_prediction_intervals_plot(self):
    #     """Test prediction intervals plotting."""
    #     # Get mean and standard deviation for predictions
    #     mean_survival = self.uncertainty['survival']['mean'].numpy()
    #     std_survival = self.uncertainty['survival']['std'].numpy()
    #     
    #     # Plot prediction intervals
    #     fig = plot_prediction_intervals(
    #         mean_survival[0],
    #         std_survival[0],
    #         time_points=np.arange(self.num_bins),
    #         confidence_level=0.95
    #     )
    #     self.assertIsNotNone(fig)
    #     plt.close(fig)
    # 
    # def test_uncertainty_heatmap(self):
    #     """Test uncertainty heatmap plotting."""
    #     # Get uncertainty metrics for a group of patients
    #     std_values = self.uncertainty['survival']['std'].numpy()[:20]  # First 20 patients
    #     
    #     # Plot uncertainty heatmap
    #     fig = plot_uncertainty_heatmap(
    #         std_values,
    #         time_points=np.arange(self.num_bins),
    #         patient_ids=np.arange(20)
    #     )
    #     self.assertIsNotNone(fig)
    #     plt.close(fig)
    
    def test_partial_dependence_plot(self):
        """Test partial dependence plot."""
        # Plot partial dependence for feature 0
        fig = plot_partial_dependence(
            self.model,
            self.X_tensor,
            feature_idx=0,
            feature_name='Feature 0',
            n_points=10,
            target='risk_score'
        )
        self.assertIsNotNone(fig)
        plt.close(fig)
    
    def test_ice_curves_plot(self):
        """Test individual conditional expectation curves."""
        # Plot ICE curves for feature 0
        fig = plot_ice_curves(
            self.model,
            self.X_tensor[:10],  # First 10 samples
            feature_idx=0,
            feature_name='Feature 0',
            n_points=10,
            target='risk_score'
        )
        self.assertIsNotNone(fig)
        plt.close(fig)
    
    def test_feature_interaction_plot(self):
        """Test feature interaction plot."""
        # Plot interaction between features 0 and 2
        fig = plot_feature_interaction(
            self.model,
            self.X_tensor,
            feature1_idx=0,
            feature2_idx=2,
            feature1_name='Feature 0',
            feature2_name='Feature 2',
            n_points=5,
            target='risk_score'
        )
        self.assertIsNotNone(fig)
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()