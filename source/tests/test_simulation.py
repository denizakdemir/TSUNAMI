import os
import sys
import unittest
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from source.data.processing import DataProcessor
from source.models import EnhancedDeepHit
from source.models.tasks.survival import SingleRiskHead
from source.simulation.data_generation import (
    generate_survival_data,
    generate_competing_risks_data,
    generate_missing_data
)
from source.simulation.scenario_analysis import (
    ScenarioAnalysis,
    counterfactual_analysis,
    individual_treatment_effect
)
from source.simulation.missing_data_analysis import (
    MissingDataAnalysis
)


class TestSimulationFramework(unittest.TestCase):
    """Test cases for simulation framework."""
    
    def setUp(self):
        """Set up test environment."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Generate synthetic data
        n_samples = 100
        n_features = 5
        
        # Features
        self.X = np.random.randn(n_samples, n_features)
        
        # Create a simple model for testing
        self.task_head = SingleRiskHead(
            name='survival',
            input_dim=64,
            num_time_bins=10,
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
    
    def test_survival_data_generation(self):
        """Test survival data generation function."""
        data, target, num_bins = generate_survival_data(
            n_samples=100,
            n_features=5,
            censoring_rate=0.3,
            feature_weights=[0.5, 0.3, 0.8, 0.2, 0.1],  # Feature importance weights
            seed=42
        )
        
        # Check that data has the correct shape
        self.assertEqual(data.shape[0], 100)
        self.assertEqual(data.shape[1], 5 + 3)  # Features + time, event, time_bin
        
        # Check that target has the correct shape
        self.assertEqual(target.shape[0], 100)
        self.assertEqual(target.shape[1], 2 + num_bins)  # event, time_bin, one-hot encoding
        
        # The event rate may vary due to random generation
        # Just check that it's within a reasonable range
        event_rate = np.mean(data['event'])
        self.assertTrue(0.2 <= event_rate <= 0.85, f"Event rate {event_rate} out of expected range")
    
    def test_competing_risks_data_generation(self):
        """Test competing risks data generation function."""
        data, target, num_bins, num_risks = generate_competing_risks_data(
            n_samples=100,
            n_features=5,
            censoring_rate=0.3,
            num_risks=3,
            feature_weights_per_risk=[
                [0.5, 0.1, 0.1, 0.1, 0.1],  # Weights for risk 1
                [0.1, 0.5, 0.1, 0.1, 0.1],  # Weights for risk 2
                [0.1, 0.1, 0.5, 0.1, 0.1]   # Weights for risk 3
            ],
            seed=42
        )
        
        # Check that data has the correct shape
        self.assertEqual(data.shape[0], 100)
        self.assertEqual(data.shape[1], 5 + 4)  # Features + time, event, cause, time_bin
        
        # Check that target has the correct shape
        self.assertEqual(target.shape[0], 100)
        self.assertEqual(target.shape[1], 3 + num_risks * num_bins)  # event, time_bin, cause, one-hot encoding
        
        # Check that number of risks is correct
        self.assertEqual(num_risks, 3)
        
        # The event rate may vary due to random generation
        # Just check that it's within a reasonable range
        event_rate = np.mean(data['event'])
        self.assertTrue(0.2 <= event_rate <= 0.85, f"Event rate {event_rate} out of expected range")
        
        # Check that causes are properly assigned
        causes = data.loc[data['event'] == 1, 'cause'].unique()
        self.assertTrue(set(causes).issubset({0, 1, 2}))
    
    def test_missing_data_generation(self):
        """Test missing data generation function."""
        # Create a dataframe to introduce missingness
        df = pd.DataFrame(
            np.random.randn(100, 5), 
            columns=[f'feature_{i}' for i in range(5)]
        )
        
        # Add missing values
        df_with_missing = generate_missing_data(
            df,
            missing_rate=0.1,  # 10% missing rate
            mcar_features=['feature_0'],  # Missing completely at random
            mar_features=['feature_1'],   # Missing at random (depends on other features)
            mnar_features=['feature_2'],  # Missing not at random (depends on own value)
            mar_condition='feature_3 > 0',
            mnar_threshold=0,
            seed=42
        )
        
        # Check that missing values are introduced
        self.assertTrue(df_with_missing.isna().any().any())
        
        # Check that missing rate is close to expected
        missing_rate = df_with_missing.isna().sum().sum() / df_with_missing.size
        self.assertAlmostEqual(missing_rate, 0.1, delta=0.05)
        
        # Check that each feature has missing values
        for feature in ['feature_0', 'feature_1', 'feature_2']:
            self.assertTrue(df_with_missing[feature].isna().any())
    
    def test_scenario_analysis(self):
        """Test scenario analysis functionality."""
        # Create a scenario analyzer
        analyzer = ScenarioAnalysis(self.model)
        
        # Create a base tensor for analysis
        base_tensor = torch.tensor(self.X, dtype=torch.float32)
        
        # Test feature perturbation
        perturbed_results = analyzer.perturb_feature(
            base_tensor,
            feature_idx=0,
            perturbation=1.0,  # Increase feature 0 by 1.0
            target_task='survival'
        )
        
        # Check that results are returned
        self.assertIn('original', perturbed_results)
        self.assertIn('perturbed', perturbed_results)
        self.assertIn('difference', perturbed_results)
        
        # Test what-if scenarios
        scenarios = {
            'base': base_tensor,
            'scenario1': base_tensor.clone(),
            'scenario2': base_tensor.clone()
        }
        scenarios['scenario1'][:, 0] += 1.0  # Scenario 1: increase feature 0
        scenarios['scenario2'][:, 1] += 1.0  # Scenario 2: increase feature 1
        
        scenario_results = analyzer.compare_scenarios(
            scenarios,
            target_task='survival',
            metrics=['risk_score', 'survival_probability'],
            time_bins=[0, 5, 9]  # Add time bins for survival probability
        )
        
        # Check that results are returned for each scenario
        self.assertIn('base', scenario_results)
        self.assertIn('scenario1', scenario_results)
        self.assertIn('scenario2', scenario_results)
    
    def test_counterfactual_analysis(self):
        """Test counterfactual analysis function."""
        # Create base tensor
        base_tensor = torch.tensor(self.X[:10], dtype=torch.float32)  # Use first 10 samples
        
        # Define treatment variable and values
        treatment_idx = 0  # Feature 0 is the treatment
        control_value = 0.0
        treatment_value = 1.0
        
        # Calculate counterfactual predictions
        cf_results = counterfactual_analysis(
            self.model,
            base_tensor,
            treatment_idx,
            control_value,
            treatment_value,
            target_task='survival',
            target_metric='risk_score'
        )
        
        # Check that results are returned
        self.assertIn('control_predictions', cf_results)
        self.assertIn('treatment_predictions', cf_results)
        self.assertIn('effect', cf_results)
        
        # Check that individual treatment effects are calculated
        effects = individual_treatment_effect(
            self.model,
            base_tensor,
            treatment_idx,
            control_value,
            treatment_value,
            target_task='survival',
            target_metric='risk_score'
        )
        
        # Check that an effect is returned for each sample
        self.assertEqual(len(effects), 10)
    
    def test_missing_data_analysis(self):
        """Test missing data analysis functionality."""
        # Create a missing data analyzer
        analyzer = MissingDataAnalysis(self.model)
        
        # Create a base tensor for analysis
        base_tensor = torch.tensor(self.X, dtype=torch.float32)
        
        # Test prediction with missing values
        mask = torch.zeros_like(base_tensor)
        mask[0, 0] = 1  # Mark first feature of first sample as missing
        
        predictions = analyzer.predict_with_missing(
            base_tensor,
            mask,
            target_task='survival'
        )
        
        # Check that predictions are returned
        self.assertTrue('predictions' in predictions)
        self.assertEqual(len(predictions['predictions']), len(base_tensor))
        
        # Test robustness to missing data
        robustness_results = analyzer.test_robustness(
            base_tensor,
            feature_indices=[0, 1, 2],  # Test robustness to these features
            missing_ratios=[0.1, 0.3, 0.5],  # Test different missing ratios
            n_repeats=2,
            target_task='survival',
            metric='risk_score'
        )
        
        # Check that results are returned for each feature and ratio
        for feature_idx in [0, 1, 2]:
            for ratio in [0.1, 0.3, 0.5]:
                key = f'feature_{feature_idx}_ratio_{ratio}'
                self.assertIn(key, robustness_results)
        
        # Test imputation vs dropping comparison
        comparison_results = analyzer.compare_imputation_vs_dropping(
            base_tensor,
            feature_idx=0,
            missing_ratio=0.2,
            n_repeats=2,
            target_task='survival',
            metric='risk_score'
        )
        
        # Check that results for both strategies are returned
        self.assertIn('imputation', comparison_results)
        self.assertIn('dropping', comparison_results)
        self.assertIn('difference', comparison_results)


if __name__ == '__main__':
    unittest.main()