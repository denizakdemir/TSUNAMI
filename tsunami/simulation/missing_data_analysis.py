"""
Missing data analysis and robustness evaluation.

This module provides functions for analyzing the impact of missing data,
evaluating model robustness to missing values, and comparing imputation
strategies for survival models.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import copy
import tqdm


class MissingDataAnalysis:
    """
    Tools for analyzing the impact of missing data on model predictions.
    
    This class provides methods for evaluating model robustness to missing
    values and comparing different strategies for handling missing data.
    """
    
    def __init__(self, model):
        """
        Initialize missing data analysis tool.
        
        Parameters
        ----------
        model : EnhancedDeepHit
            The trained model to analyze
        """
        self.model = model
    
    def predict_with_missing(
        self,
        X: torch.Tensor,
        missing_mask: torch.Tensor,
        target_task: str,
        metric: str = 'risk_score',
        time_bin: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate predictions with specified missing values.
        
        Parameters
        ----------
        X : torch.Tensor
            Input features [n_samples, n_features]
            
        missing_mask : torch.Tensor
            Binary mask indicating missing values [n_samples, n_features]
            
        target_task : str
            Name of the task to analyze
            
        metric : str, default='risk_score'
            Metric to extract ('risk_score', 'survival', 'hazard')
            
        time_bin : int, optional
            Time bin for survival/hazard functions (required if metric is not 'risk_score')
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing predictions with missing values
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Make a copy of the input tensor
        X_missing = X.clone()
        
        # Apply missing mask (set to NaN or a special value)
        # In practice, the model should handle this correctly if it's designed to handle missing values
        
        # NOTE: Different models handle missing values differently.
        # Some models expect NaN values, others use a missing mask tensor.
        # Here we're providing both options but only one might be necessary.
        
        # Use NaN to represent missing values
        X_missing[missing_mask.bool()] = float('nan')
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model.predict(
                X_missing,
                missing_mask=missing_mask
            )
        
        # Extract relevant metric
        if metric == 'risk_score':
            result = predictions['task_outputs'][target_task]['risk_score'].cpu().numpy()
        elif metric == 'survival':
            if time_bin is None:
                result = predictions['task_outputs'][target_task]['survival'].cpu().numpy()
            else:
                result = predictions['task_outputs'][target_task]['survival'][:, time_bin].cpu().numpy()
        elif metric == 'hazard':
            if time_bin is None:
                result = predictions['task_outputs'][target_task]['hazard'].cpu().numpy()
            else:
                result = predictions['task_outputs'][target_task]['hazard'][:, time_bin].cpu().numpy()
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        return {'predictions': result}
    
    def test_robustness(
        self,
        X: torch.Tensor,
        feature_indices: Optional[List[int]] = None,
        missing_ratios: List[float] = [0.1, 0.3, 0.5],
        n_repeats: int = 10,
        target_task: str = 'survival',
        metric: str = 'risk_score',
        time_bin: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Test model robustness to missing values in different features.
        
        Parameters
        ----------
        X : torch.Tensor
            Input features [n_samples, n_features]
            
        feature_indices : List[int], optional
            Indices of features to test (default: all features)
            
        missing_ratios : List[float], default=[0.1, 0.3, 0.5]
            Ratios of missing values to test
            
        n_repeats : int, default=10
            Number of repetitions for each configuration
            
        target_task : str, default='survival'
            Name of the task to analyze
            
        metric : str, default='risk_score'
            Metric to extract ('risk_score', 'survival', 'hazard')
            
        time_bin : int, optional
            Time bin for survival/hazard functions (required if metric is not 'risk_score')
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing robustness results for each feature and ratio
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Get all feature indices if not provided
        if feature_indices is None:
            feature_indices = list(range(X.shape[1]))
        
        # Get baseline predictions (without missing values)
        with torch.no_grad():
            baseline_preds = self.model.predict(X)
        
        # Extract baseline metric
        if metric == 'risk_score':
            baseline_metric = baseline_preds['task_outputs'][target_task]['risk_score'].cpu().numpy()
        elif metric == 'survival':
            if time_bin is None:
                raise ValueError("time_bin is required for survival metric")
            baseline_metric = baseline_preds['task_outputs'][target_task]['survival'][:, time_bin].cpu().numpy()
        elif metric == 'hazard':
            if time_bin is None:
                raise ValueError("time_bin is required for hazard metric")
            baseline_metric = baseline_preds['task_outputs'][target_task]['hazard'][:, time_bin].cpu().numpy()
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Initialize results
        results = {}
        
        # Test each feature and missing ratio
        for feature_idx in feature_indices:
            for ratio in missing_ratios:
                # Initialize metrics for this configuration
                rmse_values = []
                
                # Repeat the experiment multiple times
                for _ in range(n_repeats):
                    # Create missing mask
                    missing_mask = torch.zeros_like(X, dtype=torch.bool)
                    
                    # Randomly set values to missing
                    rand_mask = torch.rand(X.shape[0]) < ratio
                    missing_mask[rand_mask, feature_idx] = True
                    
                    # Generate predictions with missing values
                    with torch.no_grad():
                        missing_preds = self.model.predict(
                            X,
                            missing_mask=missing_mask
                        )
                    
                    # Extract metric
                    if metric == 'risk_score':
                        missing_metric = missing_preds['task_outputs'][target_task]['risk_score'].cpu().numpy()
                    elif metric == 'survival':
                        missing_metric = missing_preds['task_outputs'][target_task]['survival'][:, time_bin].cpu().numpy()
                    elif metric == 'hazard':
                        missing_metric = missing_preds['task_outputs'][target_task]['hazard'][:, time_bin].cpu().numpy()
                    
                    # Compute RMSE
                    rmse = np.sqrt(np.mean((missing_metric - baseline_metric) ** 2))
                    rmse_values.append(rmse)
                
                # Compute average RMSE
                mean_rmse = np.mean(rmse_values)
                std_rmse = np.std(rmse_values)
                
                # Store results
                key = f"feature_{feature_idx}_ratio_{ratio}"
                results[key] = {
                    'mean_rmse': mean_rmse,
                    'std_rmse': std_rmse,
                    'all_rmse': rmse_values
                }
        
        return results
    
    def compare_imputation_vs_dropping(
        self,
        X: torch.Tensor,
        feature_idx: int,
        missing_ratio: float = 0.3,
        n_repeats: int = 10,
        target_task: str = 'survival',
        metric: str = 'risk_score',
        time_bin: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compare imputation vs. dropping strategies for missing data.
        
        Parameters
        ----------
        X : torch.Tensor
            Input features [n_samples, n_features]
            
        feature_idx : int
            Index of the feature to test
            
        missing_ratio : float, default=0.3
            Ratio of missing values to introduce
            
        n_repeats : int, default=10
            Number of repetitions
            
        target_task : str, default='survival'
            Name of the task to analyze
            
        metric : str, default='risk_score'
            Metric to extract ('risk_score', 'survival', 'hazard')
            
        time_bin : int, optional
            Time bin for survival/hazard functions (required if metric is not 'risk_score')
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing comparison results
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Get baseline predictions (without missing values)
        with torch.no_grad():
            baseline_preds = self.model.predict(X)
        
        # Extract baseline metric
        if metric == 'risk_score':
            baseline_metric = baseline_preds['task_outputs'][target_task]['risk_score'].cpu().numpy()
        elif metric == 'survival':
            if time_bin is None:
                raise ValueError("time_bin is required for survival metric")
            baseline_metric = baseline_preds['task_outputs'][target_task]['survival'][:, time_bin].cpu().numpy()
        elif metric == 'hazard':
            if time_bin is None:
                raise ValueError("time_bin is required for hazard metric")
            baseline_metric = baseline_preds['task_outputs'][target_task]['hazard'][:, time_bin].cpu().numpy()
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Initialize results
        imputation_rmse = []
        dropping_rmse = []
        
        # Repeat the experiment multiple times
        for _ in range(n_repeats):
            # Create missing mask
            missing_mask = torch.zeros_like(X, dtype=torch.bool)
            
            # Randomly set values to missing
            rand_mask = torch.rand(X.shape[0]) < missing_ratio
            missing_mask[rand_mask, feature_idx] = True
            
            # Strategy 1: Imputation (model handles missing values)
            with torch.no_grad():
                imputation_preds = self.model.predict(
                    X,
                    missing_mask=missing_mask
                )
            
            # Extract metric for imputation
            if metric == 'risk_score':
                imputation_metric = imputation_preds['task_outputs'][target_task]['risk_score'].cpu().numpy()
            elif metric == 'survival':
                imputation_metric = imputation_preds['task_outputs'][target_task]['survival'][:, time_bin].cpu().numpy()
            elif metric == 'hazard':
                imputation_metric = imputation_preds['task_outputs'][target_task]['hazard'][:, time_bin].cpu().numpy()
            
            # Strategy 2: Drop the feature (set to zero or mean)
            X_dropped = X.clone()
            
            # Set the feature to its mean value
            mean_value = torch.mean(X[:, feature_idx])
            X_dropped[rand_mask, feature_idx] = mean_value
            
            with torch.no_grad():
                dropping_preds = self.model.predict(X_dropped)
            
            # Extract metric for dropping
            if metric == 'risk_score':
                dropping_metric = dropping_preds['task_outputs'][target_task]['risk_score'].cpu().numpy()
            elif metric == 'survival':
                dropping_metric = dropping_preds['task_outputs'][target_task]['survival'][:, time_bin].cpu().numpy()
            elif metric == 'hazard':
                dropping_metric = dropping_preds['task_outputs'][target_task]['hazard'][:, time_bin].cpu().numpy()
            
            # Compute RMSE for both strategies
            imputation_rmse.append(np.sqrt(np.mean((imputation_metric - baseline_metric) ** 2)))
            dropping_rmse.append(np.sqrt(np.mean((dropping_metric - baseline_metric) ** 2)))
        
        # Compute statistics
        mean_imputation_rmse = np.mean(imputation_rmse)
        std_imputation_rmse = np.std(imputation_rmse)
        
        mean_dropping_rmse = np.mean(dropping_rmse)
        std_dropping_rmse = np.std(dropping_rmse)
        
        # Compute difference
        mean_diff = mean_imputation_rmse - mean_dropping_rmse
        
        return {
            'imputation': {
                'mean_rmse': mean_imputation_rmse,
                'std_rmse': std_imputation_rmse,
                'all_rmse': imputation_rmse
            },
            'dropping': {
                'mean_rmse': mean_dropping_rmse,
                'std_rmse': std_dropping_rmse,
                'all_rmse': dropping_rmse
            },
            'difference': mean_diff
        }
    
    def feature_importance_under_missingness(
        self,
        X: torch.Tensor,
        missing_rate: float = 0.2,
        target_task: str = 'survival',
        metric: str = 'risk_score',
        time_bin: Optional[int] = None,
        n_repeats: int = 5
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute feature importance under missing data conditions.
        
        Parameters
        ----------
        X : torch.Tensor
            Input features [n_samples, n_features]
            
        missing_rate : float, default=0.2
            Rate of missing values to introduce
            
        target_task : str, default='survival'
            Name of the task to analyze
            
        metric : str, default='risk_score'
            Metric to extract ('risk_score', 'survival', 'hazard')
            
        time_bin : int, optional
            Time bin for survival/hazard functions (required if metric is not 'risk_score')
            
        n_repeats : int, default=5
            Number of repetitions
            
        Returns
        -------
        Dict[int, Dict[str, float]]
            Dictionary mapping feature indices to importance metrics
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Get baseline predictions (without missing values)
        with torch.no_grad():
            baseline_preds = self.model.predict(X)
        
        # Extract baseline metric
        if metric == 'risk_score':
            baseline_metric = baseline_preds['task_outputs'][target_task]['risk_score'].cpu().numpy()
        elif metric == 'survival':
            if time_bin is None:
                raise ValueError("time_bin is required for survival metric")
            baseline_metric = baseline_preds['task_outputs'][target_task]['survival'][:, time_bin].cpu().numpy()
        elif metric == 'hazard':
            if time_bin is None:
                raise ValueError("time_bin is required for hazard metric")
            baseline_metric = baseline_preds['task_outputs'][target_task]['hazard'][:, time_bin].cpu().numpy()
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Initialize importance dictionary
        importance = {}
        
        # Analyze each feature
        for feature_idx in range(X.shape[1]):
            # Initialize metrics for this feature
            rmse_values = []
            
            # Repeat the experiment
            for _ in range(n_repeats):
                # Create missing mask with all features missing at random
                missing_mask = torch.rand_like(X) < missing_rate
                
                # Generate predictions with random missingness
                with torch.no_grad():
                    random_missing_preds = self.model.predict(
                        X,
                        missing_mask=missing_mask
                    )
                
                # Extract metric
                if metric == 'risk_score':
                    random_missing_metric = random_missing_preds['task_outputs'][target_task]['risk_score'].cpu().numpy()
                elif metric == 'survival':
                    random_missing_metric = random_missing_preds['task_outputs'][target_task]['survival'][:, time_bin].cpu().numpy()
                elif metric == 'hazard':
                    random_missing_metric = random_missing_preds['task_outputs'][target_task]['hazard'][:, time_bin].cpu().numpy()
                
                # Create new missing mask with current feature always present
                feature_present_mask = missing_mask.clone()
                feature_present_mask[:, feature_idx] = False
                
                # Generate predictions with feature always present
                with torch.no_grad():
                    feature_present_preds = self.model.predict(
                        X,
                        missing_mask=feature_present_mask
                    )
                
                # Extract metric
                if metric == 'risk_score':
                    feature_present_metric = feature_present_preds['task_outputs'][target_task]['risk_score'].cpu().numpy()
                elif metric == 'survival':
                    feature_present_metric = feature_present_preds['task_outputs'][target_task]['survival'][:, time_bin].cpu().numpy()
                elif metric == 'hazard':
                    feature_present_metric = feature_present_preds['task_outputs'][target_task]['hazard'][:, time_bin].cpu().numpy()
                
                # Compute improvement in RMSE
                random_rmse = np.sqrt(np.mean((random_missing_metric - baseline_metric) ** 2))
                feature_present_rmse = np.sqrt(np.mean((feature_present_metric - baseline_metric) ** 2))
                
                # Improvement is the difference in RMSE
                improvement = random_rmse - feature_present_rmse
                rmse_values.append(improvement)
            
            # Compute average improvement
            mean_improvement = np.mean(rmse_values)
            std_improvement = np.std(rmse_values)
            
            # Store importance
            importance[feature_idx] = {
                'mean_improvement': mean_improvement,
                'std_improvement': std_improvement,
                'all_improvements': rmse_values
            }
        
        # Normalize importance scores
        max_importance = max([info['mean_improvement'] for info in importance.values()])
        if max_importance > 0:
            for feature_idx in importance:
                importance[feature_idx]['normalized_importance'] = importance[feature_idx]['mean_improvement'] / max_importance
        
        return importance