"""
Scenario analysis and counterfactual prediction functions.

This module provides functions for performing scenario analysis
and counterfactual predictions with survival models, including:
- Feature perturbation analysis
- What-if scenario comparison
- Counterfactual analysis
- Individual treatment effect estimation
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import copy


class ScenarioAnalysis:
    """
    Tools for performing scenario analysis with survival models.
    
    This class provides methods for exploring how changes in feature
    values affect predictions, enabling what-if analysis and feature
    perturbation studies.
    """
    
    def __init__(self, model):
        """
        Initialize scenario analysis tool.
        
        Parameters
        ----------
        model : EnhancedDeepHit
            The trained model to use for scenario analysis
        """
        self.model = model
    
    def perturb_feature(
        self,
        X: torch.Tensor,
        feature_idx: int,
        perturbation: float,
        target_task: str,
        metric: str = 'risk_score',
        time_bin: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze the effect of perturbing a feature.
        
        Parameters
        ----------
        X : torch.Tensor
            Input features [n_samples, n_features]
            
        feature_idx : int
            Index of the feature to perturb
            
        perturbation : float
            Amount to add to the feature
            
        target_task : str
            Name of the task to analyze
            
        metric : str, default='risk_score'
            Metric to analyze ('risk_score', 'survival', 'hazard')
            
        time_bin : int, optional
            Time bin for survival/hazard functions (required if metric is not 'risk_score')
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'original': Original predictions
            - 'perturbed': Predictions after perturbation
            - 'difference': Difference between original and perturbed predictions
            - 'percent_change': Percent change in predictions
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Make a copy of the input tensor
        X_perturbed = X.clone()
        
        # Apply perturbation
        X_perturbed[:, feature_idx] += perturbation
        
        # Get original predictions
        with torch.no_grad():
            original_preds = self.model.predict(X)
            perturbed_preds = self.model.predict(X_perturbed)
        
        # Extract metrics
        if metric == 'risk_score':
            original_metric = original_preds['task_outputs'][target_task]['risk_score'].cpu().numpy()
            perturbed_metric = perturbed_preds['task_outputs'][target_task]['risk_score'].cpu().numpy()
        elif metric == 'survival':
            if time_bin is None:
                raise ValueError("time_bin is required for survival metric")
            
            original_metric = original_preds['task_outputs'][target_task]['survival'][:, time_bin].cpu().numpy()
            perturbed_metric = perturbed_preds['task_outputs'][target_task]['survival'][:, time_bin].cpu().numpy()
        elif metric == 'hazard':
            if time_bin is None:
                raise ValueError("time_bin is required for hazard metric")
            
            original_metric = original_preds['task_outputs'][target_task]['hazard'][:, time_bin].cpu().numpy()
            perturbed_metric = perturbed_preds['task_outputs'][target_task]['hazard'][:, time_bin].cpu().numpy()
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Compute difference and percent change
        difference = perturbed_metric - original_metric
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            percent_change = np.where(
                np.abs(original_metric) > 1e-10,
                100 * difference / original_metric,
                np.nan
            )
        
        return {
            'original': original_metric,
            'perturbed': perturbed_metric,
            'difference': difference,
            'percent_change': percent_change
        }
    
    def compare_scenarios(
        self,
        scenarios: Dict[str, torch.Tensor],
        target_task: str,
        metrics: List[str] = ['risk_score'],
        time_bins: Optional[List[int]] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compare predictions across different scenarios.
        
        Parameters
        ----------
        scenarios : Dict[str, torch.Tensor]
            Dictionary mapping scenario names to input tensors
            
        target_task : str
            Name of the task to analyze
            
        metrics : List[str], default=['risk_score']
            List of metrics to analyze ('risk_score', 'survival_probability', 'hazard_rate')
            
        time_bins : List[int], optional
            Time bins for survival/hazard metrics (required if those metrics are included)
            
        Returns
        -------
        Dict[str, Dict[str, np.ndarray]]
            Dictionary mapping scenario names to dictionaries of metrics
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Check if time_bins is provided when needed
        if any(m in metrics for m in ['survival_probability', 'hazard_rate']) and time_bins is None:
            raise ValueError("time_bins is required for survival_probability and hazard_rate metrics")
        
        # Initialize results
        results = {}
        
        # Analyze each scenario
        for scenario_name, X in scenarios.items():
            # Get predictions
            with torch.no_grad():
                predictions = self.model.predict(X)
            
            # Extract metrics
            scenario_results = {}
            
            for metric in metrics:
                if metric == 'risk_score':
                    scenario_results[metric] = predictions['task_outputs'][target_task]['risk_score'].cpu().numpy()
                elif metric == 'survival_probability':
                    survival = predictions['task_outputs'][target_task]['survival'].cpu().numpy()
                    
                    # Extract probabilities at specified time bins
                    for time_bin in time_bins:
                        if time_bin < survival.shape[1]:
                            scenario_results[f'survival_t{time_bin}'] = survival[:, time_bin]
                elif metric == 'hazard_rate':
                    hazard = predictions['task_outputs'][target_task]['hazard'].cpu().numpy()
                    
                    # Extract hazard rates at specified time bins
                    for time_bin in time_bins:
                        if time_bin < hazard.shape[1]:
                            scenario_results[f'hazard_t{time_bin}'] = hazard[:, time_bin]
            
            # Store results for this scenario
            results[scenario_name] = scenario_results
        
        return results
    
    def perturbation_importance(
        self,
        X: torch.Tensor,
        target_task: str,
        metric: str = 'risk_score',
        time_bin: Optional[int] = None,
        perturbation_size: float = 1.0,
        standardize: bool = True
    ) -> Dict[int, float]:
        """
        Calculate feature importance based on perturbation analysis.
        
        Parameters
        ----------
        X : torch.Tensor
            Input features [n_samples, n_features]
            
        target_task : str
            Name of the task to analyze
            
        metric : str, default='risk_score'
            Metric to analyze ('risk_score', 'survival', 'hazard')
            
        time_bin : int, optional
            Time bin for survival/hazard functions (required if metric is not 'risk_score')
            
        perturbation_size : float, default=1.0
            Size of perturbation in standard deviations
            
        standardize : bool, default=True
            Whether to standardize perturbations by feature standard deviations
            
        Returns
        -------
        Dict[int, float]
            Dictionary mapping feature indices to importance scores
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Compute feature standard deviations if standardizing
        if standardize:
            feature_stds = torch.std(X, dim=0)
        else:
            feature_stds = torch.ones(X.shape[1])
        
        # Initialize importance scores
        importance = {}
        
        # Analyze each feature
        for i in range(X.shape[1]):
            # Compute perturbation
            perturb = perturbation_size * feature_stds[i]
            
            # Compute effect of perturbation
            result = self.perturb_feature(
                X=X,
                feature_idx=i,
                perturbation=perturb,
                target_task=target_task,
                metric=metric,
                time_bin=time_bin
            )
            
            # Use mean absolute difference as importance
            importance[i] = np.mean(np.abs(result['difference']))
        
        # Normalize importance scores
        max_importance = max(importance.values())
        if max_importance > 0:
            for i in importance:
                importance[i] /= max_importance
        
        return importance


def counterfactual_analysis(
    model: Any,
    X: torch.Tensor,
    treatment_idx: int,
    control_value: float,
    treatment_value: float,
    target_task: str,
    target_metric: str = 'risk_score',
    time_bin: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Perform counterfactual analysis comparing control and treatment.
    
    Parameters
    ----------
    model : Any
        The trained model to use for predictions
        
    X : torch.Tensor
        Input features [n_samples, n_features]
        
    treatment_idx : int
        Index of the treatment feature
        
    control_value : float
        Value to use for control scenario
        
    treatment_value : float
        Value to use for treatment scenario
        
    target_task : str
        Name of the task to analyze
        
    target_metric : str, default='risk_score'
        Metric to analyze ('risk_score', 'survival', 'hazard')
        
    time_bin : int, optional
        Time bin for survival/hazard metrics (required if not using risk_score)
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'control_predictions': Predictions under control
        - 'treatment_predictions': Predictions under treatment
        - 'effect': Difference between treatment and control
    """
    # Set model to evaluation mode
    model.eval()
    
    # Create control and treatment scenarios
    X_control = X.clone()
    X_treatment = X.clone()
    
    # Set treatment values
    X_control[:, treatment_idx] = control_value
    X_treatment[:, treatment_idx] = treatment_value
    
    # Get predictions
    with torch.no_grad():
        control_preds = model.predict(X_control)
        treatment_preds = model.predict(X_treatment)
    
    # Extract relevant metric
    if target_metric == 'risk_score':
        control_metric = control_preds['task_outputs'][target_task]['risk_score'].cpu().numpy()
        treatment_metric = treatment_preds['task_outputs'][target_task]['risk_score'].cpu().numpy()
    elif target_metric == 'survival':
        if time_bin is None:
            raise ValueError("time_bin is required for survival metric")
        
        control_metric = control_preds['task_outputs'][target_task]['survival'][:, time_bin].cpu().numpy()
        treatment_metric = treatment_preds['task_outputs'][target_task]['survival'][:, time_bin].cpu().numpy()
    elif target_metric == 'hazard':
        if time_bin is None:
            raise ValueError("time_bin is required for hazard metric")
        
        control_metric = control_preds['task_outputs'][target_task]['hazard'][:, time_bin].cpu().numpy()
        treatment_metric = treatment_preds['task_outputs'][target_task]['hazard'][:, time_bin].cpu().numpy()
    else:
        raise ValueError(f"Unsupported metric: {target_metric}")
    
    # Compute treatment effect
    effect = treatment_metric - control_metric
    
    return {
        'control_predictions': control_metric,
        'treatment_predictions': treatment_metric,
        'effect': effect
    }


def individual_treatment_effect(
    model: Any,
    X: torch.Tensor,
    treatment_idx: int,
    control_value: float,
    treatment_value: float,
    target_task: str,
    target_metric: str = 'risk_score',
    time_bin: Optional[int] = None
) -> np.ndarray:
    """
    Compute individual treatment effect for each instance.
    
    Parameters
    ----------
    model : Any
        The trained model to use for predictions
        
    X : torch.Tensor
        Input features [n_samples, n_features]
        
    treatment_idx : int
        Index of the treatment feature
        
    control_value : float
        Value to use for control scenario
        
    treatment_value : float
        Value to use for treatment scenario
        
    target_task : str
        Name of the task to analyze
        
    target_metric : str, default='risk_score'
        Metric to analyze ('risk_score', 'survival', 'hazard')
        
    time_bin : int, optional
        Time bin for survival/hazard metrics (required if not using risk_score)
        
    Returns
    -------
    np.ndarray
        Individual treatment effects [n_samples]
    """
    # Perform counterfactual analysis
    results = counterfactual_analysis(
        model=model,
        X=X,
        treatment_idx=treatment_idx,
        control_value=control_value,
        treatment_value=treatment_value,
        target_task=target_task,
        target_metric=target_metric,
        time_bin=time_bin
    )
    
    # Return individual treatment effects
    return results['effect']