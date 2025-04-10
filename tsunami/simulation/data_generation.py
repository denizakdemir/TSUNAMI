"""
Functions for generating synthetic survival data.

This module provides functions for generating synthetic data for various
survival analysis scenarios, including:
- Single-risk survival data
- Competing risks data
- Missing data patterns
- Time-dependent covariates
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
import math


def generate_survival_data(
    n_samples: int = 1000,
    n_features: int = 20,
    censoring_rate: float = 0.3,
    feature_weights: Optional[List[float]] = None,
    nonlinear_effects: bool = False,
    include_interactions: bool = False,
    include_categorical: bool = False,
    num_categories: Optional[List[int]] = None,
    missing_rate: float = 0.0,
    seed: Optional[int] = None
) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """
    Generate synthetic survival data.
    
    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples to generate
        
    n_features : int, default=20
        Number of features to generate
        
    censoring_rate : float, default=0.3
        Proportion of censored instances
        
    feature_weights : List[float], optional
        List of feature importance weights, if None, random weights are used
        
    nonlinear_effects : bool, default=False
        Whether to include nonlinear effects
        
    include_interactions : bool, default=False
        Whether to include interaction effects
        
    include_categorical : bool, default=False
        Whether to include categorical features
        
    num_categories : List[int], optional
        Number of categories for each categorical feature
        
    missing_rate : float, default=0.0
        Rate of missing values to introduce
        
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray, int]
        - pd.DataFrame with features and outcome variables
        - Target array in the format expected by the model
        - Number of time bins
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Determine number of continuous and categorical features
    n_categorical = 0
    if include_categorical:
        n_categorical = min(n_features // 4, 5)  # 25% categorical, up to 5
    
    n_continuous = n_features - n_categorical
    
    # Generate continuous features
    X_continuous = np.random.randn(n_samples, n_continuous)
    
    # Generate categorical features
    X_categorical = None
    cat_cols = []
    
    if n_categorical > 0:
        X_categorical = np.zeros((n_samples, n_categorical))
        
        # Use provided category counts or generate random ones
        if num_categories is None:
            num_categories = [np.random.randint(2, 6) for _ in range(n_categorical)]
        
        for i in range(n_categorical):
            X_categorical[:, i] = np.random.randint(
                0, num_categories[i], size=n_samples
            )
            
            cat_cols.append(f"cat_{i}")
    
    # Generate feature weights if not provided
    if feature_weights is None:
        feature_weights = np.random.uniform(-1, 1, n_continuous)
        
        # Add weights for categorical features
        if n_categorical > 0:
            cat_weights = []
            for n_cats in num_categories:
                # Generate a weight per category
                cat_weights.extend(np.random.uniform(-1, 1, n_cats))
    else:
        # Ensure correct length of weights
        if len(feature_weights) != n_continuous:
            raise ValueError(
                f"Expected {n_continuous} feature weights, got {len(feature_weights)}"
            )
    
    # Compute risk scores
    risk = np.zeros(n_samples)
    
    # Linear effects from continuous features
    risk += np.dot(X_continuous, feature_weights[:n_continuous])
    
    # Nonlinear effects
    if nonlinear_effects:
        # Add quadratic effects for a subset of features
        num_nonlinear = min(3, n_continuous)
        nonlinear_idx = np.random.choice(n_continuous, num_nonlinear, replace=False)
        
        for idx in nonlinear_idx:
            # Add quadratic effect
            risk += 0.5 * (X_continuous[:, idx] ** 2)
    
    # Interaction effects
    if include_interactions and n_continuous >= 2:
        # Add interactions between pairs of features
        num_interactions = min(3, n_continuous // 2)
        
        for _ in range(num_interactions):
            # Select two features randomly
            idx1, idx2 = np.random.choice(n_continuous, 2, replace=False)
            
            # Add interaction effect
            risk += 0.5 * X_continuous[:, idx1] * X_continuous[:, idx2]
    
    # Effects from categorical features
    if n_categorical > 0:
        for i in range(n_categorical):
            # Get the number of categories
            n_cats = num_categories[i]
            
            # Generate effect for each category
            for j in range(n_cats):
                mask = X_categorical[:, i] == j
                
                if np.sum(mask) > 0:
                    # Apply weight for this category
                    if feature_weights is not None and len(feature_weights) > n_continuous + j:
                        weight = feature_weights[n_continuous + j]
                    else:
                        weight = np.random.uniform(-1, 1)
                    
                    risk[mask] += weight
    
    # Generate survival times from exponential distribution
    # Higher risk -> lower survival time
    lambda_i = np.exp(risk)
    T = np.random.exponential(scale=1.0 / lambda_i)
    
    # Generate censoring times with better calibration to achieve desired censoring rate
    # We need to adjust the scale to get closer to the desired censoring rate
    median_time = np.median(T)
    C = np.random.exponential(
        scale=median_time * np.log(2) / np.log(1 / censoring_rate),
        size=n_samples
    )
    
    # Determine observed time and event indicator
    time = np.minimum(T, C)
    event = (T <= C).astype(np.float32)
    
    # Discretize time into bins
    num_bins = 30
    max_time = np.percentile(time, 99)  # Use 99th percentile to avoid extreme values
    bin_edges = np.linspace(0, max_time, num_bins + 1)
    time_bin = np.digitize(time, bin_edges) - 1
    time_bin = np.clip(time_bin, 0, num_bins - 1)
    
    # Create a dataframe
    continuous_cols = [f"feature_{i}" for i in range(n_continuous)]
    
    if n_categorical > 0:
        all_cols = continuous_cols + cat_cols
        
        # Combine continuous and categorical features
        X_all = np.column_stack([X_continuous, X_categorical])
    else:
        all_cols = continuous_cols
        X_all = X_continuous
    
    data = pd.DataFrame(X_all, columns=all_cols)
    
    # Convert categorical columns to categorical type
    for col in cat_cols:
        data[col] = data[col].astype(int).astype('category')
    
    # Add outcome variables
    data['time'] = time
    data['event'] = event
    data['time_bin'] = time_bin
    
    # Add missing values if requested
    if missing_rate > 0:
        data = add_missing_values(data, missing_rate, continuous_cols)
    
    # Create target for the model
    # Format: [event_indicator, time_bin, one_hot_encoding]
    target = np.zeros((n_samples, 2 + num_bins))
    target[:, 0] = event
    target[:, 1] = time_bin
    
    # One-hot encoding of time
    for i in range(n_samples):
        if event[i]:
            # For events, mark the event time
            target[i, 2 + time_bin[i]] = 1
        else:
            # For censored, mark all times after censoring as unknown (-1)
            target[i, 2 + time_bin[i]:] = -1
    
    return data, target, num_bins


def generate_competing_risks_data(
    n_samples: int = 1000,
    n_features: int = 20,
    censoring_rate: float = 0.3,
    num_risks: int = 2,
    feature_weights_per_risk: Optional[List[List[float]]] = None,
    risk_prevalence: Optional[List[float]] = None,
    nonlinear_effects: bool = False,
    include_interactions: bool = False,
    include_categorical: bool = False,
    num_categories: Optional[List[int]] = None,
    missing_rate: float = 0.0,
    seed: Optional[int] = None
) -> Tuple[pd.DataFrame, np.ndarray, int, int]:
    """
    Generate synthetic competing risks data.
    
    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples to generate
        
    n_features : int, default=20
        Number of features to generate
        
    censoring_rate : float, default=0.3
        Proportion of censored instances
        
    num_risks : int, default=2
        Number of competing risks
        
    feature_weights_per_risk : List[List[float]], optional
        List of feature importance weights for each risk,
        if None, random weights are used
        
    risk_prevalence : List[float], optional
        Relative prevalence of each risk, if None, equal prevalence is used
        
    nonlinear_effects : bool, default=False
        Whether to include nonlinear effects
        
    include_interactions : bool, default=False
        Whether to include interaction effects
        
    include_categorical : bool, default=False
        Whether to include categorical features
        
    num_categories : List[int], optional
        Number of categories for each categorical feature
        
    missing_rate : float, default=0.0
        Rate of missing values to introduce
        
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray, int, int]
        - pd.DataFrame with features and outcome variables
        - Target array in the format expected by the model
        - Number of time bins
        - Number of risks
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Determine number of continuous and categorical features
    n_categorical = 0
    if include_categorical:
        n_categorical = min(n_features // 4, 5)  # 25% categorical, up to 5
    
    n_continuous = n_features - n_categorical
    
    # Generate continuous features
    X_continuous = np.random.randn(n_samples, n_continuous)
    
    # Generate categorical features
    X_categorical = None
    cat_cols = []
    
    if n_categorical > 0:
        X_categorical = np.zeros((n_samples, n_categorical))
        
        # Use provided category counts or generate random ones
        if num_categories is None:
            num_categories = [np.random.randint(2, 6) for _ in range(n_categorical)]
        
        for i in range(n_categorical):
            X_categorical[:, i] = np.random.randint(
                0, num_categories[i], size=n_samples
            )
            
            cat_cols.append(f"cat_{i}")
    
    # Generate feature weights for each risk if not provided
    if feature_weights_per_risk is None:
        feature_weights_per_risk = []
        
        for _ in range(num_risks):
            # Generate weights for continuous features
            weights = np.random.uniform(-1, 1, n_continuous)
            
            # Add weights for categorical features (not implemented yet)
            feature_weights_per_risk.append(weights)
    else:
        # Ensure correct number of risk weight lists
        if len(feature_weights_per_risk) != num_risks:
            raise ValueError(
                f"Expected {num_risks} risk weight lists, got {len(feature_weights_per_risk)}"
            )
        
        # Ensure correct length of weights for each risk
        for i, weights in enumerate(feature_weights_per_risk):
            if len(weights) != n_continuous:
                raise ValueError(
                    f"Expected {n_continuous} feature weights for risk {i}, got {len(weights)}"
                )
    
    # Set risk prevalence if not provided
    if risk_prevalence is None:
        risk_prevalence = [1.0 / num_risks] * num_risks
    else:
        # Normalize to sum to 1
        total = sum(risk_prevalence)
        risk_prevalence = [p / total for p in risk_prevalence]
    
    # Compute risk scores for each competing risk
    risk_scores = np.zeros((n_samples, num_risks))
    
    for risk_idx in range(num_risks):
        # Linear effects from continuous features
        weights = feature_weights_per_risk[risk_idx]
        risk = np.dot(X_continuous, weights[:n_continuous])
        
        # Nonlinear effects
        if nonlinear_effects:
            # Add quadratic effects for a subset of features
            num_nonlinear = min(3, n_continuous)
            nonlinear_idx = np.random.choice(n_continuous, num_nonlinear, replace=False)
            
            for idx in nonlinear_idx:
                # Add quadratic effect
                risk += 0.5 * (X_continuous[:, idx] ** 2) * np.sign(weights[idx])
        
        # Interaction effects
        if include_interactions and n_continuous >= 2:
            # Add interactions between pairs of features
            num_interactions = min(3, n_continuous // 2)
            
            for _ in range(num_interactions):
                # Select two features randomly
                idx1, idx2 = np.random.choice(n_continuous, 2, replace=False)
                
                # Add interaction effect
                risk += 0.5 * X_continuous[:, idx1] * X_continuous[:, idx2] * np.sign(weights[idx1] * weights[idx2])
        
        # Effects from categorical features (to be implemented)
        
        # Store risk scores
        risk_scores[:, risk_idx] = risk
    
    # Generate survival times for each risk from exponential distribution
    T = np.zeros((n_samples, num_risks))
    
    for risk_idx in range(num_risks):
        # Higher risk -> lower survival time
        lambda_i = np.exp(risk_scores[:, risk_idx]) * risk_prevalence[risk_idx]
        T[:, risk_idx] = np.random.exponential(scale=1.0 / lambda_i)
    
    # Generate censoring times
    # Use the minimum event time to calibrate censoring
    min_event_time = np.min(T, axis=1)
    C = np.random.exponential(
        scale=np.percentile(min_event_time, 70) / np.log(1 / (1 - censoring_rate)),
        size=n_samples
    )
    
    # Determine observed time, event indicator, and cause
    time = np.min(np.column_stack([T, C.reshape(-1, 1)]), axis=1)
    event = (np.min(T, axis=1) <= C).astype(np.float32)
    cause = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        if event[i]:
            # Determine which risk occurred first
            cause[i] = np.argmin(T[i, :])
    
    # Discretize time into bins
    num_bins = 30
    max_time = np.percentile(time, 99)  # Use 99th percentile to avoid extreme values
    bin_edges = np.linspace(0, max_time, num_bins + 1)
    time_bin = np.digitize(time, bin_edges) - 1
    time_bin = np.clip(time_bin, 0, num_bins - 1)
    
    # Create a dataframe
    continuous_cols = [f"feature_{i}" for i in range(n_continuous)]
    
    if n_categorical > 0:
        all_cols = continuous_cols + cat_cols
        
        # Combine continuous and categorical features
        X_all = np.column_stack([X_continuous, X_categorical])
    else:
        all_cols = continuous_cols
        X_all = X_continuous
    
    data = pd.DataFrame(X_all, columns=all_cols)
    
    # Convert categorical columns to categorical type
    for col in cat_cols:
        data[col] = data[col].astype(int).astype('category')
    
    # Add outcome variables
    data['time'] = time
    data['event'] = event
    data['cause'] = np.where(event, cause, -1)  # -1 for censored
    data['time_bin'] = time_bin
    
    # Add missing values if requested
    if missing_rate > 0:
        data = add_missing_values(data, missing_rate, continuous_cols)
    
    # Create target for the model
    # Format: [event_indicator, time_bin, cause_index, one_hot_encoding]
    target = np.zeros((n_samples, 3 + num_risks * num_bins))
    target[:, 0] = event
    target[:, 1] = time_bin
    target[:, 2] = np.where(event, cause, -1)  # -1 for censored
    
    # One-hot encoding of time and cause
    for i in range(n_samples):
        if event[i]:
            # For events, mark the event time and cause
            cause_offset = cause[i] * num_bins
            target[i, 3 + cause_offset + time_bin[i]] = 1
        else:
            # For censored, mark all times after censoring as unknown (-1)
            for c in range(num_risks):
                cause_offset = c * num_bins
                target[i, 3 + cause_offset + time_bin[i]:3 + (c+1) * num_bins] = -1
    
    return data, target, num_bins, num_risks


def add_missing_values(
    df: pd.DataFrame,
    missing_rate: float = 0.1,
    continuous_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Add missing values to continuous features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    missing_rate : float
        Rate of missing values to introduce
        
    continuous_cols : List[str], optional
        List of continuous feature columns. If None, uses all numeric columns
        
    Returns
    -------
    pd.DataFrame
        Dataframe with missing values
    """
    # Create a copy of the dataframe
    df_missing = df.copy()
    
    # If continuous_cols not provided, use all numeric columns
    if continuous_cols is None:
        continuous_cols = df.select_dtypes(include=np.number).columns.tolist()
        # Exclude outcome variables
        exclude_cols = ['time', 'event', 'time_bin', 'cause']
        continuous_cols = [col for col in continuous_cols if col not in exclude_cols]
    
    # Add missing values to continuous features
    for col in continuous_cols:
        # Generate random mask
        mask = np.random.random(len(df)) < missing_rate
        
        # Apply mask
        df_missing.loc[mask, col] = np.nan
    
    return df_missing


def generate_missing_data(
    df: pd.DataFrame,
    missing_rate: float = 0.1,
    mcar_features: Optional[List[str]] = None,
    mar_features: Optional[List[str]] = None,
    mnar_features: Optional[List[str]] = None,
    mar_condition: Optional[str] = None,
    mnar_threshold: Optional[float] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate missing data patterns in a dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    missing_rate : float, default=0.1
        Overall rate of missing values
        
    mcar_features : List[str], optional
        Features to make Missing Completely At Random
        
    mar_features : List[str], optional
        Features to make Missing At Random (depends on other features)
        
    mnar_features : List[str], optional
        Features to make Missing Not At Random (depends on own value)
        
    mar_condition : str, optional
        Condition for MAR pattern (e.g., "feature_1 > 0")
        
    mnar_threshold : float, optional
        Threshold for MNAR pattern (values above this are more likely to be missing)
        
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        Dataframe with missing values
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Create a copy of the dataframe
    df_missing = df.copy()
    
    # Get numerical features
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Default to MCAR for all numeric features if not specified
    if mcar_features is None and mar_features is None and mnar_features is None:
        mcar_features = numeric_cols
    
    # Initialize lists if not provided
    mcar_features = mcar_features or []
    mar_features = mar_features or []
    mnar_features = mnar_features or []
    
    # MCAR pattern: Missing Completely At Random
    for col in mcar_features:
        if col in df.columns:
            # Generate random mask
            mask = np.random.random(len(df)) < missing_rate
            
            # Apply mask
            df_missing.loc[mask, col] = np.nan
    
    # MAR pattern: Missing At Random (depends on other features)
    if mar_features and mar_condition:
        # Evaluate the condition
        try:
            condition_mask = df.eval(mar_condition)
            
            # Increase missing probability for rows that satisfy the condition
            for col in mar_features:
                if col in df.columns:
                    mask = np.random.random(len(df)) < (missing_rate * 2 * condition_mask + missing_rate * 0.5 * (~condition_mask))
                    
                    # Apply mask
                    df_missing.loc[mask, col] = np.nan
        except Exception as e:
            print(f"Error evaluating MAR condition: {e}")
    
    # MNAR pattern: Missing Not At Random (depends on own value)
    if mnar_features:
        # Default threshold is the median if not provided
        if mnar_threshold is None:
            mnar_threshold = 0.0
        
        for col in mnar_features:
            if col in df.columns and col in numeric_cols:
                # Calculate probability of missing based on the feature's own value
                # Values above threshold are more likely to be missing
                is_above = df[col] > mnar_threshold
                
                # Generate mask with increased probability for values above threshold
                mask = np.random.random(len(df)) < (missing_rate * 2 * is_above + missing_rate * 0.5 * (~is_above))
                
                # Apply mask
                df_missing.loc[mask, col] = np.nan
    
    return df_missing


def generate_longitudinal_data(
    n_subjects: int = 100,
    n_timepoints: int = 5,
    n_features: int = 10,
    n_static_features: int = 5,
    missing_rate: float = 0.1,
    event_rate: float = 0.7,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic longitudinal data with time-varying covariates.
    
    Parameters
    ----------
    n_subjects : int, default=100
        Number of subjects
        
    n_timepoints : int, default=5
        Maximum number of time points per subject
        
    n_features : int, default=10
        Number of time-varying features
        
    n_static_features : int, default=5
        Number of static features
        
    missing_rate : float, default=0.1
        Proportion of missing values in time-varying features
        
    event_rate : float, default=0.7
        Proportion of subjects who experience an event
        
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        Longitudinal data in long format
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize data list
    data_list = []
    
    # Generate static features (same for all time points for a subject)
    static_features = np.random.randn(n_subjects, n_static_features)
    
    # Generate event indicators
    events = np.random.random(n_subjects) < event_rate
    
    # Generate time of event/censoring for each subject
    follow_up_times = np.zeros(n_subjects)
    
    for i in range(n_subjects):
        if events[i]:
            # For events, generate a time between 1 and n_timepoints
            follow_up_times[i] = np.random.randint(1, n_timepoints + 1)
        else:
            # For censored, generate a time between 1 and n_timepoints
            follow_up_times[i] = np.random.randint(1, n_timepoints + 1)
    
    # Generate data for each subject
    for subject_id in range(n_subjects):
        # Determine number of observations for this subject
        n_obs = int(follow_up_times[subject_id])
        
        # Generate time-varying features with temporal trends
        time_varying_features = np.zeros((n_obs, n_features))
        
        # Initial values
        time_varying_features[0] = np.random.randn(n_features)
        
        # Add temporal trends
        for t in range(1, n_obs):
            # Add autoregressive component and random noise
            time_varying_features[t] = 0.8 * time_varying_features[t-1] + 0.2 * np.random.randn(n_features)
        
        # Add missing values
        missing_mask = np.random.random((n_obs, n_features)) < missing_rate
        time_varying_features[missing_mask] = np.nan
        
        # Create records for each time point
        for t in range(n_obs):
            # Determine if this is an event time point
            is_event = (events[subject_id] and t == n_obs - 1)
            
            # Create record
            record = {
                'subject_id': subject_id,
                'time_point': t,
                'event': int(is_event),
                'censored': int(not events[subject_id] and t == n_obs - 1)
            }
            
            # Add static features
            for j in range(n_static_features):
                record[f'static_{j}'] = static_features[subject_id, j]
            
            # Add time-varying features
            for j in range(n_features):
                record[f'feature_{j}'] = time_varying_features[t, j]
            
            data_list.append(record)
    
    # Convert to dataframe
    df = pd.DataFrame(data_list)
    
    return df