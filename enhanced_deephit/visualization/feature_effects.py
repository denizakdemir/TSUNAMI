"""
Visualization functions for feature effects and dependence.

This module provides functions for visualizing how features affect
predictions, including:
- Partial dependence plots
- Individual conditional expectation (ICE) curves
- Feature interaction plots
- Effect modifiers visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from matplotlib.figure import Figure
import torch


def plot_partial_dependence(
    model: Any,
    X: torch.Tensor,
    feature_idx: int,
    feature_name: Optional[str] = None,
    n_points: int = 20,
    target: str = 'risk_score',
    task_name: Optional[str] = None,
    time_bin: Optional[int] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    feature_range: Optional[Tuple[float, float]] = None,
    show_distribution: bool = True,
    ax: Optional[plt.Axes] = None,
    categorical_info: Optional[Dict] = None,
    sample_weights: Optional[torch.Tensor] = None
) -> Figure:
    """
    Plot partial dependence of prediction on a specified feature.
    
    Parameters
    ----------
    model : Any
        The trained model to use for predictions
        
    X : torch.Tensor
        Input features [n_samples, n_features]
        
    feature_idx : int
        Index of the feature to analyze
        
    feature_name : str, optional
        Name of the feature for plotting
        
    n_points : int, default=20
        Number of points to evaluate
        
    target : str, default='risk_score'
        Target prediction to analyze ('risk_score', 'survival', 'hazard')
        
    task_name : str, optional
        Name of the task for multi-task models
        
    time_bin : int, optional
        Time bin for survival/hazard functions (required if target is not 'risk_score')
        
    figsize : Tuple[float, float], default=(10, 6)
        Figure size if ax is not provided
        
    title : str, optional
        Plot title (default is auto-generated)
        
    xlabel : str, optional
        X-axis label (default is feature_name)
        
    ylabel : str, optional
        Y-axis label (default is based on target)
        
    feature_range : Tuple[float, float], optional
        Range of feature values to explore (default is data min/max)
        
    show_distribution : bool, default=True
        Whether to show the feature value distribution as a histogram
        
    ax : plt.Axes, optional
        Axes to plot on (creates new figure if not provided)
        
    categorical_info : Dict, optional
        Information about the feature if it's categorical, including
        original values mapping
        
    sample_weights : torch.Tensor, optional
        Sample weights for weighted feature effect calculations [n_samples]
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    # Create new figure if needed
    if ax is None:
        if show_distribution:
            # Create figure with two subplots (main plot and distribution)
            fig, (ax, dist_ax) = plt.subplots(
                2, 1, figsize=figsize, 
                gridspec_kw={'height_ratios': [3, 1]},
                sharex=True
            )
        else:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        dist_ax = None
    
    # Set feature name if not provided
    if feature_name is None:
        feature_name = f"Feature {feature_idx}"
    
    # Check if we're dealing with a categorical feature
    is_categorical = categorical_info is not None
    
    # Set axis labels if not provided
    if xlabel is None:
        if is_categorical and 'original_name' in categorical_info:
            xlabel = categorical_info['original_name']
        else:
            xlabel = feature_name
    
    if ylabel is None:
        if target == 'risk_score':
            ylabel = "Risk Score"
        elif target == 'survival':
            ylabel = f"Survival Probability (t={time_bin})" if time_bin is not None else "Survival Probability"
        elif target == 'hazard':
            ylabel = f"Hazard Rate (t={time_bin})" if time_bin is not None else "Hazard Rate"
        else:
            ylabel = "Prediction"
    
    # Set title if not provided
    if title is None:
        display_name = categorical_info.get('original_name', feature_name) if is_categorical else feature_name
        title = f"Partial Dependence of {target.replace('_', ' ').title()} on {display_name}"
    
    # Extract feature values
    feature_values = X[:, feature_idx].numpy()
    
    # For categorical features, use actual category values rather than numerical embedding values
    if is_categorical:
        # Determine unique feature indices based on the mapping
        reverse_mapping = categorical_info.get('reverse_mapping', {})
        
        if reverse_mapping:
            # Use the categories directly rather than numerical grid points
            category_indices = sorted(reverse_mapping.keys())
            
            # Compute partial dependence for each category
            pd_values = []
            category_labels = []
            
            for idx in category_indices:
                # Convert index to actual category value for label
                # Make sure we display the original categorical label, not its numeric representation
                original_label = reverse_mapping[idx]
                # If the original label is a number or any other type, ensure it's displayed as a string
                # This ensures we see the actual category names like "Male"/"Female" not index values
                category_labels.append(str(original_label))
                
                # Create a copy of the data - all embedding dimensions must be set appropriately
                # For simplicity, we'll use one-hot style embedding initialization
                X_modified = X.clone()
                
                # If this is part of a multi-dimensional embedding, we need to find all dimensions
                embed_dim = categorical_info.get('embed_dim', 1)
                embed_base = feature_idx - (feature_idx % embed_dim)
                
                # Reset all embedding dimensions to 0
                for i in range(embed_dim):
                    if embed_base + i < X.shape[1]:
                        X_modified[:, embed_base + i] = 0.0
                
                # Set the first dimension to a normalized value based on the category index
                # This mimics how categorical values are initially encoded
                X_modified[:, embed_base] = idx / categorical_info.get('cardinality', 1)
                
                # Get predictions
                with torch.no_grad():
                    if sample_weights is not None:
                        predictions = model.predict(X_modified, sample_weights=sample_weights)
                    else:
                        predictions = model.predict(X_modified)
                
                # Extract relevant prediction
                if task_name is None:
                    task_name = list(predictions['task_outputs'].keys())[0]
                
                if target == 'risk_score':
                    pred_value = predictions['task_outputs'][task_name]['risk_score'].mean().item()
                elif target == 'survival':
                    if time_bin is None:
                        time_bin = 0  # Default to first time bin
                    pred_value = predictions['task_outputs'][task_name]['survival'][:, time_bin].mean().item()
                elif target == 'hazard':
                    if time_bin is None:
                        time_bin = 0  # Default to first time bin
                    pred_value = predictions['task_outputs'][task_name]['hazard'][:, time_bin].mean().item()
                else:
                    raise ValueError(f"Unsupported target: {target}")
                
                pd_values.append(pred_value)
            
            # Convert to numpy array
            pd_values = np.array(pd_values)
            
            # Plot categorical partial dependence as bar chart
            x_pos = np.arange(len(category_labels))
            ax.bar(x_pos, pd_values, alpha=0.7)
            
            # Set x-axis labels to actual category names, not internal indices
            # Make sure we're showing the original category values, not numeric codes
            ax.set_xticks(x_pos)
            ax.set_xticklabels(category_labels, rotation=45, ha='right')
            
            # Add distribution as bar chart if requested
            if show_distribution and dist_ax is not None:
                # Count instances in each category
                counts = {}
                for val in feature_values:
                    # Determine which category this value corresponds to
                    # For simplicity, find closest category index
                    # Find the closest category index
                    closest_idx = min(category_indices, key=lambda x: abs(val - x/categorical_info.get('cardinality', 1)))
                    # Get the original category label from the reverse mapping
                    cat = reverse_mapping.get(closest_idx, 'Unknown')
                    # Convert to string to ensure consistent handling
                    cat = str(cat)
                    counts[cat] = counts.get(cat, 0) + 1
                
                # Plot distribution
                # Convert all mapped values to strings for consistency when looking up in counts dictionary
                dist_values = [counts.get(str(reverse_mapping.get(idx, 'Unknown')), 0) for idx in category_indices]
                total = sum(dist_values)
                dist_values = [v/total for v in dist_values] if total > 0 else dist_values
                
                dist_ax.bar(x_pos, dist_values, alpha=0.5)
                dist_ax.set_ylabel('Proportion')
                dist_ax.set_xticks(x_pos)
                dist_ax.set_xticklabels(category_labels, rotation=45, ha='right')
                dist_ax.grid(True, alpha=0.3)
        else:
            # Fallback to continuous treatment if no mapping available
            grid_values, pd_values = _compute_continuous_dependence(
                model, X, feature_idx, feature_range, n_points, target, task_name, time_bin, sample_weights
            )
            
            # Plot partial dependence
            ax.plot(grid_values, pd_values, 'b-', linewidth=2)
            
            # Add feature distribution if requested
            if show_distribution and dist_ax is not None:
                dist_ax.hist(feature_values, bins=30, alpha=0.5, density=True)
                dist_ax.set_ylabel('Density')
                dist_ax.grid(True, alpha=0.3)
    else:
        # Determine feature range for continuous features
        if feature_range is None:
            min_val = np.min(feature_values)
            max_val = np.max(feature_values)
            # Add some padding
            range_width = max_val - min_val
            min_val -= range_width * 0.1
            max_val += range_width * 0.1
            feature_range = (min_val, max_val)
        
        # For continuous features, compute partial dependence over a grid
        grid_values, pd_values = _compute_continuous_dependence(
            model, X, feature_idx, feature_range, n_points, target, task_name, time_bin, sample_weights
        )
        
        # Plot partial dependence
        ax.plot(grid_values, pd_values, 'b-', linewidth=2)
        
        # Add feature distribution if requested
        if show_distribution and dist_ax is not None:
            dist_ax.hist(feature_values, bins=30, alpha=0.5, density=True)
            dist_ax.set_ylabel('Density')
            dist_ax.grid(True, alpha=0.3)
    
    # Set axis labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add reference line at mean prediction
    with torch.no_grad():
        if sample_weights is not None:
            predictions = model.predict(X, sample_weights=sample_weights)
        else:
            predictions = model.predict(X)
    
    # Extract task name if not provided
    if task_name is None:
        task_name = list(predictions['task_outputs'].keys())[0]
    
    if target == 'risk_score':
        mean_pred = predictions['task_outputs'][task_name]['risk_score'].mean().item()
    elif target == 'survival':
        if time_bin is None:
            time_bin = 0  # Default to first time bin
        mean_pred = predictions['task_outputs'][task_name]['survival'][:, time_bin].mean().item()
    elif target == 'hazard':
        if time_bin is None:
            time_bin = 0  # Default to first time bin
        mean_pred = predictions['task_outputs'][task_name]['hazard'][:, time_bin].mean().item()
    
    # For categorical, add horizontal line; for continuous, use traditional reference line
    if is_categorical and len(category_labels) > 0:
        ax.axhline(
            mean_pred,
            color='r',
            linestyle='--',
            alpha=0.5,
            label=f"Mean {target.replace('_', ' ').title()}"
        )
    else:
        ax.axhline(
            mean_pred,
            color='r',
            linestyle='--',
            alpha=0.5,
            label=f"Mean {target.replace('_', ' ').title()}"
        )
    
    # Add legend
    ax.legend(loc='best')
    
    plt.tight_layout()
    return fig


def _compute_continuous_dependence(
    model: Any,
    X: torch.Tensor,
    feature_idx: int,
    feature_range: Tuple[float, float],
    n_points: int,
    target: str,
    task_name: Optional[str] = None,
    time_bin: Optional[int] = None,
    sample_weights: Optional[torch.Tensor] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to compute continuous partial dependence.
    
    Parameters
    ----------
    model : Any
        The trained model to use for predictions
        
    X : torch.Tensor
        Input features [n_samples, n_features]
        
    feature_idx : int
        Index of the feature to analyze
        
    feature_range : Tuple[float, float]
        Range of feature values to explore
        
    n_points : int
        Number of points to evaluate
        
    target : str
        Target prediction to analyze ('risk_score', 'survival', 'hazard')
        
    task_name : str, optional
        Name of the task for multi-task models
        
    time_bin : int, optional
        Time bin for survival/hazard functions
        
    sample_weights : torch.Tensor, optional
        Sample weights for weighted feature effect calculations [n_samples]
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing:
        - Grid values (feature values)
        - Partial dependence values (predictions)
    """
    # Generate feature values to evaluate
    grid_values = np.linspace(feature_range[0], feature_range[1], n_points)
    
    # Compute partial dependence
    pd_values = []
    
    for value in grid_values:
        # Create a copy of the data
        X_modified = X.clone()
        
        # Replace the feature with the current value
        X_modified[:, feature_idx] = float(value)
        
        # Get predictions
        with torch.no_grad():
            if sample_weights is not None:
                predictions = model.predict(X_modified, sample_weights=sample_weights)
            else:
                predictions = model.predict(X_modified)
        
        # Extract relevant prediction
        if task_name is None:
            task_name = list(predictions['task_outputs'].keys())[0]
        
        if target == 'risk_score':
            pred_value = predictions['task_outputs'][task_name]['risk_score'].mean().item()
        elif target == 'survival':
            if time_bin is None:
                time_bin = 0  # Default to first time bin
            pred_value = predictions['task_outputs'][task_name]['survival'][:, time_bin].mean().item()
        elif target == 'hazard':
            if time_bin is None:
                time_bin = 0  # Default to first time bin
            pred_value = predictions['task_outputs'][task_name]['hazard'][:, time_bin].mean().item()
        else:
            raise ValueError(f"Unsupported target: {target}")
        
        pd_values.append(pred_value)
    
    # Convert to numpy array
    pd_values = np.array(pd_values)
    
    return grid_values, pd_values


def plot_ice_curves(
    model: Any,
    X: torch.Tensor,
    feature_idx: int,
    feature_name: Optional[str] = None,
    n_points: int = 20,
    target: str = 'risk_score',
    task_name: Optional[str] = None,
    time_bin: Optional[int] = None,
    sample_size: Optional[int] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    feature_range: Optional[Tuple[float, float]] = None,
    show_mean: bool = True,
    center_curves: bool = True,
    alpha: float = 0.1,
    line_color: str = 'blue',
    mean_color: str = 'red',
    ax: Optional[plt.Axes] = None
) -> Figure:
    """
    Plot individual conditional expectation (ICE) curves.
    
    Parameters
    ----------
    model : Any
        The trained model to use for predictions
        
    X : torch.Tensor
        Input features [n_samples, n_features]
        
    feature_idx : int
        Index of the feature to analyze
        
    feature_name : str, optional
        Name of the feature for plotting
        
    n_points : int, default=20
        Number of points to evaluate
        
    target : str, default='risk_score'
        Target prediction to analyze ('risk_score', 'survival', 'hazard')
        
    task_name : str, optional
        Name of the task for multi-task models
        
    time_bin : int, optional
        Time bin for survival/hazard functions (required if target is not 'risk_score')
        
    sample_size : int, optional
        Number of samples to use for ICE curves (uses all if None)
        
    figsize : Tuple[float, float], default=(10, 6)
        Figure size if ax is not provided
        
    title : str, optional
        Plot title (default is auto-generated)
        
    xlabel : str, optional
        X-axis label (default is feature_name)
        
    ylabel : str, optional
        Y-axis label (default is based on target)
        
    feature_range : Tuple[float, float], optional
        Range of feature values to explore (default is data min/max)
        
    show_mean : bool, default=True
        Whether to show the mean curve (partial dependence)
        
    center_curves : bool, default=True
        Whether to center the curves at their mean
        
    alpha : float, default=0.1
        Transparency for individual curves
        
    line_color : str, default='blue'
        Color for individual curves
        
    mean_color : str, default='red'
        Color for mean curve
        
    ax : plt.Axes, optional
        Axes to plot on (creates new figure if not provided)
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    # Create new figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Set feature name if not provided
    if feature_name is None:
        feature_name = f"Feature {feature_idx}"
    
    # Set axis labels if not provided
    if xlabel is None:
        xlabel = feature_name
    
    if ylabel is None:
        if target == 'risk_score':
            ylabel = "Risk Score"
        elif target == 'survival':
            ylabel = f"Survival Probability (t={time_bin})" if time_bin is not None else "Survival Probability"
        elif target == 'hazard':
            ylabel = f"Hazard Rate (t={time_bin})" if time_bin is not None else "Hazard Rate"
        else:
            ylabel = "Prediction"
    
    # Set title if not provided
    if title is None:
        title = f"Individual Conditional Expectation Curves for {feature_name}"
    
    # Extract feature values
    feature_values = X[:, feature_idx].numpy()
    
    # Determine feature range
    if feature_range is None:
        min_val = np.min(feature_values)
        max_val = np.max(feature_values)
        # Add some padding
        range_width = max_val - min_val
        min_val -= range_width * 0.1
        max_val += range_width * 0.1
        feature_range = (min_val, max_val)
    
    # Generate feature values to evaluate
    grid_values = np.linspace(feature_range[0], feature_range[1], n_points)
    
    # Sample instances if requested
    if sample_size is not None and sample_size < X.shape[0]:
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sampled = X[indices]
    else:
        X_sampled = X
    
    # Compute ICE curves
    ice_curves = []
    
    for i in range(X_sampled.shape[0]):
        # Create a copy of the instance
        X_instance = X_sampled[i:i+1].repeat(n_points, 1)
        
        # Set feature values
        for j in range(n_points):
            X_instance[j, feature_idx] = float(grid_values[j])
        
        # Get predictions
        with torch.no_grad():
            predictions = model.predict(X_instance)
        
        # Extract relevant prediction
        if task_name is None:
            task_name = list(predictions['task_outputs'].keys())[0]
        
        if target == 'risk_score':
            curve = predictions['task_outputs'][task_name]['risk_score'].cpu().numpy()
        elif target == 'survival':
            if time_bin is None:
                time_bin = 0  # Default to first time bin
            curve = predictions['task_outputs'][task_name]['survival'][:, time_bin].cpu().numpy()
        elif target == 'hazard':
            if time_bin is None:
                time_bin = 0  # Default to first time bin
            curve = predictions['task_outputs'][task_name]['hazard'][:, time_bin].cpu().numpy()
        else:
            raise ValueError(f"Unsupported target: {target}")
        
        ice_curves.append(curve)
    
    # Convert to numpy array
    ice_curves = np.array(ice_curves)
    
    # Center curves if requested
    if center_curves:
        # Subtract the value at the minimum feature value from each curve
        ice_curves = ice_curves - ice_curves[:, 0].reshape(-1, 1)
    
    # Compute mean curve
    mean_curve = np.mean(ice_curves, axis=0)
    
    # Plot individual curves
    for i in range(ice_curves.shape[0]):
        ax.plot(
            grid_values,
            ice_curves[i],
            color=line_color,
            alpha=alpha,
            linewidth=0.5
        )
    
    # Plot mean curve if requested
    if show_mean:
        ax.plot(
            grid_values,
            mean_curve,
            color=mean_color,
            linewidth=2,
            label="Mean Effect"
        )
        
        # Add legend
        ax.legend(loc='best')
    
    # Set axis labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_feature_interaction(
    model: Any,
    X: torch.Tensor,
    feature1_idx: int,
    feature2_idx: int,
    feature1_name: Optional[str] = None,
    feature2_name: Optional[str] = None,
    n_points: int = 10,
    target: str = 'risk_score',
    task_name: Optional[str] = None,
    time_bin: Optional[int] = None,
    figsize: Tuple[float, float] = (10, 8),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    zlabel: Optional[str] = None,
    feature1_range: Optional[Tuple[float, float]] = None,
    feature2_range: Optional[Tuple[float, float]] = None,
    plot_type: str = 'contour',
    ax: Optional[plt.Axes] = None
) -> Figure:
    """
    Plot interaction effects between two features.
    
    Parameters
    ----------
    model : Any
        The trained model to use for predictions
        
    X : torch.Tensor
        Input features [n_samples, n_features]
        
    feature1_idx : int
        Index of the first feature
        
    feature2_idx : int
        Index of the second feature
        
    feature1_name : str, optional
        Name of the first feature for plotting
        
    feature2_name : str, optional
        Name of the second feature for plotting
        
    n_points : int, default=10
        Number of points to evaluate for each feature
        
    target : str, default='risk_score'
        Target prediction to analyze ('risk_score', 'survival', 'hazard')
        
    task_name : str, optional
        Name of the task for multi-task models
        
    time_bin : int, optional
        Time bin for survival/hazard functions (required if target is not 'risk_score')
        
    figsize : Tuple[float, float], default=(10, 8)
        Figure size if ax is not provided
        
    title : str, optional
        Plot title (default is auto-generated)
        
    xlabel : str, optional
        X-axis label (default is feature1_name)
        
    ylabel : str, optional
        Y-axis label (default is feature2_name)
        
    zlabel : str, optional
        Z-axis label for 3D plots (default is based on target)
        
    feature1_range : Tuple[float, float], optional
        Range of values for feature 1 (default is data min/max)
        
    feature2_range : Tuple[float, float], optional
        Range of values for feature 2 (default is data min/max)
        
    plot_type : str, default='contour'
        Type of plot ('contour', 'surface', 'heatmap')
        
    ax : plt.Axes, optional
        Axes to plot on (creates new figure if not provided)
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    # Create new figure if needed
    if ax is None:
        if plot_type == 'surface':
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Set feature names if not provided
    if feature1_name is None:
        feature1_name = f"Feature {feature1_idx}"
    
    if feature2_name is None:
        feature2_name = f"Feature {feature2_idx}"
    
    # Set axis labels if not provided
    if xlabel is None:
        xlabel = feature1_name
    
    if ylabel is None:
        ylabel = feature2_name
    
    if zlabel is None:
        if target == 'risk_score':
            zlabel = "Risk Score"
        elif target == 'survival':
            zlabel = f"Survival Probability (t={time_bin})" if time_bin is not None else "Survival Probability"
        elif target == 'hazard':
            zlabel = f"Hazard Rate (t={time_bin})" if time_bin is not None else "Hazard Rate"
        else:
            zlabel = "Prediction"
    
    # Set title if not provided
    if title is None:
        title = f"Interaction Effect between {feature1_name} and {feature2_name}"
    
    # Extract feature values
    feature1_values = X[:, feature1_idx].numpy()
    feature2_values = X[:, feature2_idx].numpy()
    
    # Determine feature ranges
    if feature1_range is None:
        min_val = np.min(feature1_values)
        max_val = np.max(feature1_values)
        # Add some padding
        range_width = max_val - min_val
        min_val -= range_width * 0.1
        max_val += range_width * 0.1
        feature1_range = (min_val, max_val)
    
    if feature2_range is None:
        min_val = np.min(feature2_values)
        max_val = np.max(feature2_values)
        # Add some padding
        range_width = max_val - min_val
        min_val -= range_width * 0.1
        max_val += range_width * 0.1
        feature2_range = (min_val, max_val)
    
    # Generate grid values
    grid1 = np.linspace(feature1_range[0], feature1_range[1], n_points)
    grid2 = np.linspace(feature2_range[0], feature2_range[1], n_points)
    
    # Create meshgrid
    X1, X2 = np.meshgrid(grid1, grid2)
    
    # Compute predictions for each grid point
    Z = np.zeros((n_points, n_points))
    
    for i in range(n_points):
        for j in range(n_points):
            # Create a batch with the average values for all instances
            X_modified = torch.mean(X, dim=0, keepdim=True).repeat(1, 1)
            
            # Set feature values
            X_modified[0, feature1_idx] = float(grid1[i])
            X_modified[0, feature2_idx] = float(grid2[j])
            
            # Get predictions
            with torch.no_grad():
                predictions = model.predict(X_modified)
            
            # Extract relevant prediction
            if task_name is None:
                task_name = list(predictions['task_outputs'].keys())[0]
            
            if target == 'risk_score':
                Z[j, i] = predictions['task_outputs'][task_name]['risk_score'].item()
            elif target == 'survival':
                if time_bin is None:
                    time_bin = 0  # Default to first time bin
                Z[j, i] = predictions['task_outputs'][task_name]['survival'][0, time_bin].item()
            elif target == 'hazard':
                if time_bin is None:
                    time_bin = 0  # Default to first time bin
                Z[j, i] = predictions['task_outputs'][task_name]['hazard'][0, time_bin].item()
            else:
                raise ValueError(f"Unsupported target: {target}")
    
    # Create plot based on type
    if plot_type == 'contour':
        # Contour plot
        contour = ax.contourf(X1, X2, Z, levels=20, cmap='viridis')
        fig.colorbar(contour, ax=ax, label=zlabel)
        
        # Add contour lines
        contour_lines = ax.contour(X1, X2, Z, levels=10, colors='black', alpha=0.5, linewidths=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=8)
        
    elif plot_type == 'surface':
        # 3D surface plot
        surf = ax.plot_surface(
            X1, X2, Z,
            cmap='viridis',
            linewidth=0,
            antialiased=True,
            alpha=0.8
        )
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=zlabel)
        
        # Set z-axis label
        ax.set_zlabel(zlabel)
        
    elif plot_type == 'heatmap':
        # Heatmap
        im = ax.imshow(
            Z,
            origin='lower',
            extent=[
                feature1_range[0], feature1_range[1],
                feature2_range[0], feature2_range[1]
            ],
            aspect='auto',
            cmap='viridis'
        )
        fig.colorbar(im, ax=ax, label=zlabel)
    
    # Set axis labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Add grid
    if plot_type != 'heatmap':
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_effect_modifier(
    model: Any,
    X: torch.Tensor,
    feature_idx: int,
    modifier_idx: int,
    feature_name: Optional[str] = None,
    modifier_name: Optional[str] = None,
    n_points: int = 20,
    target: str = 'risk_score',
    task_name: Optional[str] = None,
    time_bin: Optional[int] = None,
    modifier_values: Optional[List[float]] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    feature_range: Optional[Tuple[float, float]] = None,
    colors: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None
) -> Figure:
    """
    Plot how a modifier feature affects the relationship between a feature and the prediction.
    
    Parameters
    ----------
    model : Any
        The trained model to use for predictions
        
    X : torch.Tensor
        Input features [n_samples, n_features]
        
    feature_idx : int
        Index of the main feature
        
    modifier_idx : int
        Index of the modifier feature
        
    feature_name : str, optional
        Name of the main feature for plotting
        
    modifier_name : str, optional
        Name of the modifier feature for plotting
        
    n_points : int, default=20
        Number of points to evaluate for the main feature
        
    target : str, default='risk_score'
        Target prediction to analyze ('risk_score', 'survival', 'hazard')
        
    task_name : str, optional
        Name of the task for multi-task models
        
    time_bin : int, optional
        Time bin for survival/hazard functions (required if target is not 'risk_score')
        
    modifier_values : List[float], optional
        Specific values of the modifier to evaluate (default is quantiles)
        
    figsize : Tuple[float, float], default=(10, 6)
        Figure size if ax is not provided
        
    title : str, optional
        Plot title (default is auto-generated)
        
    xlabel : str, optional
        X-axis label (default is feature_name)
        
    ylabel : str, optional
        Y-axis label (default is based on target)
        
    feature_range : Tuple[float, float], optional
        Range of values for the main feature (default is data min/max)
        
    colors : List[str], optional
        Colors for different modifier values
        
    ax : plt.Axes, optional
        Axes to plot on (creates new figure if not provided)
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    # Create new figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Set feature names if not provided
    if feature_name is None:
        feature_name = f"Feature {feature_idx}"
    
    if modifier_name is None:
        modifier_name = f"Feature {modifier_idx}"
    
    # Set axis labels if not provided
    if xlabel is None:
        xlabel = feature_name
    
    if ylabel is None:
        if target == 'risk_score':
            ylabel = "Risk Score"
        elif target == 'survival':
            ylabel = f"Survival Probability (t={time_bin})" if time_bin is not None else "Survival Probability"
        elif target == 'hazard':
            ylabel = f"Hazard Rate (t={time_bin})" if time_bin is not None else "Hazard Rate"
        else:
            ylabel = "Prediction"
    
    # Set title if not provided
    if title is None:
        title = f"Effect of {feature_name} on {target.replace('_', ' ').title()} Moderated by {modifier_name}"
    
    # Extract feature values
    feature_values = X[:, feature_idx].numpy()
    modifier_values_data = X[:, modifier_idx].numpy()
    
    # Determine feature range
    if feature_range is None:
        min_val = np.min(feature_values)
        max_val = np.max(feature_values)
        # Add some padding
        range_width = max_val - min_val
        min_val -= range_width * 0.1
        max_val += range_width * 0.1
        feature_range = (min_val, max_val)
    
    # Determine modifier values to evaluate
    if modifier_values is None:
        # Use quantiles
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        modifier_values = [np.quantile(modifier_values_data, q) for q in quantiles]
    
    # Default colors
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(modifier_values)))
    
    # Generate feature values to evaluate
    grid_values = np.linspace(feature_range[0], feature_range[1], n_points)
    
    # Create a base instance with average values
    X_base = torch.mean(X, dim=0, keepdim=True)
    
    # Plot curves for each modifier value
    for i, mod_val in enumerate(modifier_values):
        pd_values = []
        
        for feat_val in grid_values:
            # Create a modified instance
            X_modified = X_base.clone()
            
            # Set feature values
            X_modified[0, feature_idx] = float(feat_val)
            X_modified[0, modifier_idx] = float(mod_val)
            
            # Get predictions
            with torch.no_grad():
                predictions = model.predict(X_modified)
            
            # Extract relevant prediction
            if task_name is None:
                task_name = list(predictions['task_outputs'].keys())[0]
            
            if target == 'risk_score':
                pred_value = predictions['task_outputs'][task_name]['risk_score'].item()
            elif target == 'survival':
                if time_bin is None:
                    time_bin = 0  # Default to first time bin
                pred_value = predictions['task_outputs'][task_name]['survival'][0, time_bin].item()
            elif target == 'hazard':
                if time_bin is None:
                    time_bin = 0  # Default to first time bin
                pred_value = predictions['task_outputs'][task_name]['hazard'][0, time_bin].item()
            else:
                raise ValueError(f"Unsupported target: {target}")
            
            pd_values.append(pred_value)
        
        # Convert to numpy array
        pd_values = np.array(pd_values)
        
        # Plot curve for this modifier value
        color = colors[i % len(colors)]
        label = f"{modifier_name} = {mod_val:.2f}"
        ax.plot(grid_values, pd_values, color=color, linewidth=2, label=label)
    
    # Set axis labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='best')
    
    plt.tight_layout()
    return fig
