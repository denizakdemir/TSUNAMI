"""
Visualization functions for uncertainty quantification.

This module provides functions for visualizing uncertainty in predictions,
including:
- Prediction intervals
- Uncertainty heatmaps
- Calibration of uncertainty estimates
- Ensemble predictions
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any
from matplotlib.figure import Figure
import seaborn as sns


def plot_prediction_intervals(
    mean_prediction: np.ndarray,
    std_prediction: np.ndarray,
    time_points: np.ndarray,
    confidence_level: float = 0.95,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Survival Prediction with Uncertainty",
    xlabel: str = "Time",
    ylabel: str = "Survival Probability",
    color: str = "blue",
    ax: Optional[plt.Axes] = None
) -> Figure:
    """
    Plot prediction intervals for survival curves.
    
    Parameters
    ----------
    mean_prediction : np.ndarray
        Mean predicted survival probabilities [num_time_bins]
        
    std_prediction : np.ndarray
        Standard deviation of predictions [num_time_bins]
        
    time_points : np.ndarray
        Time points corresponding to predictions
        
    confidence_level : float, default=0.95
        Confidence level for prediction intervals (0.95 = 95%)
        
    figsize : Tuple[float, float], default=(10, 6)
        Figure size if ax is not provided
        
    title : str, default="Survival Prediction with Uncertainty"
        Plot title
        
    xlabel : str, default="Time"
        X-axis label
        
    ylabel : str, default="Survival Probability"
        Y-axis label
        
    color : str, default="blue"
        Line and fill color
        
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
    
    # Z-value for the specified confidence level
    z_value = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence_level, 1.96)
    
    # Compute prediction intervals
    lower_bound = np.clip(mean_prediction - z_value * std_prediction, 0, 1)
    upper_bound = np.clip(mean_prediction + z_value * std_prediction, 0, 1)
    
    # Plot mean prediction
    line, = ax.step(
        time_points,
        mean_prediction,
        where='post',
        label="Mean Prediction",
        color=color
    )
    
    # Add prediction intervals
    ax.fill_between(
        time_points,
        lower_bound,
        upper_bound,
        alpha=0.3,
        color=color,
        step='post',
        label=f"{int(confidence_level*100)}% Prediction Interval"
    )
    
    # Set axis labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.05)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc="best")
    
    plt.tight_layout()
    return fig


def plot_uncertainty_heatmap(
    uncertainty_values: np.ndarray,
    time_points: np.ndarray,
    patient_ids: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (12, 8),
    title: str = "Prediction Uncertainty Heatmap",
    xlabel: str = "Time",
    ylabel: str = "Patient ID",
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None
) -> Figure:
    """
    Plot heatmap of prediction uncertainty across patients and time.
    
    Parameters
    ----------
    uncertainty_values : np.ndarray
        Uncertainty values (e.g., standard deviations) [num_patients, num_time_bins]
        
    time_points : np.ndarray
        Time points corresponding to predictions
        
    patient_ids : np.ndarray, optional
        Patient identifiers for y-axis labels
        
    figsize : Tuple[float, float], default=(12, 8)
        Figure size if ax is not provided
        
    title : str, default="Prediction Uncertainty Heatmap"
        Plot title
        
    xlabel : str, default="Time"
        X-axis label
        
    ylabel : str, default="Patient ID"
        Y-axis label
        
    cmap : str, default="viridis"
        Colormap for heatmap
        
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
    
    # Generate patient IDs if not provided
    if patient_ids is None:
        patient_ids = np.arange(uncertainty_values.shape[0])
    
    # Create heatmap
    im = ax.imshow(
        uncertainty_values,
        aspect='auto',
        cmap=cmap,
        interpolation='nearest',
        extent=[time_points[0], time_points[-1], len(patient_ids)-0.5, -0.5]
    )
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Uncertainty (Std. Dev.)", rotation=-90, va="bottom")
    
    # Set axis labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Set y-ticks for patient IDs (limit to avoid overcrowding)
    max_yticks = 30
    if len(patient_ids) > max_yticks:
        ytick_step = len(patient_ids) // max_yticks
        ytick_indices = np.arange(0, len(patient_ids), ytick_step)
        ax.set_yticks(ytick_indices)
        ax.set_yticklabels([str(patient_ids[i]) for i in ytick_indices])
    else:
        ax.set_yticks(np.arange(len(patient_ids)))
        ax.set_yticklabels([str(pid) for pid in patient_ids])
    
    plt.tight_layout()
    return fig


def plot_uncertainty_calibration(
    uncertainty_values: np.ndarray,
    prediction_errors: np.ndarray,
    num_bins: int = 10,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Uncertainty Calibration",
    xlabel: str = "Predicted Uncertainty (Std. Dev.)",
    ylabel: str = "Observed Error (RMSE)",
    ax: Optional[plt.Axes] = None
) -> Figure:
    """
    Plot calibration of uncertainty estimates.
    
    Parameters
    ----------
    uncertainty_values : np.ndarray
        Uncertainty values (e.g., standard deviations) [num_samples]
        
    prediction_errors : np.ndarray
        Observed prediction errors [num_samples]
        
    num_bins : int, default=10
        Number of bins for grouping uncertainty values
        
    figsize : Tuple[float, float], default=(10, 6)
        Figure size if ax is not provided
        
    title : str, default="Uncertainty Calibration"
        Plot title
        
    xlabel : str, default="Predicted Uncertainty (Std. Dev.)"
        X-axis label
        
    ylabel : str, default="Observed Error (RMSE)"
        Y-axis label
        
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
    
    # Create bins based on uncertainty values
    bins = np.linspace(
        np.min(uncertainty_values),
        np.max(uncertainty_values),
        num_bins + 1
    )
    
    # Initialize arrays for bin statistics
    bin_means_uncertainty = []
    bin_means_error = []
    bin_counts = []
    
    # Compute statistics for each bin
    for i in range(len(bins) - 1):
        # Find samples in this bin
        bin_mask = (uncertainty_values >= bins[i]) & (uncertainty_values < bins[i+1])
        
        if np.sum(bin_mask) > 0:
            # Compute mean uncertainty and error for this bin
            mean_uncertainty = np.mean(uncertainty_values[bin_mask])
            mean_error = np.mean(prediction_errors[bin_mask])
            count = np.sum(bin_mask)
            
            bin_means_uncertainty.append(mean_uncertainty)
            bin_means_error.append(mean_error)
            bin_counts.append(count)
    
    # Convert to numpy arrays
    bin_means_uncertainty = np.array(bin_means_uncertainty)
    bin_means_error = np.array(bin_means_error)
    bin_counts = np.array(bin_counts)
    
    # Size points according to bin counts
    relative_sizes = 20 + 100 * (bin_counts / np.max(bin_counts))
    
    # Plot calibration curve
    ax.scatter(
        bin_means_uncertainty,
        bin_means_error,
        s=relative_sizes,
        alpha=0.7,
        edgecolors='black'
    )
    
    # Add count labels
    for i in range(len(bin_means_uncertainty)):
        ax.annotate(
            f"{bin_counts[i]}",
            (bin_means_uncertainty[i], bin_means_error[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    # Add diagonal line (perfect calibration: uncertainty = error)
    min_val = min(np.min(bin_means_uncertainty), np.min(bin_means_error))
    max_val = max(np.max(bin_means_uncertainty), np.max(bin_means_error))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect calibration')
    
    # Add regression line
    try:
        from scipy import stats
        slope, intercept, r_value, p_value, _ = stats.linregress(
            bin_means_uncertainty,
            bin_means_error
        )
        
        x_reg = np.array([min_val, max_val])
        y_reg = intercept + slope * x_reg
        
        ax.plot(
            x_reg,
            y_reg,
            'r-',
            label=f'Regression line (r²={r_value**2:.2f})'
        )
    except ImportError:
        pass
    
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


def plot_ensemble_predictions(
    prediction_samples: np.ndarray,
    time_points: np.ndarray,
    max_curves: int = 50,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Ensemble Predictions",
    xlabel: str = "Time",
    ylabel: str = "Survival Probability",
    color: str = "blue",
    alpha: float = 0.1,
    ax: Optional[plt.Axes] = None
) -> Figure:
    """
    Plot ensemble predictions from multiple model samples.
    
    Parameters
    ----------
    prediction_samples : np.ndarray
        Samples from ensemble predictions [num_samples, num_time_bins]
        
    time_points : np.ndarray
        Time points corresponding to predictions
        
    max_curves : int, default=50
        Maximum number of individual curves to plot
        
    figsize : Tuple[float, float], default=(10, 6)
        Figure size if ax is not provided
        
    title : str, default="Ensemble Predictions"
        Plot title
        
    xlabel : str, default="Time"
        X-axis label
        
    ylabel : str, default="Survival Probability"
        Y-axis label
        
    color : str, default="blue"
        Line color
        
    alpha : float, default=0.1
        Transparency for individual curves
        
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
    
    # Calculate mean prediction
    mean_prediction = np.mean(prediction_samples, axis=0)
    
    # Calculate confidence intervals
    lower_95 = np.percentile(prediction_samples, 2.5, axis=0)
    upper_95 = np.percentile(prediction_samples, 97.5, axis=0)
    
    # Limit number of curves to plot
    num_samples = min(prediction_samples.shape[0], max_curves)
    
    # Plot individual curves
    for i in range(num_samples):
        ax.step(
            time_points,
            prediction_samples[i],
            where='post',
            color=color,
            alpha=alpha,
            linewidth=0.5
        )
    
    # Plot mean prediction
    ax.step(
        time_points,
        mean_prediction,
        where='post',
        label="Mean Prediction",
        color=color,
        linewidth=2
    )
    
    # Add confidence intervals
    ax.fill_between(
        time_points,
        lower_95,
        upper_95,
        step='post',
        alpha=0.3,
        color=color,
        label="95% Confidence Interval"
    )
    
    # Set axis labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.05)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc="best")
    
    plt.tight_layout()
    return fig


def plot_credible_regions(
    time_points: np.ndarray,
    prediction_samples: np.ndarray,
    quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95],
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Survival Prediction Credible Regions",
    xlabel: str = "Time",
    ylabel: str = "Survival Probability",
    cmap: str = "Blues",
    ax: Optional[plt.Axes] = None
) -> Figure:
    """
    Plot credible regions (quantiles) for survival predictions.
    
    Parameters
    ----------
    time_points : np.ndarray
        Time points corresponding to predictions
        
    prediction_samples : np.ndarray
        Samples from posterior distribution [num_samples, num_time_bins]
        
    quantiles : List[float], default=[0.05, 0.25, 0.5, 0.75, 0.95]
        Quantiles to plot
        
    figsize : Tuple[float, float], default=(10, 6)
        Figure size if ax is not provided
        
    title : str, default="Survival Prediction Credible Regions"
        Plot title
        
    xlabel : str, default="Time"
        X-axis label
        
    ylabel : str, default="Survival Probability"
        Y-axis label
        
    cmap : str, default="Blues"
        Colormap for credible regions
        
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
    
    # Ensure quantiles are sorted
    quantiles = sorted(quantiles)
    
    # Calculate quantiles
    quantile_curves = {}
    for q in quantiles:
        quantile_curves[q] = np.percentile(prediction_samples, q * 100, axis=0)
    
    # Get colormap
    cmap_obj = plt.cm.get_cmap(cmap)
    
    # Plot regions between pairs of quantiles
    for i in range(len(quantiles) // 2):
        lower_q = quantiles[i]
        upper_q = quantiles[-(i+1)]
        
        # Skip if lower_q is greater than or equal to upper_q
        if lower_q >= upper_q:
            continue
        
        # Color intensity based on how central the region is
        # Outer regions are lighter, inner regions are darker
        color_intensity = 0.3 + 0.7 * (i / (len(quantiles) // 2))
        fill_color = cmap_obj(color_intensity)
        
        ax.fill_between(
            time_points,
            quantile_curves[lower_q],
            quantile_curves[upper_q],
            step='post',
            alpha=0.6,
            color=fill_color,
            label=f"{int(upper_q*100)}-{int(lower_q*100)}% Credible Region"
        )
    
    # Plot median
    if 0.5 in quantile_curves:
        ax.step(
            time_points,
            quantile_curves[0.5],
            where='post',
            color='black',
            linewidth=2,
            label="Median"
        )
    
    # Set axis labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.05)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc="best")
    
    plt.tight_layout()
    return fig