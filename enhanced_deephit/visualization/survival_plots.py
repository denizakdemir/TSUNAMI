"""
Visualization functions for survival analysis.

This module provides functions for plotting various survival analysis
visualizations, including:
- Survival curves
- Cumulative incidence functions (CIF)
- Calibration curves
- Risk stratification curves
- Landmark analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any
from matplotlib.figure import Figure


def plot_survival_curve(
    survival_curves: Union[np.ndarray, List[np.ndarray]],
    time_points: np.ndarray,
    labels: Optional[List[str]] = None,
    uncertainty: Optional[np.ndarray] = None,
    confidence_level: float = 0.95,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Predicted Survival Curve",
    risk_table: bool = False,
    xlabel: str = "Time",
    ylabel: str = "Survival Probability",
    legend_loc: str = "best",
    colors: Optional[List[str]] = None,
    linestyles: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None
) -> Figure:
    """
    Plot survival curves.
    
    Parameters
    ----------
    survival_curves : np.ndarray or List[np.ndarray]
        Survival probabilities, either a single curve [num_time_bins] or
        multiple curves [[num_samples, num_time_bins]] or list of curves
        
    time_points : np.ndarray
        Time points corresponding to survival probabilities
        
    labels : List[str], optional
        Labels for each curve (for legend)
        
    uncertainty : np.ndarray, optional
        Standard deviation or confidence interval for uncertainty visualization
        
    confidence_level : float, default=0.95
        Confidence level for uncertainty bands (0.95 = 95%)
        
    figsize : Tuple[float, float], default=(10, 6)
        Figure size if ax is not provided
        
    title : str, default="Predicted Survival Curve"
        Plot title
        
    risk_table : bool, default=False
        Whether to include a risk table below the plot
        
    xlabel : str, default="Time"
        X-axis label
        
    ylabel : str, default="Survival Probability"
        Y-axis label
        
    legend_loc : str, default="best"
        Legend location
        
    colors : List[str], optional
        Colors for each curve
        
    linestyles : List[str], optional
        Line styles for each curve
        
    ax : plt.Axes, optional
        Axes to plot on (creates new figure if not provided)
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    # Convert input to numpy array if needed
    if isinstance(survival_curves, list):
        survival_curves = np.array(survival_curves)
    
    # Create new figure if needed
    if ax is None:
        if risk_table:
            fig, (ax, risk_ax) = plt.subplots(
                2, 1, 
                figsize=figsize, 
                gridspec_kw={'height_ratios': [3, 1]},
                sharex=True
            )
        else:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Default colors and linestyles
    if colors is None:
        colors = plt.cm.tab10.colors
    
    if linestyles is None:
        linestyles = ['-'] * 10
    
    # Plot single curve or multiple curves
    if survival_curves.ndim == 1:
        # Single curve
        line, = ax.step(
            time_points, 
            survival_curves, 
            where='post', 
            label=labels[0] if labels else None,
            color=colors[0],
            linestyle=linestyles[0]
        )
        
        # Add uncertainty band if provided
        if uncertainty is not None:
            z_value = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence_level, 1.96)
            lower_bound = np.clip(survival_curves - z_value * uncertainty, 0, 1)
            upper_bound = np.clip(survival_curves + z_value * uncertainty, 0, 1)
            
            ax.fill_between(
                time_points,
                lower_bound,
                upper_bound,
                alpha=0.3,
                color=line.get_color(),
                step='post',
                label=f"{int(confidence_level*100)}% Confidence Interval"
            )
    else:
        # Multiple curves
        n_curves = len(survival_curves)
        
        for i in range(n_curves):
            curve = survival_curves[i]
            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]
            label = labels[i] if labels and i < len(labels) else f"Curve {i+1}"
            
            line, = ax.step(
                time_points, 
                curve, 
                where='post', 
                label=label,
                color=color,
                linestyle=linestyle
            )
            
            # Add uncertainty band if provided
            if uncertainty is not None and uncertainty.ndim > 1 and i < len(uncertainty):
                z_value = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence_level, 1.96)
                lower_bound = np.clip(curve - z_value * uncertainty[i], 0, 1)
                upper_bound = np.clip(curve + z_value * uncertainty[i], 0, 1)
                
                ax.fill_between(
                    time_points,
                    lower_bound,
                    upper_bound,
                    alpha=0.3,
                    color=line.get_color(),
                    step='post'
                )
    
    # Set axis labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.05)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend if needed
    if labels or uncertainty is not None:
        ax.legend(loc=legend_loc)
    
    # Add risk table if requested
    if risk_table:
        # For single curve
        if survival_curves.ndim == 1:
            at_risk = np.ones(len(time_points))
            for i in range(1, len(time_points)):
                # Approximate number at risk based on survival probability
                at_risk[i] = at_risk[i-1] * (survival_curves[i-1] / survival_curves[i] if survival_curves[i] > 0 else 1)
            
            risk_ax.step(time_points, at_risk, where='post', color=colors[0])
            risk_ax.set_ylabel('Number at Risk')
            risk_ax.set_ylim(0, at_risk[0]*1.05)
            risk_ax.grid(True, alpha=0.3)
        else:
            # For multiple curves, we would need sample-level data to compute this accurately
            # This is an approximation
            for i in range(len(survival_curves)):
                curve = survival_curves[i]
                at_risk = np.ones(len(time_points))
                
                for j in range(1, len(time_points)):
                    at_risk[j] = at_risk[j-1] * (curve[j-1] / curve[j] if curve[j] > 0 else 1)
                
                risk_ax.step(
                    time_points, 
                    at_risk, 
                    where='post', 
                    color=colors[i % len(colors)],
                    linestyle=linestyles[i % len(linestyles)]
                )
            
            risk_ax.set_ylabel('Number at Risk')
            risk_ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_cumulative_incidence(
    cif: Union[np.ndarray, List[np.ndarray]],
    time_points: np.ndarray,
    risk_names: List[str],
    uncertainty: Optional[np.ndarray] = None,
    confidence_level: float = 0.95,
    stacked: bool = False,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Cumulative Incidence Functions",
    xlabel: str = "Time",
    ylabel: str = "Cumulative Incidence",
    legend_loc: str = "best",
    colors: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None
) -> Figure:
    """
    Plot cumulative incidence functions (CIF) for competing risks.
    
    Parameters
    ----------
    cif : np.ndarray or List[np.ndarray]
        Cumulative incidence functions, shape [num_risks, num_time_bins]
        or [num_samples, num_risks, num_time_bins]
        
    time_points : np.ndarray
        Time points corresponding to CIF values
        
    risk_names : List[str]
        Names of competing risks
        
    uncertainty : np.ndarray, optional
        Standard deviation or confidence interval for uncertainty visualization
        
    confidence_level : float, default=0.95
        Confidence level for uncertainty bands (0.95 = 95%)
        
    stacked : bool, default=False
        Whether to create a stacked plot
        
    figsize : Tuple[float, float], default=(10, 6)
        Figure size if ax is not provided
        
    title : str, default="Cumulative Incidence Functions"
        Plot title
        
    xlabel : str, default="Time"
        X-axis label
        
    ylabel : str, default="Cumulative Incidence"
        Y-axis label
        
    legend_loc : str, default="best"
        Legend location
        
    colors : List[str], optional
        Colors for each risk
        
    ax : plt.Axes, optional
        Axes to plot on (creates new figure if not provided)
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    # Convert input to numpy array if needed
    if isinstance(cif, list):
        cif = np.array(cif)
    
    # Create new figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Default colors
    if colors is None:
        colors = plt.cm.tab10.colors
    
    # Determine dimensions
    if cif.ndim == 2:
        # Single patient, multiple risks
        num_risks = cif.shape[0]
        
        if stacked:
            # Stacked plot
            cumulative = np.zeros_like(time_points)
            
            for i in range(num_risks):
                ax.fill_between(
                    time_points,
                    cumulative,
                    cumulative + cif[i],
                    step='post',
                    alpha=0.7,
                    label=risk_names[i] if i < len(risk_names) else f"Risk {i+1}",
                    color=colors[i % len(colors)]
                )
                cumulative += cif[i]
        else:
            # Individual curves
            for i in range(num_risks):
                line, = ax.step(
                    time_points,
                    cif[i],
                    where='post',
                    label=risk_names[i] if i < len(risk_names) else f"Risk {i+1}",
                    color=colors[i % len(colors)]
                )
                
                # Add uncertainty band if provided
                if uncertainty is not None and i < uncertainty.shape[0]:
                    z_value = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence_level, 1.96)
                    lower_bound = np.clip(cif[i] - z_value * uncertainty[i], 0, 1)
                    upper_bound = np.clip(cif[i] + z_value * uncertainty[i], 0, 1)
                    
                    ax.fill_between(
                        time_points,
                        lower_bound,
                        upper_bound,
                        alpha=0.3,
                        color=line.get_color(),
                        step='post'
                    )
    else:
        # Multiple patients, multiple risks (use first patient)
        patient_idx = 0
        num_risks = cif.shape[1]
        
        if stacked:
            # Stacked plot
            cumulative = np.zeros_like(time_points)
            
            for i in range(num_risks):
                ax.fill_between(
                    time_points,
                    cumulative,
                    cumulative + cif[patient_idx, i],
                    step='post',
                    alpha=0.7,
                    label=risk_names[i] if i < len(risk_names) else f"Risk {i+1}",
                    color=colors[i % len(colors)]
                )
                cumulative += cif[patient_idx, i]
        else:
            # Individual curves
            for i in range(num_risks):
                line, = ax.step(
                    time_points,
                    cif[patient_idx, i],
                    where='post',
                    label=risk_names[i] if i < len(risk_names) else f"Risk {i+1}",
                    color=colors[i % len(colors)]
                )
                
                # Add uncertainty band if provided
                if uncertainty is not None and patient_idx < uncertainty.shape[0] and i < uncertainty.shape[1]:
                    z_value = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence_level, 1.96)
                    lower_bound = np.clip(cif[patient_idx, i] - z_value * uncertainty[patient_idx, i], 0, 1)
                    upper_bound = np.clip(cif[patient_idx, i] + z_value * uncertainty[patient_idx, i], 0, 1)
                    
                    ax.fill_between(
                        time_points,
                        lower_bound,
                        upper_bound,
                        alpha=0.3,
                        color=line.get_color(),
                        step='post'
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
    ax.legend(loc=legend_loc)
    
    plt.tight_layout()
    return fig


def plot_calibration_curve(
    survival_probs: np.ndarray,
    event_indicators: np.ndarray,
    event_times: np.ndarray,
    time_bin: int,
    num_groups: int = 10,
    figsize: Tuple[float, float] = (8, 8),
    title: str = "Calibration Curve",
    xlabel: str = "Predicted Probability",
    ylabel: str = "Observed Probability",
    ax: Optional[plt.Axes] = None
) -> Figure:
    """
    Plot calibration curve for survival predictions.
    
    Parameters
    ----------
    survival_probs : np.ndarray
        Predicted survival probabilities [num_samples, num_time_bins]
        
    event_indicators : np.ndarray
        Event indicators (1 if event occurred, 0 if censored) [num_samples]
        
    event_times : np.ndarray
        Event/censoring times (as time bin indices) [num_samples]
        
    time_bin : int
        Time bin index for calibration assessment
        
    num_groups : int, default=10
        Number of groups (quantiles) for calibration curve
        
    figsize : Tuple[float, float], default=(8, 8)
        Figure size if ax is not provided
        
    title : str, default="Calibration Curve"
        Plot title
        
    xlabel : str, default="Predicted Probability"
        X-axis label
        
    ylabel : str, default="Observed Probability"
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
    
    # Extract survival probabilities at the specified time bin
    if time_bin < survival_probs.shape[1]:
        surv_at_time = survival_probs[:, time_bin]
    else:
        # Use the last available time bin if the specified one is out of range
        surv_at_time = survival_probs[:, -1]
    
    # Compute event status at or before the time bin
    # 1 if event at or before time_bin, 0 otherwise
    event_status = np.zeros_like(event_indicators)
    for i in range(len(event_indicators)):
        if event_indicators[i] == 1 and event_times[i] <= time_bin:
            event_status[i] = 1
    
    # Compute observed survival (1 - event status)
    observed_surv = 1 - event_status
    
    # Create groups (quantiles) based on predicted survival
    quantiles = np.linspace(0, 1, num_groups + 1)
    thresholds = np.quantile(surv_at_time, quantiles)
    
    # Ensure unique thresholds
    thresholds = np.unique(thresholds)
    
    # Initialize arrays for plot
    mean_predicted = []
    mean_observed = []
    group_sizes = []
    
    # Compute mean predicted and observed survival for each group
    for i in range(len(thresholds) - 1):
        lower = thresholds[i]
        upper = thresholds[i+1]
        
        # Find samples in this group
        if i == len(thresholds) - 2:
            # Include upper bound in last group
            group_idx = (surv_at_time >= lower) & (surv_at_time <= upper)
        else:
            group_idx = (surv_at_time >= lower) & (surv_at_time < upper)
        
        if np.sum(group_idx) > 0:
            # Compute mean predicted survival
            mean_pred = np.mean(surv_at_time[group_idx])
            
            # Compute mean observed survival
            mean_obs = np.mean(observed_surv[group_idx])
            
            # Store results
            mean_predicted.append(mean_pred)
            mean_observed.append(mean_obs)
            group_sizes.append(np.sum(group_idx))
    
    # Convert to numpy arrays
    mean_predicted = np.array(mean_predicted)
    mean_observed = np.array(mean_observed)
    group_sizes = np.array(group_sizes)
    
    # Plot calibration curve
    ax.scatter(
        mean_predicted,
        mean_observed,
        s=group_sizes / np.max(group_sizes) * 100,
        alpha=0.7,
        edgecolors='black'
    )
    
    # Add group size annotations
    for i in range(len(mean_predicted)):
        ax.annotate(
            f"{group_sizes[i]}",
            (mean_predicted[i], mean_observed[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    # Add diagonal line (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    # Add lowess curve if available
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        y_smooth = lowess(mean_observed, mean_predicted, frac=0.6)
        ax.plot(y_smooth[:, 0], y_smooth[:, 1], 'r-', lw=2, label='Lowess curve')
    except ImportError:
        # Fallback to simple interpolation
        ax.plot(mean_predicted, mean_observed, 'r-', lw=2, label='Interpolation')
    
    # Set axis labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (Time Bin {time_bin})")
    
    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    return fig


def plot_risk_stratification(
    survival_curves: np.ndarray,
    time_points: np.ndarray,
    event_indicators: np.ndarray,
    event_times: np.ndarray,
    num_groups: int = 3,
    stratification_time: Optional[int] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Risk Stratification",
    xlabel: str = "Time",
    ylabel: str = "Survival Probability",
    legend_loc: str = "best",
    colors: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None
) -> Figure:
    """
    Plot Kaplan-Meier curves stratified by predicted risk.
    
    Parameters
    ----------
    survival_curves : np.ndarray
        Predicted survival probabilities [num_samples, num_time_bins]
        
    time_points : np.ndarray
        Time points corresponding to survival probabilities
        
    event_indicators : np.ndarray
        Event indicators (1 if event occurred, 0 if censored) [num_samples]
        
    event_times : np.ndarray
        Event/censoring times (as time bin indices) [num_samples]
        
    num_groups : int, default=3
        Number of risk groups
        
    stratification_time : int, optional
        Time bin for risk stratification (uses median survival time if None)
        
    figsize : Tuple[float, float], default=(10, 6)
        Figure size if ax is not provided
        
    title : str, default="Risk Stratification"
        Plot title
        
    xlabel : str, default="Time"
        X-axis label
        
    ylabel : str, default="Survival Probability"
        Y-axis label
        
    legend_loc : str, default="best"
        Legend location
        
    colors : List[str], optional
        Colors for each risk group
        
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
    
    # Default colors
    if colors is None:
        colors = ['green', 'blue', 'red']
        if num_groups > 3:
            colors = plt.cm.viridis(np.linspace(0, 1, num_groups))
    
    # Determine stratification time if not provided
    if stratification_time is None:
        # Use median survival time (time where survival = 0.5)
        median_survival = np.zeros(len(survival_curves))
        
        for i in range(len(survival_curves)):
            curve = survival_curves[i]
            # Find time bin where survival drops below 0.5
            below_median = np.where(curve <= 0.5)[0]
            
            if len(below_median) > 0:
                median_survival[i] = below_median[0]
            else:
                # Use last time bin if survival doesn't drop below 0.5
                median_survival[i] = len(curve) - 1
        
        # Use median of median survival times
        stratification_time = int(np.median(median_survival))
    
    # Extract survival probabilities at the stratification time
    surv_at_time = survival_curves[:, stratification_time]
    
    # Create risk groups using quantiles
    risk_quantiles = np.linspace(0, 1, num_groups + 1)
    risk_thresholds = np.quantile(surv_at_time, risk_quantiles)
    
    # Ensure unique thresholds
    risk_thresholds = np.unique(risk_thresholds)
    
    # Compute Kaplan-Meier curves for each risk group
    from lifelines import KaplanMeierFitter
    
    for i in range(len(risk_thresholds) - 1):
        lower = risk_thresholds[i]
        upper = risk_thresholds[i+1]
        
        # Find samples in this risk group
        if i == len(risk_thresholds) - 2:
            # Include upper bound in last group
            group_idx = (surv_at_time >= lower) & (surv_at_time <= upper)
        else:
            group_idx = (surv_at_time >= lower) & (surv_at_time < upper)
        
        if np.sum(group_idx) > 0:
            # Extract event indicators and times for this group
            group_events = event_indicators[group_idx]
            group_times = event_times[group_idx]
            
            # Convert to proper format for lifelines
            # Ensure times are not negative
            group_times = time_points[np.clip(group_times, 0, len(time_points) - 1)]
            
            # Fit Kaplan-Meier model
            kmf = KaplanMeierFitter()
            kmf.fit(
                group_times,
                event_observed=group_events,
                label=f"Risk Group {i+1} (n={np.sum(group_idx)})"
            )
            
            # Plot survival curve
            kmf.plot_survival_function(
                ax=ax,
                ci_show=True,
                color=colors[i % len(colors)]
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
    ax.legend(loc=legend_loc)
    
    # Add vertical line for stratification time
    ax.axvline(
        time_points[stratification_time],
        color='black',
        linestyle='--',
        alpha=0.5,
        label=f"Stratification Time"
    )
    
    plt.tight_layout()
    return fig


def plot_landmark_analysis(
    survival_curves: np.ndarray,
    time_points: np.ndarray,
    landmark_time: int,
    censor_before_landmark: bool = True,
    num_groups: int = 3,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Landmark Analysis",
    xlabel: str = "Time",
    ylabel: str = "Survival Probability",
    legend_loc: str = "best",
    colors: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None
) -> Figure:
    """
    Plot landmark analysis for survival predictions.
    
    Parameters
    ----------
    survival_curves : np.ndarray
        Predicted survival probabilities [num_samples, num_time_bins]
        
    time_points : np.ndarray
        Time points corresponding to survival probabilities
        
    landmark_time : int
        Time bin index for landmark time
        
    censor_before_landmark : bool, default=True
        Whether to censor events before landmark time
        
    num_groups : int, default=3
        Number of risk groups for stratification
        
    figsize : Tuple[float, float], default=(10, 6)
        Figure size if ax is not provided
        
    title : str, default="Landmark Analysis"
        Plot title
        
    xlabel : str, default="Time"
        X-axis label
        
    ylabel : str, default="Survival Probability"
        Y-axis label
        
    legend_loc : str, default="best"
        Legend location
        
    colors : List[str], optional
        Colors for each risk group
        
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
    
    # Default colors
    if colors is None:
        colors = ['green', 'blue', 'red']
        if num_groups > 3:
            colors = plt.cm.viridis(np.linspace(0, 1, num_groups))
    
    # Check that landmark time is valid
    if landmark_time >= len(time_points):
        landmark_time = len(time_points) - 1
    
    # Extract survival probabilities at the landmark time
    surv_at_landmark = survival_curves[:, landmark_time]
    
    # Create risk groups using quantiles
    risk_quantiles = np.linspace(0, 1, num_groups + 1)
    risk_thresholds = np.quantile(surv_at_landmark, risk_quantiles)
    
    # Ensure unique thresholds
    risk_thresholds = np.unique(risk_thresholds)
    
    # Plot survival curves for each risk group
    for i in range(len(risk_thresholds) - 1):
        lower = risk_thresholds[i]
        upper = risk_thresholds[i+1]
        
        # Find samples in this risk group
        if i == len(risk_thresholds) - 2:
            # Include upper bound in last group
            group_idx = (surv_at_landmark >= lower) & (surv_at_landmark <= upper)
        else:
            group_idx = (surv_at_landmark >= lower) & (surv_at_landmark < upper)
        
        if np.sum(group_idx) > 0:
            # Extract survival curves for this group
            group_curves = survival_curves[group_idx]
            
            # Compute conditional survival probability after landmark
            conditional_surv = np.zeros((len(group_curves), len(time_points) - landmark_time))
            
            for j in range(len(group_curves)):
                # Conditional survival: S(t | t ≥ landmark) = S(t) / S(landmark)
                conditional_surv[j] = group_curves[j, landmark_time:] / group_curves[j, landmark_time]
            
            # Average across samples in the group
            mean_conditional_surv = np.mean(conditional_surv, axis=0)
            
            # Confidence interval (95%)
            if len(group_curves) > 1:
                std_dev = np.std(conditional_surv, axis=0)
                ci_lower = np.clip(mean_conditional_surv - 1.96 * std_dev / np.sqrt(len(group_curves)), 0, 1)
                ci_upper = np.clip(mean_conditional_surv + 1.96 * std_dev / np.sqrt(len(group_curves)), 0, 1)
            else:
                ci_lower = mean_conditional_surv
                ci_upper = mean_conditional_surv
            
            # Plot
            line, = ax.step(
                time_points[landmark_time:],
                mean_conditional_surv,
                where='post',
                label=f"Risk Group {i+1} (n={np.sum(group_idx)})",
                color=colors[i % len(colors)]
            )
            
            # Add confidence interval
            ax.fill_between(
                time_points[landmark_time:],
                ci_lower,
                ci_upper,
                alpha=0.3,
                color=line.get_color(),
                step='post'
            )
    
    # Set axis labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (Landmark: {time_points[landmark_time]})")
    
    # Set y-axis limits
    ax.set_ylim(0, 1.05)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc=legend_loc)
    
    # Add vertical line for landmark time
    ax.axvline(
        time_points[landmark_time],
        color='black',
        linestyle='--',
        alpha=0.5,
        label=f"Landmark Time"
    )
    
    plt.tight_layout()
    return fig