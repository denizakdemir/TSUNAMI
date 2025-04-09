"""
Feature importance and effect measures for enhanced DeepHit model.

This module provides various methods for computing feature importance
and effect measures, including:
- Permutation importance
- SHAP (SHapley Additive exPlanations) values
- Integrated gradients
- Attention-based feature importance
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from tqdm import tqdm
from copy import deepcopy


class PermutationImportance:
    """
    Calculate feature importance using permutation approach.
    
    This method measures how much model prediction accuracy decreases when
    a single feature's values are randomly shuffled, breaking the relationship
    between the feature and the target.
    """
    
    def __init__(self, model):
        """
        Initialize permutation importance calculator.
        
        Parameters
        ----------
        model : EnhancedDeepHit
            The trained model for which to compute importance
        """
        self.model = model
    
    def compute_importance(
        self,
        X: Union[torch.Tensor, Dict[str, torch.Tensor]],
        y: Optional[torch.Tensor] = None,
        n_repeats: int = 5,
        feature_names: Optional[List[str]] = None,
        metric: str = 'risk_score',
        task_name: Optional[str] = None,
        cat_feature_info: Optional[List[Dict]] = None,
        use_original_names: bool = True,
        sample_weights: Optional[torch.Tensor] = None,
        pool_embeddings: bool = True
    ) -> Dict[str, float]:
        """
        Compute permutation importance scores.
        
        Parameters
        ----------
        X : torch.Tensor
            Input features [n_samples, n_features]
            
        y : torch.Tensor, optional
            Target values, used for computing performance metrics
            
        n_repeats : int, default=5
            Number of times to permute each feature
            
        feature_names : List[str], optional
            Names of features for labeling results
            
        metric : str, default='risk_score'
            Metric to use for importance calculation ('risk_score', 'c_index')
            
        task_name : str, optional
            Name of the task to compute importance for (if multi-task model)
            
        cat_feature_info : List[Dict], optional
            Information about categorical features for proper grouping
            
        use_original_names : bool, default=True
            Whether to use original variable names for categorical features
            
        sample_weights : torch.Tensor, optional
            Sample weights for weighted importance calculations [n_samples]
            
        pool_embeddings : bool, default=True
            Whether to pool importance scores for embedding dimensions of the same feature
            
        Returns
        -------
        Dict[str, float]
            Dictionary mapping feature names to importance scores
        """
        # If no task name provided, use the first task
        if task_name is None:
            if hasattr(self.model, 'task_manager') and len(self.model.task_manager.task_heads) > 0:
                task_name = self.model.task_manager.task_heads[0].name
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Handle both tensor and dictionary input formats
        if isinstance(X, dict):
            # For dictionary input, extract the continuous features
            X_tensor = X['continuous']
            continuous_input = X
        else:
            # For tensor input, create the dictionary format that the model expects
            X_tensor = X
            continuous_input = {'continuous': X}
            
        # Get baseline predictions
        with torch.no_grad():
            if sample_weights is not None:
                baseline_preds = self.model.predict(**continuous_input, sample_weights=sample_weights)
            else:
                baseline_preds = self.model.predict(**continuous_input)
        
        # Extract relevant metric from predictions
        if metric == 'risk_score':
            if task_name in baseline_preds['task_outputs']:
                baseline_metric = baseline_preds['task_outputs'][task_name]['risk_score'].cpu().numpy()
            else:
                # Try to find an available task
                for name, outputs in baseline_preds['task_outputs'].items():
                    if 'risk_score' in outputs:
                        baseline_metric = outputs['risk_score'].cpu().numpy()
                        task_name = name
                        break
        elif metric == 'c_index' and y is not None:
            # Compute c-index using survival targets
            event_indicator = y[:, 0].cpu().numpy()
            event_time = y[:, 1].cpu().numpy()
            risk_scores = baseline_preds['task_outputs'][task_name]['risk_score'].cpu().numpy()
            
            # Pass sample weights to c-index calculation if available
            if sample_weights is not None:
                weights_np = sample_weights.cpu().numpy()
                baseline_metric = self._compute_c_index(risk_scores, event_time, event_indicator, weights_np)
            else:
                baseline_metric = self._compute_c_index(risk_scores, event_time, event_indicator)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Get number of features
        n_features = X_tensor.shape[1]
        
        # Generate feature names if not provided
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Initialize importance scores
        raw_importance_scores = {name: 0.0 for name in feature_names}
        
        # Generate feature index groups for categorical variables
        cat_feature_groups = {}
        if cat_feature_info:
            # Group feature indices by original variable name
            feature_name_to_idx = {name: i for i, name in enumerate(feature_names)}
            
            for cat_info in cat_feature_info:
                orig_name = cat_info.get('original_name', cat_info['name'])
                
                # Find all feature indices that belong to this categorical variable
                embed_indices = []
                embed_prefix = f"{cat_info['name']}_embed_"
                
                for i, name in enumerate(feature_names):
                    if name.startswith(embed_prefix):
                        embed_indices.append(i)
                
                if embed_indices:
                    cat_feature_groups[orig_name] = embed_indices
        
        # For each feature (or feature group for categorical), permute its values and compute importance
        processed_indices = set()
        
        # First, handle categorical feature groups
        for cat_name, indices in cat_feature_groups.items():
            # Skip if we've already processed these indices
            if any(idx in processed_indices for idx in indices):
                continue
            
            # Mark these indices as processed
            processed_indices.update(indices)
            
            feature_importance = 0.0
            
            # Repeat permutation multiple times for stability
            for j in range(n_repeats):
                # Create a copy of the input tensor
                X_permuted = X_tensor.clone()
                
                # Permute all embedding dimensions together
                perm_idx = torch.randperm(X_tensor.shape[0])
                for idx in indices:
                    X_permuted[:, idx] = X_permuted[perm_idx, idx]
                
                # Create the input format that the model expects
                if isinstance(X, dict):
                    permuted_input = dict(X)  # Make a copy of the original dict
                    permuted_input['continuous'] = X_permuted
                else:
                    permuted_input = {'continuous': X_permuted}
                
                # Get predictions with permuted feature
                with torch.no_grad():
                    if sample_weights is not None:
                        permuted_preds = self.model.predict(**permuted_input, sample_weights=sample_weights)
                    else:
                        permuted_preds = self.model.predict(**permuted_input)
                
                # Extract metric from permuted predictions
                if metric == 'risk_score':
                    permuted_metric = permuted_preds['task_outputs'][task_name]['risk_score'].cpu().numpy()
                    # Compute mean absolute difference in risk scores
                    if sample_weights is not None:
                        weights_np = sample_weights.cpu().numpy()
                        importance = np.average(np.abs(baseline_metric - permuted_metric), weights=weights_np)
                    else:
                        importance = np.mean(np.abs(baseline_metric - permuted_metric))
                elif metric == 'c_index' and y is not None:
                    # Compute c-index using survival targets
                    risk_scores = permuted_preds['task_outputs'][task_name]['risk_score'].cpu().numpy()
                    if sample_weights is not None:
                        weights_np = sample_weights.cpu().numpy()
                        permuted_metric = self._compute_c_index(risk_scores, event_time, event_indicator, weights_np)
                    else:
                        permuted_metric = self._compute_c_index(risk_scores, event_time, event_indicator)
                    # Compute difference in c-index
                    importance = baseline_metric - permuted_metric
                
                feature_importance += importance
            
            # Average importance over repeats
            # Store under the original category name
            raw_importance_scores[cat_name] = feature_importance / n_repeats
            
            # Also assign to all embedding dimensions to make normalization work
            for idx in indices:
                raw_importance_scores[feature_names[idx]] = feature_importance / n_repeats
        
        # Handle remaining individual features
        for i in range(n_features):
            if i in processed_indices:
                continue
                
            feature_name = feature_names[i]
            feature_importance = 0.0
            
            # Repeat permutation multiple times for stability
            for j in range(n_repeats):
                # Create a copy of the input tensor
                X_permuted = X_tensor.clone()
                
                # Permute the feature
                perm_idx = torch.randperm(X_tensor.shape[0])
                X_permuted[:, i] = X_permuted[perm_idx, i]
                
                # Create the input format that the model expects
                if isinstance(X, dict):
                    permuted_input = dict(X)  # Make a copy of the original dict
                    permuted_input['continuous'] = X_permuted
                else:
                    permuted_input = {'continuous': X_permuted}
                
                # Get predictions with permuted feature
                with torch.no_grad():
                    if sample_weights is not None:
                        permuted_preds = self.model.predict(**permuted_input, sample_weights=sample_weights)
                    else:
                        permuted_preds = self.model.predict(**permuted_input)
                
                # Extract metric from permuted predictions
                if metric == 'risk_score':
                    permuted_metric = permuted_preds['task_outputs'][task_name]['risk_score'].cpu().numpy()
                    # Compute mean absolute difference in risk scores
                    if sample_weights is not None:
                        weights_np = sample_weights.cpu().numpy()
                        importance = np.average(np.abs(baseline_metric - permuted_metric), weights=weights_np)
                    else:
                        importance = np.mean(np.abs(baseline_metric - permuted_metric))
                elif metric == 'c_index' and y is not None:
                    # Compute c-index using survival targets
                    risk_scores = permuted_preds['task_outputs'][task_name]['risk_score'].cpu().numpy()
                    if sample_weights is not None:
                        weights_np = sample_weights.cpu().numpy()
                        permuted_metric = self._compute_c_index(risk_scores, event_time, event_indicator, weights_np)
                    else:
                        permuted_metric = self._compute_c_index(risk_scores, event_time, event_indicator)
                    # Compute difference in c-index
                    importance = baseline_metric - permuted_metric
                
                feature_importance += importance
            
            # Average importance over repeats
            raw_importance_scores[feature_name] = feature_importance / n_repeats
        
        # Normalize importance scores
        max_importance = max(raw_importance_scores.values())
        normalized_scores = {}
        
        if max_importance > 0:
            for name, value in raw_importance_scores.items():
                normalized_scores[name] = value / max_importance
        else:
            normalized_scores = raw_importance_scores.copy()
        
        # Pool embedding dimensions if requested and we don't have cat_feature_info
        if pool_embeddings and not cat_feature_info:
            normalized_scores = self._pool_embedding_importance(normalized_scores)
            
        # Filter to return only one importance score per feature/category
        final_scores = {}
        processed_names = set()
        
        # Add categorical features first
        for cat_name in cat_feature_groups.keys():
            if cat_name not in processed_names:
                final_scores[cat_name] = normalized_scores[cat_name]
                processed_names.add(cat_name)
        
        # Add remaining features
        for name, value in normalized_scores.items():
            # Skip embedding dimensions and already processed names
            if name in processed_names or any(name.startswith(f"{cat_info['name']}_embed_") for cat_info in (cat_feature_info or [])):
                continue
                
            final_scores[name] = value
            processed_names.add(name)
        
        return final_scores
        
    def _pool_embedding_importance(self, importance_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Pool importance scores for embedding dimensions of the same feature.
        
        Parameters
        ----------
        importance_dict : Dict[str, float]
            Dictionary mapping feature names to importance scores
            
        Returns
        -------
        Dict[str, float]
            Dictionary with pooled importance scores where embedding dimensions
            are combined into a single score per feature
        """
        # Create a new dictionary to store pooled scores
        pooled_dict = {}
        
        # Track features that we've already processed
        processed_features = set()
        
        # First, identify embedding features by looking for "_embed_" in the name
        for name, value in importance_dict.items():
            if "_embed_" in name:
                # Extract the base feature name (before the _embed_ suffix)
                base_feature = name.split("_embed_")[0]
                
                # Skip if we've already processed this base feature
                if base_feature in processed_features:
                    continue
                
                # Find all embedding dimensions for this feature
                embed_values = []
                for embed_name, embed_value in importance_dict.items():
                    if embed_name.startswith(f"{base_feature}_embed_"):
                        embed_values.append(embed_value)
                
                # Pool the values (use sum as default pooling strategy)
                if embed_values:
                    pooled_dict[base_feature] = sum(embed_values)
                    processed_features.add(base_feature)
            elif name not in processed_features and "_embed_" not in name:
                # For non-embedding features, just copy the value
                pooled_dict[name] = value
        
        return pooled_dict
    
    def plot_importance(
        self,
        importance_scores: Dict[str, float],
        top_k: Optional[int] = None,
        figsize: Tuple[float, float] = (10, 6),
        title: str = 'Feature Importance (Permutation Method)',
        color: str = 'skyblue'
    ) -> plt.Figure:
        """
        Plot feature importance scores.
        
        Parameters
        ----------
        importance_scores : Dict[str, float]
            Dictionary mapping feature names to importance scores
            
        top_k : int, optional
            Number of top features to show, default is all
            
        figsize : Tuple[float, float], default=(10, 6)
            Figure size
            
        title : str, default='Feature Importance (Permutation Method)'
            Plot title
            
        color : str, default='skyblue'
            Bar color
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        # Sort importance scores
        sorted_importance = sorted(
            importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Limit to top k features if specified
        if top_k is not None:
            sorted_importance = sorted_importance[:top_k]
        
        # Extract names and scores
        feature_names = [x[0] for x in sorted_importance]
        scores = [x[1] for x in sorted_importance]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, scores, color=color)
        
        # Set labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Importance Score')
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def _compute_c_index(
        self,
        risk_scores: np.ndarray,
        event_time: np.ndarray,
        event_indicator: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute concordance index for survival predictions.
        
        Parameters
        ----------
        risk_scores : np.ndarray
            Predicted risk scores
            
        event_time : np.ndarray
            Event times
            
        event_indicator : np.ndarray
            Event indicators (1 if event occurred, 0 if censored)
            
        sample_weights : np.ndarray, optional
            Sample weights for weighted c-index calculation
            
        Returns
        -------
        float
            Concordance index
        """
        n_samples = len(risk_scores)
        
        # Initialize counters
        concordant = 0
        discordant = 0
        tied_risk = 0
        
        # Count comparable pairs
        comparable_pairs = 0
        
        # Use uniform weights if none provided
        if sample_weights is None:
            sample_weights = np.ones(n_samples)
            
        # Compute pairwise comparisons
        for i in range(n_samples):
            if event_indicator[i] == 0:
                # Skip censored samples for first position
                continue
                
            for j in range(n_samples):
                # Valid comparison if:
                # 1. i had an event and j was censored after i's event, or
                # 2. Both i and j had events, and i's event was before j's
                if (event_indicator[i] == 1 and 
                    ((event_indicator[j] == 0 and event_time[j] > event_time[i]) or
                     (event_indicator[j] == 1 and event_time[i] < event_time[j]))):
                    
                    # Calculate pair weight as product of individual weights
                    pair_weight = sample_weights[i] * sample_weights[j]
                    
                    # i should have a higher risk score than j
                    if risk_scores[i] > risk_scores[j]:
                        concordant += pair_weight
                    elif risk_scores[i] < risk_scores[j]:
                        discordant += pair_weight
                    else:
                        tied_risk += pair_weight
                    
                    comparable_pairs += pair_weight
        
        # Compute concordance index
        if comparable_pairs > 0:
            return (concordant + 0.5 * tied_risk) / comparable_pairs
        else:
            return 0.5  # Default value when no comparable pairs exist


class ShapImportance:
    """
    Calculate feature importance using SHAP (SHapley Additive exPlanations).
    
    This method approximates Shapley values, which represent the contribution
    of each feature to the prediction for a specific instance.
    """
    
    def __init__(self, model):
        """
        Initialize SHAP importance calculator.
        
        Parameters
        ----------
        model : EnhancedDeepHit
            The trained model for which to compute importance
        """
        self.model = model
    
    def compute_importance(
        self,
        X: Union[torch.Tensor, Dict[str, torch.Tensor]],
        n_samples: int = 100,
        background_samples: Optional[torch.Tensor] = None,
        feature_names: Optional[List[str]] = None,
        task_name: Optional[str] = None,
        target_output: str = 'risk_score',
        pool_embeddings: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Compute SHAP values for feature importance.
        
        Parameters
        ----------
        X : torch.Tensor
            Input features [n_samples, n_features]
            
        n_samples : int, default=100
            Number of background samples for approximating expectation
            
        background_samples : torch.Tensor, optional
            Specific background samples to use, default is random samples from X
            
        feature_names : List[str], optional
            Names of features for labeling results
            
        task_name : str, optional
            Name of the task to compute importance for (if multi-task model)
            
        target_output : str, default='risk_score'
            Target output to explain ('risk_score', 'survival', 'hazard')
            
        pool_embeddings : bool, default=True
            Whether to pool importance scores for embedding dimensions of the same feature
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping feature names to SHAP values [n_samples]
        """
        # If no task name provided, use the first task
        if task_name is None:
            if hasattr(self.model, 'task_manager') and len(self.model.task_manager.task_heads) > 0:
                task_name = self.model.task_manager.task_heads[0].name
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Handle both tensor and dictionary input formats
        if isinstance(X, dict):
            # For dictionary input, extract the continuous features
            X_tensor = X['continuous']
            continuous_input = X
        else:
            # For tensor input, create the dictionary format that the model expects
            X_tensor = X
            continuous_input = {'continuous': X}
        
        # Generate feature names if not provided
        n_features = X_tensor.shape[1]
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Create background samples if not provided
        if background_samples is None:
            # Sample randomly from the input data
            indices = torch.randperm(X_tensor.shape[0])[:n_samples]
            background_samples = X_tensor[indices]
            
        # Create background input in the format the model expects
        background_input = {'continuous': background_samples}
        
        # Compute background value (expected prediction)
        with torch.no_grad():
            background_preds = self.model.predict(**background_input)
            
        if target_output == 'risk_score':
            background_value = background_preds['task_outputs'][task_name]['risk_score'].mean().item()
        elif target_output == 'survival':
            # Use mean survival at time bin 0 as background value
            background_value = background_preds['task_outputs'][task_name]['survival'][:, 0].mean().item()
        elif target_output == 'hazard':
            # Use mean hazard at time bin 0 as background value
            background_value = background_preds['task_outputs'][task_name]['hazard'][:, 0].mean().item()
        
        # Get prediction function for the specified output
        def predict_fn(x_tensor):
            with torch.no_grad():
                input_dict = {'continuous': x_tensor}
                preds = self.model.predict(**input_dict)
                if target_output == 'risk_score':
                    return preds['task_outputs'][task_name]['risk_score'].cpu().numpy()
                elif target_output == 'survival':
                    return preds['task_outputs'][task_name]['survival'][:, 0].cpu().numpy()
                elif target_output == 'hazard':
                    return preds['task_outputs'][task_name]['hazard'][:, 0].cpu().numpy()
        
        # Initialize SHAP values
        shap_values = np.zeros((X_tensor.shape[0], n_features))
        
        # Compute SHAP values for each instance
        for i in tqdm(range(X_tensor.shape[0]), desc="Computing SHAP values"):
            x_instance = X_tensor[i].unsqueeze(0)
            
            # Calculate instance prediction
            instance_pred = predict_fn(x_instance)[0]
            
            # Compute Shapley values using permutation sampling
            instance_shap = np.zeros(n_features)
            
            # Sample permutations to approximate Shapley values
            n_iterations = min(100, 2 ** n_features)  # Limit iterations for large feature sets
            
            for _ in range(n_iterations):
                # Generate random permutation of features
                perm = torch.randperm(n_features)
                
                # Initialize predictions
                pred_with = background_value
                pred_without = background_value
                
                # Create masked tensors
                x_with = background_samples[0].clone().unsqueeze(0)
                x_without = background_samples[0].clone().unsqueeze(0)
                
                # Iterate through features in the permutation
                for j in range(n_features):
                    feature_idx = perm[j].item()
                    
                    # Update "with" instance to include the next feature
                    x_with[:, feature_idx] = x_instance[:, feature_idx]
                    
                    # Get predictions
                    pred_with_feature = predict_fn(x_with)[0]
                    pred_without_feature = pred_without
                    
                    # Update "without" instance for next iteration
                    x_without[:, feature_idx] = x_instance[:, feature_idx]
                    pred_without = pred_with_feature
                    
                    # Add marginal contribution to Shapley value
                    instance_shap[feature_idx] += (pred_with_feature - pred_without_feature) / n_iterations
            
            # Store SHAP values for this instance
            shap_values[i] = instance_shap
        
        # Create dictionary mapping feature names to SHAP values
        shap_dict = {name: shap_values[:, i] for i, name in enumerate(feature_names)}
        
        # Pool embedding dimensions if requested
        if pool_embeddings:
            shap_dict = self._pool_embedding_importance(shap_dict)
        
        return shap_dict
        
    def _pool_embedding_importance(self, importance_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Pool importance scores for embedding dimensions of the same feature.
        
        Parameters
        ----------
        importance_dict : Dict[str, np.ndarray]
            Dictionary mapping feature names to importance arrays
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with pooled importance scores where embedding dimensions
            are combined into a single score per feature
        """
        # Create a new dictionary to store pooled scores
        pooled_dict = {}
        
        # Track features that we've already processed
        processed_features = set()
        
        # First, identify embedding features by looking for "_embed_" in the name
        for name, values in importance_dict.items():
            if "_embed_" in name:
                # Extract the base feature name (before the _embed_ suffix)
                base_feature = name.split("_embed_")[0]
                
                # Skip if we've already processed this base feature
                if base_feature in processed_features:
                    continue
                
                # Find all embedding dimensions for this feature
                embed_values = []
                embed_names = []
                for embed_name, embed_value in importance_dict.items():
                    if embed_name.startswith(f"{base_feature}_embed_"):
                        embed_values.append(embed_value)
                        embed_names.append(embed_name)
                
                # Pool the values by summing across embedding dimensions
                if embed_values:
                    # Sum arrays element-wise
                    pooled_values = np.zeros_like(embed_values[0])
                    for v in embed_values:
                        pooled_values += v
                    
                    pooled_dict[base_feature] = pooled_values
                    processed_features.add(base_feature)
                    
                    # Add the names to processed to avoid duplicates
                    for embed_name in embed_names:
                        processed_features.add(embed_name)
            elif name not in processed_features:
                # For non-embedding features, just copy the values
                pooled_dict[name] = importance_dict[name]
        
        return pooled_dict
    
    def plot_importance(
        self,
        shap_values: Dict[str, np.ndarray],
        max_display: int = 10,
        figsize: Tuple[float, float] = (10, 6),
        title: str = 'Feature Importance (SHAP Values)',
        plot_type: str = 'bar'
    ) -> plt.Figure:
        """
        Plot SHAP values for feature importance.
        
        Parameters
        ----------
        shap_values : Dict[str, np.ndarray]
            Dictionary mapping feature names to SHAP values
            
        max_display : int, default=10
            Maximum number of features to display
            
        figsize : Tuple[float, float], default=(10, 6)
            Figure size
            
        title : str, default='Feature Importance (SHAP Values)'
            Plot title
            
        plot_type : str, default='bar'
            Type of plot ('bar' or 'beeswarm')
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = {name: np.abs(values).mean() for name, values in shap_values.items()}
        
        # Sort features by importance
        sorted_features = sorted(
            mean_abs_shap.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Limit to max_display features
        sorted_features = sorted_features[:max_display]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        if plot_type == 'bar':
            # Extract feature names and mean SHAP values
            feature_names = [f[0] for f in sorted_features]
            mean_values = [f[1] for f in sorted_features]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(feature_names))
            ax.barh(y_pos, mean_values, color='skyblue')
            
            # Set labels and title
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_title(title)
        
        elif plot_type == 'beeswarm':
            # Create dataframe for SHAP values
            plot_data = []
            
            for name, _ in sorted_features:
                values = shap_values[name]
                for v in values:
                    plot_data.append((name, v))
            
            df = pd.DataFrame(plot_data, columns=['Feature', 'SHAP Value'])
            
            # Plot beeswarm using seaborn if available
            try:
                import seaborn as sns
                sns.stripplot(
                    x='SHAP Value',
                    y='Feature',
                    data=df,
                    alpha=0.4,
                    jitter=True,
                    ax=ax
                )
            except ImportError:
                # Fallback to scatter plot
                for i, (name, _) in enumerate(sorted_features):
                    ax.scatter(shap_values[name], [i] * len(shap_values[name]), alpha=0.4)
                
                # Set labels
                ax.set_yticks(np.arange(len(sorted_features)))
                ax.set_yticklabels([f[0] for f in sorted_features])
                ax.set_xlabel('SHAP Value')
            
            ax.set_title(title)
        
        plt.tight_layout()
        return fig


class IntegratedGradients:
    """
    Calculate feature importance using integrated gradients.
    
    This method computes importance by integrating the gradients of the
    model's output with respect to the inputs along a straight-line path
    from a baseline to the input.
    """
    
    def __init__(self, model):
        """
        Initialize integrated gradients calculator.
        
        Parameters
        ----------
        model : EnhancedDeepHit
            The trained model for which to compute importance
        """
        self.model = model
    
    def compute_importance(
        self,
        X: Union[torch.Tensor, Dict[str, torch.Tensor]],
        target_class: str = 'risk_score',
        baseline: Optional[torch.Tensor] = None,
        feature_names: Optional[List[str]] = None,
        task_name: Optional[str] = None,
        n_steps: int = 50,
        pool_embeddings: bool = True
    ) -> Dict[str, float]:
        """
        Compute feature importance using integrated gradients.
        
        Parameters
        ----------
        X : torch.Tensor
            Input features [n_samples, n_features]
            
        target_class : str, default='risk_score'
            Target output for gradient calculation 
            ('risk_score', 'survival', 'hazard')
            
        baseline : torch.Tensor, optional
            Baseline input for integrated gradients (default: zeros)
            
        feature_names : List[str], optional
            Names of features for labeling results
            
        task_name : str, optional
            Name of the task to compute importance for (if multi-task model)
            
        n_steps : int, default=50
            Number of steps for numerical integration
            
        pool_embeddings : bool, default=True
            Whether to pool importance scores for embedding dimensions of the same feature
            
        Returns
        -------
        Dict[str, float]
            Dictionary mapping feature names to importance values
        """
        # If no task name provided, use the first task
        if task_name is None:
            if hasattr(self.model, 'task_manager') and len(self.model.task_manager.task_heads) > 0:
                task_name = self.model.task_manager.task_heads[0].name
                
        # Handle both tensor and dictionary input formats
        if isinstance(X, dict):
            # For dictionary input, extract the continuous features
            X_tensor = X['continuous']
            continuous_input = X
        else:
            # For tensor input, create the dictionary format that the model expects
            X_tensor = X
            continuous_input = {'continuous': X}
        
        # Generate feature names if not provided
        n_features = X_tensor.shape[1]
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Set baseline if not provided
        if baseline is None:
            baseline = torch.zeros_like(X_tensor)
        
        # Ensure input requires gradients
        X_input = X_tensor.clone().detach().requires_grad_(True)
        
        # Set model to eval mode but enable gradients
        self.model.eval()
        
        # Define function to get output value
        def get_output(input_tensor):
            # Create the input format that the model expects
            model_input = {'continuous': input_tensor}
            
            predictions = self.model.predict(**model_input)
            
            if target_class == 'risk_score':
                return predictions['task_outputs'][task_name]['risk_score']
            elif target_class == 'survival':
                # Use survival at time bin 0
                return predictions['task_outputs'][task_name]['survival'][:, 0]
            elif target_class == 'hazard':
                # Use hazard at time bin 0
                return predictions['task_outputs'][task_name]['hazard'][:, 0]
            else:
                raise ValueError(f"Unsupported target class: {target_class}")
        
        # For integrated gradients, we need to modify our approach since we're going 
        # through multiple layers of the model, including the encoder
        # Since we've passed X_input through several functions, let's simplify
        
        # Instead of computing gradients directly with the get_output function,
        # we'll use a simpler approximation approach for integrated gradients
        
        # Compute the baseline prediction
        baseline_pred = get_output(baseline)
        
        # Compute the input prediction
        input_pred = get_output(X_input)
        
        # Compute difference to be approximated
        pred_diff = input_pred - baseline_pred
        
        # We'll directly compute attributions using a sampling-based approach
        attributions = torch.zeros_like(X_input)
        
        # Sample points along the path and compute finite differences
        for step in range(1, n_steps + 1):
            # Compute intermediate point along the path
            alpha = step / n_steps
            interpolated_input = baseline + alpha * (X_input - baseline)
            
            # Get prediction at this point
            interp_pred = get_output(interpolated_input)
            
            # Compute contribution at this step (simplified approximation)
            if step > 1:
                prev_alpha = (step - 1) / n_steps
                prev_interpolated = baseline + prev_alpha * (X_input - baseline)
                prev_pred = get_output(prev_interpolated)
                
                # Compute difference in predictions
                pred_step_diff = interp_pred - prev_pred
                
                # Compute input difference at this step
                input_diff = interpolated_input - prev_interpolated
                
                # Approximate gradients by finite differences
                attributions += (X_input - baseline) * pred_step_diff.unsqueeze(1) / input_diff
        
        # Average over steps
        attributions = attributions / n_steps
        
        # Sum attributions across batch dimension to get overall importance
        importance_values = attributions.abs().mean(dim=0).detach().numpy()
        
        # Create dictionary of feature importance
        # Check if number of features match and handle mismatch
        if len(feature_names) > len(importance_values):
            # More feature names than importance values
            # Use only as many names as there are values
            truncated_names = feature_names[:len(importance_values)]
            importance_dict = {name: importance_values[i] for i, name in enumerate(truncated_names)}
        elif len(feature_names) < len(importance_values):
            # More importance values than feature names
            # Use only as many values as there are names
            importance_dict = {name: importance_values[i] for i, name in enumerate(feature_names)}
        else:
            # Sizes match
            importance_dict = {name: importance_values[i] for i, name in enumerate(feature_names)}
        
        # Pool embedding dimensions if requested
        if pool_embeddings:
            importance_dict = self._pool_embedding_importance(importance_dict)
        
        # Normalize importance values
        max_importance = max(importance_dict.values())
        if max_importance > 0:
            for name in importance_dict:
                importance_dict[name] /= max_importance
        
        return importance_dict
        
    def _pool_embedding_importance(self, importance_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Pool importance scores for embedding dimensions of the same feature.
        
        Parameters
        ----------
        importance_dict : Dict[str, float]
            Dictionary mapping feature names to importance values
            
        Returns
        -------
        Dict[str, float]
            Dictionary with pooled importance scores where embedding dimensions
            are combined into a single score per feature
        """
        # Create a new dictionary to store pooled scores
        pooled_dict = {}
        
        # Track features that we've already processed
        processed_features = set()
        
        # First, identify embedding features by looking for "_embed_" in the name
        for name, value in importance_dict.items():
            if "_embed_" in name:
                # Extract the base feature name (before the _embed_ suffix)
                base_feature = name.split("_embed_")[0]
                
                # Skip if we've already processed this base feature
                if base_feature in processed_features:
                    continue
                
                # Find all embedding dimensions for this feature
                embed_values = []
                for embed_name, embed_value in importance_dict.items():
                    if embed_name.startswith(f"{base_feature}_embed_"):
                        embed_values.append(embed_value)
                
                # Pool the values (use sum as default pooling strategy)
                if embed_values:
                    pooled_dict[base_feature] = sum(embed_values)
                    processed_features.add(base_feature)
            elif name not in processed_features and "_embed_" not in name:
                # For non-embedding features, just copy the value
                pooled_dict[name] = value
        
        return pooled_dict
    
    def plot_importance(
        self,
        importance_values: Dict[str, float],
        top_k: Optional[int] = None,
        figsize: Tuple[float, float] = (10, 6),
        title: str = 'Feature Importance (Integrated Gradients)',
        color: str = 'limegreen'
    ) -> plt.Figure:
        """
        Plot feature importance from integrated gradients.
        
        Parameters
        ----------
        importance_values : Dict[str, float]
            Dictionary mapping feature names to importance values
            
        top_k : int, optional
            Number of top features to show, default is all
            
        figsize : Tuple[float, float], default=(10, 6)
            Figure size
            
        title : str, default='Feature Importance (Integrated Gradients)'
            Plot title
            
        color : str, default='limegreen'
            Bar color
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        # Sort importance values
        sorted_importance = sorted(
            importance_values.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Limit to top k features if specified
        if top_k is not None:
            sorted_importance = sorted_importance[:top_k]
        
        # Extract names and values
        feature_names = [x[0] for x in sorted_importance]
        values = [x[1] for x in sorted_importance]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, values, color=color)
        
        # Set labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Importance')
        ax.set_title(title)
        
        plt.tight_layout()
        return fig


class AttentionImportance:
    """
    Calculate feature importance using attention weights.
    
    This method uses the attention weights from the transformer layers
    to derive feature importance scores.
    """
    
    def __init__(self, model):
        """
        Initialize attention-based importance calculator.
        
        Parameters
        ----------
        model : EnhancedDeepHit
            The trained model for which to compute importance
        """
        self.model = model
    
    def compute_importance(
        self,
        X: Union[torch.Tensor, Dict[str, torch.Tensor]],
        feature_names: Optional[List[str]] = None,
        layer_idx: int = -1,
        head_idx: Optional[int] = None,
        aggregation: str = 'mean',
        pool_embeddings: bool = True
    ) -> Dict[str, float]:
        """
        Compute feature importance from attention weights.
        
        Parameters
        ----------
        X : torch.Tensor
            Input features [n_samples, n_features]
            
        feature_names : List[str], optional
            Names of features for labeling results
            
        layer_idx : int, default=-1
            Index of transformer layer to use for importance (-1 for last layer)
            
        head_idx : int, optional
            Index of attention head to use (None for all heads)
            
        aggregation : str, default='mean'
            Method to aggregate attention weights ('mean', 'max', 'sum')
            
        pool_embeddings : bool, default=True
            Whether to pool importance scores for embedding dimensions of the same feature
            
        Returns
        -------
        Dict[str, float]
            Dictionary mapping feature names to importance scores
        """
        # Handle both tensor and dictionary input formats
        if isinstance(X, dict):
            # For dictionary input, extract the continuous features
            X_tensor = X['continuous']
            continuous_input = X
        else:
            # For tensor input, create the dictionary format that the model expects
            X_tensor = X
            continuous_input = {'continuous': X_tensor}
            
        # Generate feature names if not provided
        n_features = X_tensor.shape[1]
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Get attention maps
        with torch.no_grad():
            _, attention_maps = self.model.encoder(**continuous_input)
        
        # Get attention map from specified layer
        if layer_idx < 0:
            layer_idx = len(attention_maps) + layer_idx
        
        if layer_idx < 0 or layer_idx >= len(attention_maps):
            raise ValueError(f"Invalid layer index {layer_idx}, model has {len(attention_maps)} layers")
        
        attention = attention_maps[layer_idx]  # [batch_size, n_heads, seq_len, seq_len]
        
        # Extract relevant attention weights
        if head_idx is not None:
            attention = attention[:, head_idx, :, :]  # [batch_size, seq_len, seq_len]
        else:
            # Average across heads
            attention = attention.mean(dim=1)  # [batch_size, seq_len, seq_len]
        
        # Compute feature importance by aggregating attention
        # We're interested in how much attention each feature receives
        if self.model.encoder.pool == 'cls':
            # For CLS token, use attention from CLS to each feature
            feature_attention = attention[:, 0, 1:n_features+1]  # [batch_size, n_features]
        else:
            # Otherwise, sum attention to each feature from all other features
            feature_attention = attention.sum(dim=1)[:, :n_features]  # [batch_size, n_features]
        
        # Aggregate across batch dimension
        if aggregation == 'mean':
            importance_values = feature_attention.mean(dim=0).cpu().numpy()
        elif aggregation == 'max':
            importance_values = feature_attention.max(dim=0)[0].cpu().numpy()
        elif aggregation == 'sum':
            importance_values = feature_attention.sum(dim=0).cpu().numpy()
        else:
            raise ValueError(f"Invalid aggregation method: {aggregation}")
        
        # Create dictionary of feature importance
        # Check if number of features match and handle mismatch
        if len(feature_names) > len(importance_values):
            # More feature names than importance values
            # Use only as many names as there are values
            truncated_names = feature_names[:len(importance_values)]
            importance_dict = {name: importance_values[i] for i, name in enumerate(truncated_names)}
        elif len(feature_names) < len(importance_values):
            # More importance values than feature names
            # Use only as many values as there are names
            importance_dict = {name: importance_values[i] for i, name in enumerate(feature_names)}
        else:
            # Sizes match
            importance_dict = {name: importance_values[i] for i, name in enumerate(feature_names)}
        
        # Pool embedding dimensions if requested
        if pool_embeddings:
            importance_dict = self._pool_embedding_importance(importance_dict)
        
        # Normalize importance values
        total_importance = sum(importance_dict.values())
        if total_importance > 0:
            for name in importance_dict:
                importance_dict[name] /= total_importance
        
        return importance_dict
        
    def _pool_embedding_importance(self, importance_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Pool importance scores for embedding dimensions of the same feature.
        
        Parameters
        ----------
        importance_dict : Dict[str, float]
            Dictionary mapping feature names to importance scores
            
        Returns
        -------
        Dict[str, float]
            Dictionary with pooled importance scores where embedding dimensions
            are combined into a single score per feature
        """
        # Create a new dictionary to store pooled scores
        pooled_dict = {}
        
        # Track features that we've already processed
        processed_features = set()
        
        # First, identify embedding features by looking for "_embed_" in the name
        for name, value in importance_dict.items():
            if "_embed_" in name:
                # Extract the base feature name (before the _embed_ suffix)
                base_feature = name.split("_embed_")[0]
                
                # Skip if we've already processed this base feature
                if base_feature in processed_features:
                    continue
                
                # Find all embedding dimensions for this feature
                embed_values = []
                for embed_name, embed_value in importance_dict.items():
                    if embed_name.startswith(f"{base_feature}_embed_"):
                        embed_values.append(embed_value)
                
                # Pool the values (use sum as default pooling strategy)
                if embed_values:
                    pooled_dict[base_feature] = sum(embed_values)
                    processed_features.add(base_feature)
            elif name not in processed_features and "_embed_" not in name:
                # For non-embedding features, just copy the value
                pooled_dict[name] = value
        
        return pooled_dict
    
    def get_attention_map(
        self,
        X: Union[torch.Tensor, Dict[str, torch.Tensor]],
        layer_idx: int = -1,
        head_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Get attention map for visualization.
        
        Parameters
        ----------
        X : torch.Tensor
            Input features [n_samples, n_features]
            
        layer_idx : int, default=-1
            Index of transformer layer to use (-1 for last layer)
            
        head_idx : int, optional
            Index of attention head to use (None for all heads average)
            
        Returns
        -------
        torch.Tensor
            Attention map [n_samples, seq_len, seq_len]
        """
        # Handle both tensor and dictionary input formats
        if isinstance(X, dict):
            # For dictionary input, extract the continuous features
            X_tensor = X['continuous']
            continuous_input = X
        else:
            # For tensor input, create the dictionary format that the model expects
            X_tensor = X
            continuous_input = {'continuous': X_tensor}
            
        # Set model to evaluation mode
        self.model.eval()
        
        # Get attention maps
        with torch.no_grad():
            _, attention_maps = self.model.encoder(**continuous_input)
        
        # Get attention map from specified layer
        if layer_idx < 0:
            layer_idx = len(attention_maps) + layer_idx
        
        if layer_idx < 0 or layer_idx >= len(attention_maps):
            raise ValueError(f"Invalid layer index {layer_idx}, model has {len(attention_maps)} layers")
        
        attention = attention_maps[layer_idx]  # [batch_size, n_heads, seq_len, seq_len]
        
        # Extract relevant attention weights
        if head_idx is not None:
            attention = attention[:, head_idx, :, :]  # [batch_size, seq_len, seq_len]
        else:
            # Average across heads
            attention = attention.mean(dim=1)  # [batch_size, seq_len, seq_len]
        
        return attention
    
    def plot_importance(
        self,
        importance_values: Dict[str, float],
        top_k: Optional[int] = None,
        figsize: Tuple[float, float] = (10, 6),
        title: str = 'Feature Importance (Attention Weights)',
        color: str = 'purple'
    ) -> plt.Figure:
        """
        Plot feature importance from attention weights.
        
        Parameters
        ----------
        importance_values : Dict[str, float]
            Dictionary mapping feature names to importance values
            
        top_k : int, optional
            Number of top features to show, default is all
            
        figsize : Tuple[float, float], default=(10, 6)
            Figure size
            
        title : str, default='Feature Importance (Attention Weights)'
            Plot title
            
        color : str, default='purple'
            Bar color
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        # Sort importance values
        sorted_importance = sorted(
            importance_values.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Limit to top k features if specified
        if top_k is not None:
            sorted_importance = sorted_importance[:top_k]
        
        # Extract names and values
        feature_names = [x[0] for x in sorted_importance]
        values = [x[1] for x in sorted_importance]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, values, color=color)
        
        # Set labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Attention Score')
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def plot_attention_map(
        self,
        attention_map: torch.Tensor,
        feature_names: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (10, 8),
        sample_idx: int = 0,
        title: str = 'Attention Map'
    ) -> plt.Figure:
        """
        Plot attention map as a heatmap.
        
        Parameters
        ----------
        attention_map : torch.Tensor
            Attention map [n_samples, seq_len, seq_len]
            
        feature_names : List[str], optional
            Names of features for axis labels
            
        figsize : Tuple[float, float], default=(10, 8)
            Figure size
            
        sample_idx : int, default=0
            Index of sample to plot
            
        title : str, default='Attention Map'
            Plot title
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        # Extract attention map for the specified sample
        attn = attention_map[sample_idx].cpu().numpy()
        
        # Generate feature names if not provided
        n_features = attn.shape[0]
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(n_features)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        im = ax.imshow(attn, cmap='viridis')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Attention Weight', rotation=-90, va='bottom')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(n_features))
        ax.set_yticks(np.arange(n_features))
        ax.set_xticklabels(feature_names)
        ax.set_yticklabels(feature_names)
        
        # Rotate tick labels and set alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        # Add title
        ax.set_title(title)
        
        # Ensure everything fits
        fig.tight_layout()
        
        return fig