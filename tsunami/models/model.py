import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import os
import json
import logging
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler

from tsunami.models.encoder import TabularTransformer
from tsunami.models.tasks.base import MultiTaskManager, TaskHead
from tsunami.data.processing import DataProcessor, build_category_info


class EnhancedDeepHit(nn.Module):
    """
    Enhanced DeepHit model for multi-task tabular survival analysis.
    
    This model extends the original DeepHit approach to support:
    - Multiple target types (survival, competing risks, classification, regression)
    - Tabular transformer architecture with categorical and continuous features
    - Missing data handling
    - Masked loss for incomplete targets
    - Variational methods for uncertainty quantification (optional)

    Attributes
    ----------
    encoder : TabularTransformer
        The transformer-based encoder for processing input features.
    task_manager : MultiTaskManager
        Manages the different prediction heads (tasks).
    num_continuous : int
        Number of continuous input features.
    cat_feat_info : List[Dict]
        Information about categorical features (name, cardinality).
    encoder_dim : int
        Dimensionality of the encoder's output representation.
    include_variational : bool
        Flag indicating if variational inference components are included.
    device : str
        The device ('cpu' or 'cuda') the model is running on.
    """
    
    def __init__(self,
                num_continuous: int,
                targets: List[TaskHead],
                cat_feat_info: Optional[List[Dict]] = None,
                encoder_dim: int = 128,
                encoder_depth: int = 4,
                encoder_heads: int = 8,
                encoder_ff_dim: int = 512,
                encoder_dropout: float = 0.1,
                encoder_feature_interaction: bool = True,
                include_variational: bool = False,
                variational_beta: float = 0.1,
                device: str = 'cpu'):
        """
        Initialize the EnhancedDeepHit model.

        Configures the encoder architecture and the multi-task prediction heads.

        Parameters
        ----------
        num_continuous : int
            Number of continuous input features. Must be non-negative.
        targets : List[TaskHead]
            A list containing initialized instances of task-specific heads
            (e.g., SingleRiskHead, ClassificationHead). At least one head must be provided.
        cat_feat_info : List[Dict], optional
            A list where each dictionary describes a categorical feature.
            Expected keys: 'name' (str) and 'cardinality' (int). Defaults to None (no categorical features).
            Example: [{'name': 'gender', 'cardinality': 2}, {'name': 'treatment', 'cardinality': 3}]
        encoder_dim : int, default=128
            The dimensionality of the embeddings and hidden layers within the transformer encoder. Must be positive.
        encoder_depth : int, default=4
            The number of transformer blocks (layers) in the encoder. Must be positive.
        encoder_heads : int, default=8
            The number of attention heads in the multi-head self-attention mechanism of the encoder. Must be positive.
        encoder_ff_dim : int, default=512
            The dimensionality of the feed-forward network within each transformer block. Must be positive.
        encoder_dropout : float, default=0.1
            The dropout rate applied within the encoder (attention, feed-forward, embeddings). Must be between 0.0 and 1.0.
        encoder_feature_interaction : bool, default=True
            If True, enables an explicit feature interaction layer within the encoder.
        include_variational : bool, default=False
            If True, incorporates variational components into the task manager for uncertainty estimation.
        variational_beta : float, default=0.1
            The weight applied to the KL divergence term in the loss when `include_variational` is True. Must be non-negative.
        device : str, default='cpu'
            The device ('cpu' or 'cuda') on which the model's parameters and computations should be placed.
        """
        super().__init__()

        # --- Input Validation ---
        assert isinstance(num_continuous, int) and num_continuous >= 0, "'num_continuous' must be a non-negative integer"
        assert isinstance(targets, list) and all(isinstance(t, TaskHead) for t in targets), "'targets' must be a list of TaskHead instances"
        assert len(targets) > 0, "At least one target TaskHead must be provided"
        if cat_feat_info is not None:
            assert isinstance(cat_feat_info, list) and all(isinstance(info, dict) for info in cat_feat_info), "'cat_feat_info' must be a list of dictionaries"
        assert isinstance(encoder_dim, int) and encoder_dim > 0, "'encoder_dim' must be a positive integer"
        assert isinstance(encoder_depth, int) and encoder_depth > 0, "'encoder_depth' must be a positive integer"
        assert isinstance(encoder_heads, int) and encoder_heads > 0, "'encoder_heads' must be a positive integer"
        assert isinstance(encoder_ff_dim, int) and encoder_ff_dim > 0, "'encoder_ff_dim' must be a positive integer"
        assert isinstance(encoder_dropout, float) and 0.0 <= encoder_dropout < 1.0, "'encoder_dropout' must be a float between 0.0 and 1.0"
        assert isinstance(encoder_feature_interaction, bool), "'encoder_feature_interaction' must be a boolean"
        assert isinstance(include_variational, bool), "'include_variational' must be a boolean"
        assert isinstance(variational_beta, float) and variational_beta >= 0.0, "'variational_beta' must be a non-negative float"
        assert device in ['cpu', 'cuda'], "'device' must be either 'cpu' or 'cuda'"
        # --- End Input Validation ---
        
        self.num_continuous = num_continuous
        self.cat_feat_info = cat_feat_info or []
        self.encoder_dim = encoder_dim
        self.include_variational = include_variational
        self.device = device
        
        # Initialize encoder
        self.encoder = TabularTransformer(
            num_continuous=num_continuous,
            cat_feat_info=cat_feat_info,
            dim=encoder_dim,
            depth=encoder_depth,
            heads=encoder_heads,
            ff_dim=encoder_ff_dim,
            attn_dropout=encoder_dropout,
            ff_dropout=encoder_dropout,
            embedding_dropout=encoder_dropout,
            feature_interaction=encoder_feature_interaction,
            missing_value_embed=True,
            pool='attention'
        )
        
        # Initialize multi-task manager
        self.task_manager = MultiTaskManager(
            encoder_dim=encoder_dim,
            task_heads=targets,
            include_variational=include_variational,
            variational_beta=variational_beta
        )
        
        # Move to device
        self.to(device)
    
    def forward(self, 
               continuous: torch.Tensor,
               targets: Optional[Dict[str, torch.Tensor]] = None,
               masks: Optional[Dict[str, torch.Tensor]] = None,
               categorical: Optional[torch.Tensor] = None,
               missing_mask: Optional[torch.Tensor] = None,
               sample_weights: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Perform a forward pass through the encoder and task heads.

        Processes input features through the TabularTransformer encoder and then
        feeds the resulting representation to the MultiTaskManager to compute
        task-specific outputs and, if targets are provided, the combined loss.

        Parameters
        ----------
        continuous : torch.Tensor
            A 2D tensor containing the continuous features for the batch.
            Shape: `(batch_size, num_continuous)`.
        targets : Dict[str, torch.Tensor], optional
            A dictionary mapping task names (str) to their corresponding target tensors.
            Required during training to compute the loss. Defaults to None.
            The shape of each target tensor depends on the specific task head.
        masks : Dict[str, torch.Tensor], optional
            A dictionary mapping task names (str) to 1D boolean or float tensors indicating
            which samples in the batch have valid targets for that task.
            Shape of each mask tensor: `(batch_size,)`. Defaults to None (all samples assumed valid).
        categorical : torch.Tensor, optional
            A 2D tensor containing the categorical features for the batch, represented as integer indices.
            Shape: `(batch_size, num_categorical)`. Must have dtype `torch.long`. Defaults to None.
        missing_mask : torch.Tensor, optional
            A 2D boolean or float tensor indicating missing values in the input features (continuous and categorical combined).
            `True` or `1` indicates a missing value.
            Shape: `(batch_size, num_continuous + num_categorical)`. Defaults to None.
        sample_weights : torch.Tensor, optional
            A 1D tensor containing weights for each sample in the batch, used for weighted loss calculation.
            Shape: `(batch_size,)`. Defaults to None.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the results of the forward pass. Contents depend on whether `targets` were provided:
            If `targets` is provided (training/evaluation):
                - 'loss': The total combined loss (scalar tensor).
                - 'task_losses': A dictionary mapping task names to their individual losses.
                - 'task_outputs': A dictionary mapping task names to their respective output dictionaries (e.g., containing 'hazard', 'survival', 'predictions').
                - 'encoder_output': The output representation from the encoder. Shape: `(batch_size, encoder_dim)`.
                - 'attention_maps': Attention maps from the encoder's self-attention layers (if configured).
                - 'variational_loss': KL divergence loss if `include_variational` is True.
            If `targets` is None (prediction):
                - 'task_outputs': A dictionary mapping task names to their respective prediction dictionaries.
                - 'encoder_output': The output representation from the encoder.
                - 'attention_maps': Attention maps from the encoder.
        """
        # --- Input Validation ---
        assert isinstance(continuous, torch.Tensor), "Input 'continuous' must be a torch.Tensor"
        batch_size = continuous.size(0)
        assert continuous.ndim == 2, f"Input 'continuous' must be 2D (batch_size, num_continuous), got {continuous.ndim}D"
        assert continuous.size(1) == self.num_continuous, f"Input 'continuous' has incorrect number of features: expected {self.num_continuous}, got {continuous.size(1)}"

        if categorical is not None:
            assert isinstance(categorical, torch.Tensor), "Input 'categorical' must be a torch.Tensor"
            assert categorical.ndim == 2, f"Input 'categorical' must be 2D (batch_size, num_categorical), got {categorical.ndim}D"
            assert categorical.size(0) == batch_size, f"Batch size mismatch between 'continuous' ({batch_size}) and 'categorical' ({categorical.size(0)})"
            assert categorical.size(1) == len(self.cat_feat_info), f"Input 'categorical' has incorrect number of features: expected {len(self.cat_feat_info)}, got {categorical.size(1)}"
            assert categorical.dtype == torch.long, f"Input 'categorical' must have dtype torch.long, got {categorical.dtype}"

        if missing_mask is not None:
            assert isinstance(missing_mask, torch.Tensor), "Input 'missing_mask' must be a torch.Tensor"
            assert missing_mask.ndim == 2, f"Input 'missing_mask' must be 2D (batch_size, num_features), got {missing_mask.ndim}D"
            assert missing_mask.size(0) == batch_size, f"Batch size mismatch between 'continuous' ({batch_size}) and 'missing_mask' ({missing_mask.size(0)})"
            expected_features = self.num_continuous + len(self.cat_feat_info)
            assert missing_mask.size(1) == expected_features, f"Input 'missing_mask' has incorrect number of features: expected {expected_features}, got {missing_mask.size(1)}"

        if sample_weights is not None:
            assert isinstance(sample_weights, torch.Tensor), "Input 'sample_weights' must be a torch.Tensor"
            assert sample_weights.ndim == 1, f"Input 'sample_weights' must be 1D (batch_size), got {sample_weights.ndim}D"
            assert sample_weights.size(0) == batch_size, f"Batch size mismatch between 'continuous' ({batch_size}) and 'sample_weights' ({sample_weights.size(0)})"

        if targets is not None:
            assert isinstance(targets, dict), "Input 'targets' must be a dictionary"
            for task_name, target_tensor in targets.items():
                assert isinstance(target_tensor, torch.Tensor), f"Target for task '{task_name}' must be a torch.Tensor"
                assert target_tensor.size(0) == batch_size, f"Batch size mismatch between 'continuous' ({batch_size}) and target '{task_name}' ({target_tensor.size(0)})"
                # Specific shape checks might be needed per task, but basic batch size check is crucial

        if masks is not None:
            assert isinstance(masks, dict), "Input 'masks' must be a dictionary"
            for task_name, mask_tensor in masks.items():
                assert isinstance(mask_tensor, torch.Tensor), f"Mask for task '{task_name}' must be a torch.Tensor"
                assert mask_tensor.size(0) == batch_size, f"Batch size mismatch between 'continuous' ({batch_size}) and mask '{task_name}' ({mask_tensor.size(0)})"
                assert mask_tensor.ndim == 1, f"Mask for task '{task_name}' must be 1D (batch_size), got {mask_tensor.ndim}D"
        # --- End Input Validation ---

        # Get encoder representation
        encoder_output, attention_maps = self.encoder(
            continuous=continuous,
            categorical=categorical,
            missing_mask=missing_mask
        )
        
        # Pass through task manager
        if targets is not None:
            # Training mode with targets
            task_results = self.task_manager(
                x=encoder_output,
                targets=targets,
                masks=masks,
                sample_weights=sample_weights
            )
            
            # Collect results
            results = {
                'loss': task_results['loss'],
                'task_losses': task_results['task_losses'],
                'task_outputs': task_results['task_outputs'],
                'encoder_output': encoder_output,
                'attention_maps': attention_maps
            }
            
            # Add variational loss if used
            if self.include_variational:
                results['variational_loss'] = task_results['variational_loss']
        else:
            # Prediction mode without targets
            task_predictions = self.task_manager.predict(encoder_output)
            
            # Collect results
            results = {
                'task_outputs': task_predictions,
                'encoder_output': encoder_output,
                'attention_maps': attention_maps
            }
        
        return results
    
    def predict(self, 
               continuous: torch.Tensor,
               categorical: Optional[torch.Tensor] = None,
               missing_mask: Optional[torch.Tensor] = None,
               sample_weights: Optional[torch.Tensor] = None,
               return_representations: bool = False,
               return_attention: bool = False) -> Dict[str, Any]:
        """
        Generate predictions for all tasks without computing loss.

        Sets the model to evaluation mode and performs a forward pass without gradients.

        Parameters
        ----------
        continuous : torch.Tensor
            Continuous features. Shape: `(batch_size, num_continuous)`.
        categorical : torch.Tensor, optional
            Categorical features. Shape: `(batch_size, num_categorical)`. Defaults to None.
        missing_mask : torch.Tensor, optional
            Missing value mask. Shape: `(batch_size, num_features)`. Defaults to None.
        sample_weights : torch.Tensor, optional
            Sample weights (currently unused in prediction but included for consistency). Shape: `(batch_size,)`. Defaults to None.
        return_representations : bool, default=False
            If True, includes the encoder's output representation in the results dictionary under the key 'encoder_output'.
        return_attention : bool, default=False
            If True, includes the attention maps from the encoder in the results dictionary under the key 'attention_maps'.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the prediction results:
            - 'task_outputs': Dictionary mapping task names to their prediction dictionaries.
            - 'encoder_output': (Optional) Encoder representation if `return_representations` is True.
            - 'attention_maps': (Optional) Encoder attention maps if `return_attention` is True.
        """
        # --- Input Validation ---
        assert isinstance(continuous, torch.Tensor), "Input 'continuous' must be a torch.Tensor"
        batch_size = continuous.size(0)
        assert continuous.ndim == 2, f"Input 'continuous' must be 2D (batch_size, num_continuous), got {continuous.ndim}D"
        assert continuous.size(1) == self.num_continuous, f"Input 'continuous' has incorrect number of features: expected {self.num_continuous}, got {continuous.size(1)}"

        if categorical is not None:
            assert isinstance(categorical, torch.Tensor), "Input 'categorical' must be a torch.Tensor"
            assert categorical.ndim == 2, f"Input 'categorical' must be 2D (batch_size, num_categorical), got {categorical.ndim}D"
            assert categorical.size(0) == batch_size, f"Batch size mismatch between 'continuous' ({batch_size}) and 'categorical' ({categorical.size(0)})"
            assert categorical.size(1) == len(self.cat_feat_info), f"Input 'categorical' has incorrect number of features: expected {len(self.cat_feat_info)}, got {categorical.size(1)}"
            assert categorical.dtype == torch.long, f"Input 'categorical' must have dtype torch.long, got {categorical.dtype}"

        if missing_mask is not None:
            assert isinstance(missing_mask, torch.Tensor), "Input 'missing_mask' must be a torch.Tensor"
            assert missing_mask.ndim == 2, f"Input 'missing_mask' must be 2D (batch_size, num_features), got {missing_mask.ndim}D"
            assert missing_mask.size(0) == batch_size, f"Batch size mismatch between 'continuous' ({batch_size}) and 'missing_mask' ({missing_mask.size(0)})"
            expected_features = self.num_continuous + len(self.cat_feat_info)
            assert missing_mask.size(1) == expected_features, f"Input 'missing_mask' has incorrect number of features: expected {expected_features}, got {missing_mask.size(1)}"

        if sample_weights is not None:
            assert isinstance(sample_weights, torch.Tensor), "Input 'sample_weights' must be a torch.Tensor"
            assert sample_weights.ndim == 1, f"Input 'sample_weights' must be 1D (batch_size), got {sample_weights.ndim}D"
            assert sample_weights.size(0) == batch_size, f"Batch size mismatch between 'continuous' ({batch_size}) and 'sample_weights' ({sample_weights.size(0)})"
        # --- End Input Validation ---

        # Set model to evaluation mode
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            results = self.forward(
                continuous=continuous,
                categorical=categorical,
                missing_mask=missing_mask,
                sample_weights=sample_weights
            )
            
            # Extract predictions
            predictions = {
                'task_outputs': results['task_outputs']
            }
            
            # Add optional outputs
            if return_representations:
                predictions['encoder_output'] = results['encoder_output']
                
            if return_attention:
                predictions['attention_maps'] = results['attention_maps']
        
        return predictions
    
    def compute_uncertainty(self, 
                          continuous: torch.Tensor,
                          categorical: Optional[torch.Tensor] = None,
                          missing_mask: Optional[torch.Tensor] = None,
                          num_samples: int = 10) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute prediction uncertainty using Monte Carlo dropout.
        
        Parameters
        ----------
        continuous : torch.Tensor
            Continuous features [batch_size, num_continuous]
            
        categorical : torch.Tensor, optional
            Categorical features [batch_size, num_categorical]
            
        missing_mask : torch.Tensor, optional
            Missing value mask [batch_size, num_features]
            
        num_samples : int, default=10
            Number of Monte Carlo samples
            
        Returns
        -------
        Dict[str, Dict[str, torch.Tensor]]
            Dictionary mapping task names to uncertainty metrics:
            - 'mean': Mean prediction
            - 'std': Standard deviation
            - 'samples': Individual samples
        """
        # --- Input Validation ---
        assert isinstance(continuous, torch.Tensor), "Input 'continuous' must be a torch.Tensor"
        batch_size = continuous.size(0)
        assert continuous.ndim == 2, f"Input 'continuous' must be 2D (batch_size, num_continuous), got {continuous.ndim}D"
        assert continuous.size(1) == self.num_continuous, f"Input 'continuous' has incorrect number of features: expected {self.num_continuous}, got {continuous.size(1)}"

        if categorical is not None:
            assert isinstance(categorical, torch.Tensor), "Input 'categorical' must be a torch.Tensor"
            assert categorical.ndim == 2, f"Input 'categorical' must be 2D (batch_size, num_categorical), got {categorical.ndim}D"
            assert categorical.size(0) == batch_size, f"Batch size mismatch between 'continuous' ({batch_size}) and 'categorical' ({categorical.size(0)})"
            assert categorical.size(1) == len(self.cat_feat_info), f"Input 'categorical' has incorrect number of features: expected {len(self.cat_feat_info)}, got {categorical.size(1)}"
            assert categorical.dtype == torch.long, f"Input 'categorical' must have dtype torch.long, got {categorical.dtype}"

        if missing_mask is not None:
            assert isinstance(missing_mask, torch.Tensor), "Input 'missing_mask' must be a torch.Tensor"
            assert missing_mask.ndim == 2, f"Input 'missing_mask' must be 2D (batch_size, num_features), got {missing_mask.ndim}D"
            assert missing_mask.size(0) == batch_size, f"Batch size mismatch between 'continuous' ({batch_size}) and 'missing_mask' ({missing_mask.size(0)})"
            expected_features = self.num_continuous + len(self.cat_feat_info)
            assert missing_mask.size(1) == expected_features, f"Input 'missing_mask' has incorrect number of features: expected {expected_features}, got {missing_mask.size(1)}"
        
        assert isinstance(num_samples, int) and num_samples > 0, "Input 'num_samples' must be a positive integer"
        # --- End Input Validation ---

        # Set model to training mode to enable dropout
        self.train()
        
        # Initialize storage for samples
        task_names = [task.name for task in self.task_manager.task_heads]
        all_samples = {name: [] for name in task_names}
        
        # Generate multiple predictions
        with torch.no_grad():
            for _ in range(num_samples):
                # Forward pass
                results = self.forward(
                    continuous=continuous,
                    categorical=categorical,
                    missing_mask=missing_mask
                )
                
                # Extract task outputs
                task_outputs = results['task_outputs']
                
                # Store predictions for each task
                for name, outputs in task_outputs.items():
                    if 'predictions' in outputs:
                        all_samples[name].append(outputs['predictions'].unsqueeze(0))
                    elif 'survival' in outputs:
                        all_samples[name].append(outputs['survival'].unsqueeze(0))
                    elif 'cif' in outputs:
                        all_samples[name].append(outputs['cif'].unsqueeze(0))
        
        # Compute statistics
        uncertainty = {}
        
        for name, samples in all_samples.items():
            if samples:
                # Stack samples
                stacked = torch.cat(samples, dim=0)
                
                # Compute mean and std - ensure floating point
                stacked = stacked.float()
                mean = torch.mean(stacked, dim=0)
                std = torch.std(stacked, dim=0)
                
                uncertainty[name] = {
                    'mean': mean,
                    'std': std,
                    'samples': stacked
                }
        
        # Set model back to evaluation mode
        self.eval()
        
        return uncertainty
    
    def get_task(self, name: str) -> Optional[TaskHead]:
        """
        Get a task head by name.
        
        Parameters
        ----------
        name : str
            Name of the task
            
        Returns
        -------
        Optional[TaskHead]
            The task head with the given name, or None if not found
        """
        return self.task_manager.get_task(name)
    
    def fit(self,
           train_loader: torch.utils.data.DataLoader,
           val_loader: Optional[torch.utils.data.DataLoader] = None,
           learning_rate: float = 1e-3,
           weight_decay: float = 1e-4,
           num_epochs: int = 100,
           patience: int = 10,
           optimize_metric: Optional[str] = None,
           callbacks: Optional[List[Any]] = None,
           use_sample_weights: bool = False) -> Dict[str, List[float]]:
        """
        Train the EnhancedDeepHit model using the provided data loaders and training configuration.

        Implements a standard training loop with support for validation, early stopping based on
        validation loss or a specified metric, learning rate scheduling, mixed precision (if CUDA is available),
        and optional callbacks.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader providing batches of training data. Each batch should be a dictionary
            containing at least 'continuous' and 'targets'. Optional keys include 'categorical',
            'missing_mask', 'masks', and 'sample_weights'.
        val_loader : torch.utils.data.DataLoader, optional
            DataLoader providing batches of validation data. Used for monitoring performance,
            learning rate scheduling, and early stopping. Defaults to None.
        learning_rate : float, default=1e-3
            The initial learning rate for the Adam optimizer.
        weight_decay : float, default=1e-4
            The weight decay (L2 penalty) for the Adam optimizer.
        num_epochs : int, default=100
            The maximum number of epochs to train for.
        patience : int, default=10
            The number of epochs to wait for improvement in the validation metric before
            early stopping. Only used if `val_loader` is provided.
        optimize_metric : str, optional
            The name of the validation metric to monitor for early stopping (e.g., 'survival_task_c_index').
            If None, validation loss is used. The metric name should match a key in the dictionary
            returned by the `evaluate` method. Assumes higher values are better for the metric.
        callbacks : List[Any], optional
            A list of callback functions or objects to be called at the end of each epoch.
            Each callback will receive the model instance, the current epoch number, and a logs dictionary.
            Defaults to None.
        use_sample_weights : bool, default=False
            If True, the training loop will look for a 'sample_weights' key in the batches
            provided by the data loaders and pass them to the forward pass for weighted loss calculation.

        Returns
        -------
        Dict[str, List[float]]
            A dictionary containing the training history, including lists of training loss,
            validation loss (if applicable), and any validation metrics recorded per epoch.
        """
        # Set model to training mode
        self.train()
        
        # Initialize optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Initialize GradScaler for mixed precision if CUDA is available
        scaler = GradScaler() if self.device == 'cuda' else None
        
        # Initialize learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience // 3, min_lr=1e-6
        )
        
        # Initialize early stopping
        best_val_loss = float('inf')
        best_val_metric = float('-inf') if optimize_metric else None
        best_epoch = 0
        best_state = None
        
        # Initialize history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            # Training step
            train_loss = 0.0
            train_metrics = {}
            num_batches = 0
            
            for batch in train_loader:
                # Extract batch data
                continuous = batch['continuous'].to(self.device)
                targets = {name: tensor.to(self.device) for name, tensor in batch['targets'].items()}
                
                # Optional elements
                categorical = batch.get('categorical', None)
                if categorical is not None:
                    categorical = categorical.to(self.device)
                    
                missing_mask = batch.get('missing_mask', None)
                if missing_mask is not None:
                    missing_mask = missing_mask.to(self.device)
                    
                masks = batch.get('masks', None)
                if masks is not None:
                    masks = {name: tensor.to(self.device) for name, tensor in masks.items()}
                
                # Extract sample weights if enabled
                sample_weights = None
                if use_sample_weights and 'sample_weights' in batch:
                    sample_weights = batch['sample_weights'].to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()

                # Forward pass with autocast if using CUDA
                if scaler is not None:
                    with autocast():
                        outputs = self.forward(
                            continuous=continuous,
                            targets=targets,
                            masks=masks,
                            categorical=categorical,
                            missing_mask=missing_mask,
                            sample_weights=sample_weights
                        )
                        loss = outputs['loss']
                else:
                    # Standard forward pass if not using CUDA
                    outputs = self.forward(
                        continuous=continuous,
                        targets=targets,
                        masks=masks,
                        categorical=categorical,
                        missing_mask=missing_mask,
                        sample_weights=sample_weights
                    )
                    loss = outputs['loss']
                
                # Backward pass with scaler if using CUDA
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard backward pass and optimizer step
                    loss.backward()
                    optimizer.step()
                
                # Accumulate loss
                train_loss += loss.item()
                num_batches += 1
            
            # Compute average training loss
            train_loss /= max(num_batches, 1)
            history['train_loss'].append(train_loss)
            
            # Validation step if loader provided
            if val_loader is not None:
                val_loss, val_metrics = self.evaluate(val_loader, use_sample_weights=use_sample_weights)
                history['val_loss'].append(val_loss)
                
                # Update learning rate scheduler
                scheduler.step(val_loss)
                
                # Check for early stopping
                improved = False
                
                if optimize_metric is not None and optimize_metric in val_metrics:
                    # Optimize for specified metric
                    current_metric = val_metrics[optimize_metric]
                    if current_metric > best_val_metric:
                        best_val_metric = current_metric
                        best_epoch = epoch
                        best_state = self.state_dict()
                        improved = True
                else:
                    # Optimize for validation loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch
                        best_state = self.state_dict()
                        improved = True
                
                # Early stopping check
                if epoch - best_epoch >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                
                # Status update
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Additional metrics
                for name, value in val_metrics.items():
                    if name not in history:
                        history[name] = []
                    history[name].append(value)
                    print(f"  {name}: {value:.4f}")
            else:
                # Status update without validation
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
            
            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self, epoch, {
                        'train_loss': train_loss,
                        'val_loss': val_loss if val_loader else None,
                        'val_metrics': val_metrics if val_loader else None,
                        'improved': improved if val_loader else None
                    })
        
        # Restore best model if validation was used
        if val_loader is not None and best_state is not None:
            self.load_state_dict(best_state)
            print(f"Restored model from epoch {best_epoch+1}")
        
        return history
    
    def evaluate(self, data_loader: torch.utils.data.DataLoader, use_sample_weights: bool = False) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the model's performance on a given dataset.

        Sets the model to evaluation mode, iterates through the data loader, computes the loss,
        and calculates task-specific metrics.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader providing batches of evaluation data. Each batch should be structured
            similarly to the training data loader batches.
        use_sample_weights : bool, default=False
            If True, uses 'sample_weights' from the batches for loss calculation during evaluation.

        Returns
        -------
        Tuple[float, Dict[str, float]]
            A tuple containing:
            - The average loss over the entire dataset (float).
            - A dictionary mapping metric names (e.g., 'task_name_metric_name') to their
              average values over the dataset (Dict[str, float]).
        """
        # Set model to evaluation mode
        self.eval()
        
        # Initialize metrics
        total_loss = 0.0
        num_batches = 0
        task_metrics = {}
        
        # Evaluation loop
        with torch.no_grad():
            for batch in data_loader:
                # Extract batch data
                continuous = batch['continuous'].to(self.device)
                targets = {name: tensor.to(self.device) for name, tensor in batch['targets'].items()}
                
                # Optional elements
                categorical = batch.get('categorical', None)
                if categorical is not None:
                    categorical = categorical.to(self.device)
                    
                missing_mask = batch.get('missing_mask', None)
                if missing_mask is not None:
                    missing_mask = missing_mask.to(self.device)
                    
                masks = batch.get('masks', None)
                if masks is not None:
                    masks = {name: tensor.to(self.device) for name, tensor in masks.items()}
                
                # Extract sample weights if enabled
                sample_weights = None
                if use_sample_weights and 'sample_weights' in batch:
                    sample_weights = batch['sample_weights'].to(self.device)
                
                # Forward pass
                outputs = self.forward(
                    continuous=continuous,
                    targets=targets,
                    masks=masks,
                    categorical=categorical,
                    missing_mask=missing_mask,
                    sample_weights=sample_weights
                )
                
                # Extract loss
                loss = outputs['loss']
                
                # Accumulate loss
                total_loss += loss.item()
                num_batches += 1
                
                # Compute task-specific metrics
                task_outputs = outputs['task_outputs']
                
                for task_name, task_output in task_outputs.items():
                    if task_name in targets:
                        task_head = self.get_task(task_name)
                        if task_head:
                            task_target = targets[task_name]
                            task_metric = task_head.compute_metrics(task_output, task_target)
                            
                            # Accumulate metrics
                            for metric_name, metric_value in task_metric.items():
                                full_name = f"{task_name}_{metric_name}"
                                if full_name not in task_metrics:
                                    task_metrics[full_name] = 0.0
                                task_metrics[full_name] += metric_value
            
            # Compute average metrics
            if num_batches > 0:
                total_loss /= num_batches
                
                for metric_name in task_metrics:
                    task_metrics[metric_name] /= num_batches
        
        # Set model back to training mode
        self.train()
        
        return total_loss, task_metrics
    
    def save(self, path: str, save_processor: bool = True, processor: Optional[DataProcessor] = None, metadata: dict = None):
        """
        Save the model to disk with versioning and metadata.
        
        Parameters
        ----------
        path : str
            Path to save the model
            
        save_processor : bool, default=True
            Whether to save the data processor
            
        processor : DataProcessor, optional
            Data processor to save
            
        metadata : dict, optional
            Additional metadata to store with the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Generate version identifier based on timestamp
        import time
        version = int(time.time())
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare comprehensive metadata
        full_metadata = {
            "version": version,
            "timestamp": timestamp,
            "architecture": {
                "num_continuous": self.num_continuous,
                "encoder_dim": self.encoder_dim,
                "encoder_depth": self.encoder.depth,
                "encoder_heads": self.encoder.heads,
                "encoder_ff_dim": getattr(self.encoder, 'ff_dim', 512),
                "encoder_dropout": getattr(self.encoder, 'attn_dropout', 0.1),
                "include_variational": self.include_variational,
                "variational_beta": getattr(self.task_manager, 'variational_beta', 0.1),
            },
            "task_config": [task.get_config() for task in self.task_manager.task_heads],
            "user_metadata": metadata or {}
        }
        
        # Save model parameters with versioning
        versioned_path = f"{path}_v{version}"
        torch.save({
            "state_dict": self.state_dict(),
            "metadata": full_metadata
        }, f"{versioned_path}.pt")
        
        # Save model configuration with versioning
        with open(f"{versioned_path}.json", 'w') as f:
            json.dump(full_metadata, f, indent=2)
        
        # Save reference to latest version
        with open(f"{path}_latest.txt", "w") as f:
            f.write(str(version))
        
        # Additionally save as the base filename for backwards compatibility
        torch.save(self.state_dict(), f"{path}.pt")
        
        # Save config as before (for backwards compatibility)
        config = {
            'num_continuous': self.num_continuous,
            'encoder_dim': self.encoder_dim,
            'encoder_depth': self.encoder.depth,
            'encoder_heads': self.encoder.heads,
            'encoder_ff_dim': getattr(self.encoder, 'ff_dim', 512),
            'encoder_dropout': getattr(self.encoder, 'attn_dropout', 0.1),
            'include_variational': self.include_variational,
            'variational_beta': getattr(self.task_manager, 'variational_beta', 0.1),
            'cat_feat_info': self.cat_feat_info,
            'task_config': [task.get_config() for task in self.task_manager.task_heads]
        }
        
        with open(f"{path}.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save data processor if provided
        if save_processor and processor is not None:
            import pickle
            processor_path = f"{versioned_path}_processor.pkl"
            with open(processor_path, 'wb') as f:
                pickle.dump(processor, f)
            
            # Also save at the original path for backwards compatibility
            with open(f"{path}_processor.pkl", 'wb') as f:
                pickle.dump(processor, f)
        
        logging.info(f"Model saved with version {version} at {timestamp}")
        return version
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu', load_processor: bool = True, version: str = 'latest') -> Tuple['EnhancedDeepHit', Optional[DataProcessor]]:
        """
        Load a model from disk.
        
        Parameters
        ----------
        path : str
            Path to the saved model
            
        device : str, default='cpu'
            Device to load the model on
            
        load_processor : bool, default=True
            Whether to load the data processor
            
        version : str, default='latest'
            Version to load. Can be:
            - 'latest': Load the most recent version
            - Specific version number as string: Load that specific version
            - None: Load using the legacy format (no versioning)
            
        Returns
        -------
        Tuple[EnhancedDeepHit, Optional[DataProcessor]]
            Tuple containing:
            - Loaded model
            - Data processor (if available)
            - Metadata dictionary (if available in versioned model)
        """
        model_path = path
        processor_path = f"{path}_processor.pkl"
        config_path = f"{path}.json"
        metadata = None
        
        # Handle versioned models
        if version is not None:
            if version == 'latest':
                # Try to read the latest version number
                latest_path = f"{path}_latest.txt"
                if os.path.exists(latest_path):
                    with open(latest_path, 'r') as f:
                        version = f.read().strip()
                        logging.info(f"Loading latest model version: {version}")
                        model_path = f"{path}_v{version}"
                        processor_path = f"{model_path}_processor.pkl"
                        config_path = f"{model_path}.json"
            else:
                # Use the specified version
                model_path = f"{path}_v{version}" 
                processor_path = f"{model_path}_processor.pkl"
                config_path = f"{model_path}.json"
        
        # Check if we're loading a versioned model with metadata
        if os.path.exists(f"{model_path}.pt"):
            try:
                # Try to load as a versioned model first
                saved_data = torch.load(f"{model_path}.pt", map_location=device)
                if isinstance(saved_data, dict) and "state_dict" in saved_data:
                    state_dict = saved_data["state_dict"]
                    metadata = saved_data.get("metadata", None)
                    
                    if metadata:
                        logging.info(f"Loaded model with version {metadata.get('version')} from {metadata.get('timestamp')}")
                else:
                    # Fall back to legacy format
                    state_dict = saved_data
            except Exception as e:
                # Fall back to legacy format
                logging.warning(f"Error loading versioned model, trying legacy format: {str(e)}")
                state_dict = torch.load(f"{path}.pt", map_location=device)
        else:
            # Use legacy path
            state_dict = torch.load(f"{path}.pt", map_location=device)
            config_path = f"{path}.json"
            processor_path = f"{path}_processor.pkl"
        
        # Load model configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract configuration from metadata if available, otherwise use legacy config
        if metadata and "architecture" in metadata:
            arch_config = metadata["architecture"]
            task_config = metadata["task_config"]
        else:
            arch_config = config
            task_config = config.get('task_config', [])
        
        # Load task configurations
        from tsunami.models.tasks.survival import SingleRiskHead, CompetingRisksHead
        from tsunami.models.tasks.standard import ClassificationHead, RegressionHead, CountDataHead
        
        task_classes = {
            'SingleRiskHead': SingleRiskHead,
            'CompetingRisksHead': CompetingRisksHead,
            'ClassificationHead': ClassificationHead,
            'RegressionHead': RegressionHead,
            'CountDataHead': CountDataHead
        }
        
        # Instantiate task heads
        tasks = []
        for task_config in config['task_config']:
            task_class = task_classes[task_config['task_type']]
            
            # Remove task_type from config
            task_params = {k: v for k, v in task_config.items() if k != 'task_type'}
            
            # Instantiate task
            task = task_class(**task_params)
            tasks.append(task)
        
        # Instantiate model
        model = cls(
            num_continuous=arch_config['num_continuous'],
            targets=tasks,
            cat_feat_info=config.get('cat_feat_info'),
            encoder_dim=arch_config['encoder_dim'],
            include_variational=arch_config.get('include_variational', False),
            device=device
        )
        
        # Load model parameters
        model.load_state_dict(state_dict)
        
        # Load data processor if requested
        processor = None
        if load_processor and os.path.exists(processor_path):
            import pickle
            with open(processor_path, 'rb') as f:
                processor = pickle.load(f)
        
        return model, processor, metadata if metadata else None
