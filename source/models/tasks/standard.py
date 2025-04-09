import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from source.models.tasks.base import TaskHead


class ClassificationHead(TaskHead):
    """
    Task head for classification tasks (binary or multi-class).
    
    Predicts probabilities for each class, supporting binary and multi-class problems.
    """
    
    def __init__(self,
                name: str,
                input_dim: int,
                num_classes: int,
                class_weights: Optional[List[float]] = None,
                dropout: float = 0.1,
                task_weight: float = 1.0):
        """
        Initialize ClassificationHead.
        
        Parameters
        ----------
        name : str
            Name of the task
            
        input_dim : int
            Dimension of the input representation
            
        num_classes : int
            Number of classes (2 for binary classification)
            
        class_weights : List[float], optional
            Weights for each class to handle imbalanced data
            
        dropout : float, default=0.1
            Dropout rate for the prediction network
            
        task_weight : float, default=1.0
            Weight of this task in the multi-task loss
        """
        super().__init__(name, input_dim, task_weight)

        # --- Input Validation ---
        assert isinstance(name, str) and name, "'name' must be a non-empty string"
        assert isinstance(input_dim, int) and input_dim > 0, "'input_dim' must be a positive integer"
        assert isinstance(num_classes, int) and num_classes >= 1, "'num_classes' must be at least 1 (use 2 for binary)"
        if class_weights is not None:
            assert isinstance(class_weights, list) and len(class_weights) == num_classes, f"'class_weights' must be a list of length {num_classes}"
            assert all(isinstance(w, (int, float)) and w >= 0 for w in class_weights), "'class_weights' must contain non-negative numbers"
        assert isinstance(dropout, float) and 0.0 <= dropout < 1.0, "'dropout' must be a float between 0.0 and 1.0"
        assert isinstance(task_weight, float) and task_weight >= 0.0, "'task_weight' must be a non-negative float"
        # --- End Input Validation ---
        
        self.num_classes = num_classes
        self.class_weights = class_weights
        
        # Architecture
        # For binary classification, output a single value
        output_dim = 1 if num_classes == 2 else num_classes
        
        self.prediction_network = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, output_dim)
        )
        
    def loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between model outputs and targets.
        
        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            Outputs from the forward pass
            
        targets : torch.Tensor
            Target class labels
            
        Returns
        -------
        torch.Tensor
            Loss value
        """
        # If loss was pre-computed in forward pass (or already available)
        if 'loss' in outputs and outputs['loss'].item() > 0:
            return outputs['loss']
        
        # If loss wasn't precomputed, compute it now
        if self.num_classes == 2:
            # Binary classification
            if 'logits' in outputs:
                logits = outputs['logits']
            else:
                # If logits not available, convert probabilities to logits
                probs = torch.clamp(outputs['probabilities'], 1e-5, 1-1e-5)
                logits = torch.log(probs / (1 - probs))
            
            # Make sure targets are the right shape for binary classification
            if targets.dim() > 1:
                # If targets have shape [batch, 1], squeeze them
                targets = targets.squeeze(-1)
                
            # Binary cross-entropy loss
            loss_fn = nn.BCEWithLogitsLoss()
            return loss_fn(logits, targets.float())
        else:
            # Multi-class classification
            if 'logits' in outputs:
                logits = outputs['logits']
            else:
                # If logits not available, use log of probabilities
                probs = torch.clamp(outputs['probabilities'], 1e-5, 1-1e-5)
                logits = torch.log(probs)
            
            # Convert targets to indices if they are one-hot encoded
            if targets.dim() > 1 and targets.size(1) == self.num_classes:
                targets = torch.argmax(targets, dim=1)
                
            # Cross-entropy loss
            if self.class_weights is not None:
                # Apply class weights
                class_weights = torch.tensor(self.class_weights, device=logits.device)
                loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            else:
                loss_fn = nn.CrossEntropyLoss()
                
            return loss_fn(logits, targets)
        
    def forward(self, 
               x: torch.Tensor, 
               targets: Optional[torch.Tensor] = None, 
               mask: Optional[torch.Tensor] = None,
               sample_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for classification prediction.
        
        Parameters
        ----------
        x : torch.Tensor
            Input representation [batch_size, input_dim]
            
        targets : torch.Tensor, optional
            Target class indices [batch_size] or one-hot encoded [batch_size, num_classes]
            
        mask : torch.Tensor, optional
            Mask indicating which samples have targets for this task
            [batch_size], where 1 means target is available
            
        sample_weights : torch.Tensor, optional
            Sample weights for weighted loss calculation [batch_size]
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'loss': Loss for this task (if targets provided)
            - 'logits': Raw model outputs
            - 'probabilities': Predicted class probabilities
            - 'predictions': Predicted class indices
        """
        # Get batch size and device
        batch_size = x.size(0)
        device = x.device
        
        # Compute logits
        logits = self.prediction_network(x)
        
        # Convert to appropriate shape for binary classification
        if self.num_classes == 2:
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits)
            predictions = (probs >= 0.5).long()
        else:
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
        
        # Initialize loss
        loss = torch.tensor(0.0, device=device)
        
        # If targets are provided, compute the loss
        if targets is not None:
            # Convert targets to appropriate format if needed
            # Adjust dimensions for binary classification
            if self.num_classes == 2:
                # Make sure targets are the right shape for binary classification
                if targets.dim() > 1:
                    # If targets have shape [batch, 1], squeeze them
                    targets = targets.squeeze(-1)
            elif targets.dim() > 1 and targets.size(1) == self.num_classes:
                # One-hot encoded, convert to indices for loss (for multi-class)
                targets = torch.argmax(targets, dim=1)
            
            # Apply mask if provided
            if mask is not None:
                # Expand mask to match targets
                mask = mask.float()
                valid_samples = torch.sum(mask)
            else:
                # All samples are valid
                valid_samples = float(batch_size)
                mask = torch.ones_like(targets, dtype=torch.float, device=device)
            
            # Apply sample weights if provided
            if sample_weights is not None:
                # Combine with mask
                combined_weights = mask * sample_weights
            else:
                combined_weights = mask
            
            # Skip loss computation if no valid samples
            if valid_samples > 0:
                # Compute loss
                if self.num_classes == 2:
                    # Binary cross-entropy loss (Logits version)
                    # L = -[y * log(sigmoid(x)) + (1 - y) * log(1 - sigmoid(x))]
                    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
                    loss_values = loss_fn(logits, targets.float())
                    
                    # Apply combined weights and normalize
                    loss = torch.sum(loss_values * combined_weights) / torch.sum(combined_weights)
                else:
                    # Multi-class cross-entropy loss
                    if self.class_weights is not None:
                        # Apply class weights
                        class_weights = torch.tensor(self.class_weights, device=device)
                        loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
                    else:
                        loss_fn = nn.CrossEntropyLoss(reduction='none')
                    
                    # Multi-class cross-entropy loss (LogSoftmax + NLLLoss)
                    # L = -log(softmax(x)_class) = - (x_class - log(sum(exp(x_j))))
                    loss_values = loss_fn(logits, targets)
                    
                    # Apply combined weights and normalize
                    loss = torch.sum(loss_values * combined_weights) / torch.sum(combined_weights)
        
        return {
            'loss': loss,
            'logits': logits,
            'probabilities': probs,
            'predictions': predictions
        }
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate classification predictions.
        
        Parameters
        ----------
        x : torch.Tensor
            Input representation [batch_size, input_dim]
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'logits': Raw model outputs
            - 'probabilities': Predicted class probabilities
            - 'predictions': Predicted class indices
        """
        # Compute predictions (no loss)
        outputs = self.forward(x)
        
        # Remove loss from outputs
        outputs.pop('loss', None)
        
        return outputs
    
    def compute_metrics(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute evaluation metrics for classification.
        
        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            Model outputs from the forward pass
            
        targets : torch.Tensor
            Ground truth class indices [batch_size]
            
        Returns
        -------
        Dict[str, float]
            Dictionary of metric names and values including:
            - 'accuracy': Overall accuracy
            - 'f1_score' (binary) or 'macro_f1_score'/'micro_f1_score' (multiclass)
            - 'auc': Area under ROC curve (for binary only)
            - 'precision'/'recall' (binary) or 'macro_precision'/'micro_precision' (multiclass)
        """
        # Extract predictions and targets
        probs = outputs['probabilities'].detach().cpu().numpy()
        preds = outputs['predictions'].detach().cpu().numpy()
        
        # Convert targets to indices if they are one-hot encoded
        if targets.dim() > 1 and targets.size(1) == self.num_classes:
            targets = torch.argmax(targets, dim=1)
            
        targets = targets.detach().cpu().numpy()
        
        # Compute metrics
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = np.mean(preds == targets)
        
        # F1 score, precision, recall
        if self.num_classes == 2 or (self.num_classes == 1 and np.isin(np.unique(targets), [0, 1]).all()):
            # Binary classification metrics
            from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
            
            # For binary with one output, flatten predictions if needed
            if self.num_classes == 1 and hasattr(preds, 'flatten'):
                preds = preds.flatten()
                if probs.ndim > 1:
                    probs = probs.flatten()
            
            metrics['f1_score'] = f1_score(targets, preds, zero_division=0)
            metrics['precision'] = precision_score(targets, preds, zero_division=0)
            metrics['recall'] = recall_score(targets, preds, zero_division=0)
            
            # ROC AUC only if we have both classes in the targets
            if len(np.unique(targets)) > 1:
                try:
                    metrics['auc'] = roc_auc_score(targets, probs)
                except Exception:
                    # If AUC calculation fails (e.g., only one class in batch), skip it
                    pass
        else:
            # Multi-class metrics
            from sklearn.metrics import f1_score, precision_score, recall_score
            
            # Include both macro and micro averaged metrics
            # Macro: Calculate metric for each class and average (treats all classes equally)
            # Micro: Calculate metric by considering each element of the label indicator matrix
            metrics['macro_f1_score'] = f1_score(targets, preds, average='macro', zero_division=0)
            metrics['micro_f1_score'] = f1_score(targets, preds, average='micro', zero_division=0)
            metrics['macro_precision'] = precision_score(targets, preds, average='macro', zero_division=0)
            metrics['micro_precision'] = precision_score(targets, preds, average='micro', zero_division=0)
            metrics['macro_recall'] = recall_score(targets, preds, average='macro', zero_division=0)
            metrics['micro_recall'] = recall_score(targets, preds, average='micro', zero_division=0)
            
            # Weighted metrics (accounts for class imbalance)
            metrics['weighted_f1_score'] = f1_score(targets, preds, average='weighted', zero_division=0)
            
            # Also add the old metrics for backwards compatibility
            metrics['f1_score'] = metrics['macro_f1_score']
            metrics['precision'] = metrics['macro_precision']
            metrics['recall'] = metrics['macro_recall']
            
            # Per-class metrics if requested - uncomment to enable
            # for cls in range(self.num_classes):
            #     cls_mask = targets == cls
            #     if np.any(cls_mask):
            #         metrics[f'class_{cls}_precision'] = precision_score(
            #             targets == cls, preds == cls, zero_division=0)
            #         metrics[f'class_{cls}_recall'] = recall_score(
            #             targets == cls, preds == cls, zero_division=0)
            #         metrics[f'class_{cls}_f1'] = f1_score(
            #             targets == cls, preds == cls, zero_division=0)
            
        return metrics
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration dictionary for serialization.
        
        Returns
        -------
        Dict[str, Any]
            Configuration dictionary
        """
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'class_weights': self.class_weights
        })
        return config


class RegressionHead(TaskHead):
    """
    Task head for regression tasks.
    
    Predicts continuous values, supporting both standard and quantile regression.
    """
    
    def __init__(self,
                name: str,
                input_dim: int,
                output_dim: int = 1,
                quantiles: Optional[List[float]] = None,
                loss_type: str = 'mse',
                dropout: float = 0.1,
                task_weight: float = 1.0):
        """
        Initialize RegressionHead.
        
        Parameters
        ----------
        name : str
            Name of the task
            
        input_dim : int
            Dimension of the input representation
            
        output_dim : int, default=1
            Number of output values to predict
            
        quantiles : List[float], optional
            List of quantiles to predict for quantile regression
            
        loss_type : str, default='mse'
            Type of loss function ('mse', 'mae', 'huber', 'quantile')
            
        dropout : float, default=0.1
            Dropout rate for the prediction network
            
        task_weight : float, default=1.0
            Weight of this task in the multi-task loss
        """
        super().__init__(name, input_dim, task_weight)

        # --- Input Validation ---
        assert isinstance(name, str) and name, "'name' must be a non-empty string"
        assert isinstance(input_dim, int) and input_dim > 0, "'input_dim' must be a positive integer"
        assert isinstance(output_dim, int) and output_dim > 0, "'output_dim' must be a positive integer"
        if quantiles is not None:
            assert isinstance(quantiles, list) and all(isinstance(q, float) and 0 < q < 1 for q in quantiles), "'quantiles' must be a list of floats between 0 and 1"
            assert loss_type == 'quantile', "loss_type must be 'quantile' when quantiles are provided"
        assert loss_type in ['mse', 'mae', 'huber', 'quantile'], "loss_type must be one of 'mse', 'mae', 'huber', 'quantile'"
        assert isinstance(dropout, float) and 0.0 <= dropout < 1.0, "'dropout' must be a float between 0.0 and 1.0"
        assert isinstance(task_weight, float) and task_weight >= 0.0, "'task_weight' must be a non-negative float"
        # --- End Input Validation ---
        
        self.output_dim = output_dim
        self.quantiles = quantiles
        self.loss_type = loss_type
        
        # Architecture
        # If using quantile regression, output dimension is multiplied by number of quantiles
        actual_output_dim = output_dim
        if quantiles is not None:
            actual_output_dim = output_dim * len(quantiles)
        
        self.prediction_network = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, actual_output_dim)
        )
        
        # Initialize loss function
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_type == 'mae':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'huber':
            self.loss_fn = nn.SmoothL1Loss(reduction='none')
        elif loss_type == 'quantile':
            # Quantile loss is handled separately
            if quantiles is None:
                raise ValueError("Quantiles must be provided for quantile regression")
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
            
    def loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between model outputs and targets.
        
        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            Outputs from the forward pass
            
        targets : torch.Tensor
            Target values for regression
            
        Returns
        -------
        torch.Tensor
            Loss value
        """
        if 'loss' in outputs:
            return outputs['loss']
            
        # If loss wasn't precomputed, compute it now
        predictions = outputs['predictions']
        
        # Ensure predictions match targets dimension
        if predictions.dim() > 1 and targets.dim() == 1:
            # If predictions are [batch_size, 1] but targets are [batch_size]
            predictions = predictions.squeeze(-1)
        
        # Handle missing output dimensions
        if targets.dim() == 1 and predictions.dim() > 1 and predictions.size(1) > 1:
            targets = targets.unsqueeze(1).expand(-1, predictions.size(1))
        
        # Compute loss
        if self.loss_type == 'quantile':
            # Quantile regression loss
            if self.quantiles is not None:
                if 'quantiles' in outputs:
                    quantile_preds = outputs['quantiles']
                    return self._compute_quantile_loss(quantile_preds, targets, torch.ones_like(targets))
        
        # Standard loss
        loss_values = self.loss_fn(predictions, targets)
        return torch.mean(loss_values)
        
    def forward(self, 
               x: torch.Tensor, 
               targets: Optional[torch.Tensor] = None, 
               mask: Optional[torch.Tensor] = None,
               sample_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for regression prediction.
        
        Parameters
        ----------
        x : torch.Tensor
            Input representation [batch_size, input_dim]
            
        targets : torch.Tensor, optional
            Target values [batch_size, output_dim] or [batch_size] for single-output regression
            
        mask : torch.Tensor, optional
            Mask indicating which samples have targets for this task
            [batch_size, output_dim], where 1 means target is available
            
        sample_weights : torch.Tensor, optional
            Sample weights for weighted loss calculation [batch_size]
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'loss': Loss for this task (if targets provided)
            - 'predictions': Predicted values
            - 'quantiles': Predicted quantiles (if applicable)
        """
        # Get batch size and device
        batch_size = x.size(0)
        device = x.device
        
        # Compute predictions
        predictions = self.prediction_network(x)
        
        # Process predictions based on regression type
        if self.quantiles is not None:
            # Reshape for quantile regression
            # [batch_size, output_dim * num_quantiles] -> [batch_size, output_dim, num_quantiles]
            n_quantiles = len(self.quantiles)
            quantile_preds = predictions.view(batch_size, self.output_dim, n_quantiles)
            
            # The central prediction is the median (or average if no median)
            median_idx = n_quantiles // 2
            predictions = quantile_preds[:, :, median_idx]
        
        # Initialize loss
        loss = torch.tensor(0.0, device=device)
        
        # If targets are provided, compute the loss
        if targets is not None:
            # Handle missing output dimensions
            if targets.dim() == 1 and self.output_dim > 1:
                targets = targets.unsqueeze(1).expand(-1, self.output_dim)
            
            # Ensure predictions match targets dimension
            if predictions.dim() > 1 and targets.dim() == 1:
                # If predictions are [batch_size, 1] but targets are [batch_size]
                predictions = predictions.squeeze(1)
            
            # Apply mask if provided
            if mask is not None:
                # Expand mask to match targets if needed
                if mask.dim() == 1 and targets.dim() > 1:
                    mask = mask.unsqueeze(1).expand(-1, targets.size(1))
                
                mask = mask.float()
                valid_samples = torch.sum(mask)
            else:
                # All samples are valid
                # Handle the case where targets are 1D
                if targets.dim() == 1:
                    valid_samples = float(batch_size)
                else:
                    valid_samples = float(batch_size * targets.size(1))
                
                # Create a mask with the same shape as targets
                mask = torch.ones_like(targets, dtype=torch.float, device=device)
            
            # Apply sample weights if provided
            if sample_weights is not None:
                # Need to expand sample_weights to match the dimensions of mask
                if targets.dim() > 1 and sample_weights.dim() == 1:
                    expanded_weights = sample_weights.unsqueeze(1).expand(-1, targets.size(1))
                    combined_weights = mask * expanded_weights
                else:
                    combined_weights = mask * sample_weights
            else:
                combined_weights = mask
            
            # Skip loss computation if no valid samples
            if valid_samples > 0:
                # Safeguard against NaN values
                targets_safe = torch.nan_to_num(targets, nan=0.0)
                predictions_safe = torch.nan_to_num(predictions, nan=0.0)
                
                # Compute loss based on type
                if self.loss_type == 'quantile':
                    # Quantile regression loss
                    # Handle NaN in quantile predictions
                    quantile_preds_safe = torch.nan_to_num(quantile_preds, nan=0.0)
                    loss = self._compute_quantile_loss(quantile_preds_safe, targets_safe, combined_weights)
                else:
                    # Standard regression loss
                    loss_values = self.loss_fn(predictions_safe, targets_safe)
                    
                    # Apply combined weights and normalize
                    loss = torch.sum(loss_values * combined_weights) / torch.sum(combined_weights)
        
        # Prepare output dictionary
        outputs = {
            'loss': loss,
            'predictions': predictions.squeeze(-1),  # Ensure predictions are [batch_size]
        }
        
        # Add quantile predictions if applicable
        if self.quantiles is not None:
            outputs['quantiles'] = quantile_preds
        
        return outputs
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate regression predictions.
        
        Parameters
        ----------
        x : torch.Tensor
            Input representation [batch_size, input_dim]
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'predictions': Predicted values
            - 'quantiles': Predicted quantiles (if applicable)
        """
        # Compute predictions (no loss)
        outputs = self.forward(x)
        
        # Remove loss from outputs
        outputs.pop('loss', None)
        
        return outputs
    
    def _compute_quantile_loss(self, quantile_preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile regression loss.
        
        Parameters
        ----------
        quantile_preds : torch.Tensor
            Predicted quantiles [batch_size, output_dim, num_quantiles]
            
        targets : torch.Tensor
            Target values [batch_size, output_dim]
            
        weights : torch.Tensor
            Weights for loss calculation (can include both mask and sample weights)
            [batch_size, output_dim]
            
        Returns
        -------
        torch.Tensor
            Quantile loss value (scalar)
        """
        # Quantile (Pinball) Loss:
        # L_q(y, f) = q * max(0, y - f) + (1 - q) * max(0, f - y)
        # where y is the target, f is the predicted quantile, and q is the quantile level.
        batch_size = targets.size(0)
        device = targets.device
        n_quantiles = len(self.quantiles)
        
        # Expand targets for each quantile
        # [batch_size, output_dim] -> [batch_size, output_dim, num_quantiles]
        expanded_targets = targets.unsqueeze(-1).expand(-1, -1, n_quantiles)
        
        # Expand weights for each quantile
        # [batch_size, output_dim] -> [batch_size, output_dim, num_quantiles]
        expanded_weights = weights.unsqueeze(-1).expand(-1, -1, n_quantiles)
        
        # Initialize quantile loss
        q_loss = torch.zeros(batch_size, self.output_dim, n_quantiles, device=device)
        
        # Compute quantile loss for each quantile
        for i, q in enumerate(self.quantiles):
            # Compute errors
            errors = expanded_targets[:, :, i] - quantile_preds[:, :, i]
            
            # Quantile loss: q * error if error > 0, (1-q) * error if error < 0
            # This encourages the prediction to be the q-th quantile of the target distribution
            q_tensor = torch.ones_like(errors, device=device) * q
            q_loss[:, :, i] = torch.max(q_tensor * errors, (q_tensor - 1) * errors)
        
        # Apply weights and normalize
        q_loss = q_loss * expanded_weights
        weight_sum = torch.sum(expanded_weights)
        
        if weight_sum > 0:
            q_loss = torch.sum(q_loss) / weight_sum
        else:
            q_loss = torch.tensor(0.0, device=device)
        
        return q_loss
    
    def compute_metrics(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute evaluation metrics for regression.
        
        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            Model outputs from the forward pass
            
        targets : torch.Tensor
            Ground truth values [batch_size, output_dim]
            
        Returns
        -------
        Dict[str, float]
            Dictionary of metric names and values:
            - 'mse': Mean squared error
            - 'mae': Mean absolute error
            - 'r2': R-squared score
            - 'quantile_loss_{q}': Quantile loss for each quantile (if applicable)
        """
        # Extract predictions and targets
        preds = outputs['predictions'].detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        
        # Ensure compatible shapes
        if targets.ndim == 1 and preds.ndim > 1:
            targets = targets.reshape(-1, 1)
        
        # Compute metrics
        metrics = {}
        
        # Mean squared error
        metrics['mse'] = np.mean((preds - targets) ** 2)
        
        # Mean absolute error
        metrics['mae'] = np.mean(np.abs(preds - targets))
        
        # R-squared score
        ss_tot = np.sum((targets - np.mean(targets, axis=0)) ** 2)
        ss_res = np.sum((targets - preds) ** 2)
        
        if ss_tot > 0:
            metrics['r2'] = 1 - (ss_res / ss_tot)
        else:
            metrics['r2'] = 0.0
        
        # Quantile metrics if applicable
        if self.quantiles is not None and 'quantiles' in outputs:
            q_preds = outputs['quantiles'].detach().cpu().numpy()
            
            for i, q in enumerate(self.quantiles):
                # Extract predictions for this quantile
                q_pred = q_preds[:, :, i]
                
                # Compute quantile loss
                errors = targets - q_pred
                q_loss = np.mean(np.maximum(q * errors, (q - 1) * errors))
                
                metrics[f'quantile_loss_{q}'] = q_loss
                
            # Compute prediction interval coverage
            if len(self.quantiles) >= 2:
                # Find the lowest and highest quantiles
                min_q = min(self.quantiles)
                max_q = max(self.quantiles)
                
                min_idx = self.quantiles.index(min_q)
                max_idx = self.quantiles.index(max_q)
                
                # Extract predictions for these quantiles
                lower_bound = q_preds[:, :, min_idx]
                upper_bound = q_preds[:, :, max_idx]
                
                # Compute coverage (percentage of targets within bounds)
                coverage = np.mean((targets >= lower_bound) & (targets <= upper_bound))
                
                metrics['prediction_interval_coverage'] = coverage
        
        return metrics
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration dictionary for serialization.
        
        Returns
        -------
        Dict[str, Any]
            Configuration dictionary
        """
        config = super().get_config()
        config.update({
            'output_dim': self.output_dim,
            'quantiles': self.quantiles,
            'loss_type': self.loss_type
        })
        return config


class CountDataHead(TaskHead):
    """
    Task head for count data prediction.
    
    Predicts count data using Poisson or negative binomial distributions.
    """
    
    def __init__(self,
                name: str,
                input_dim: int,
                distribution: str = 'poisson',
                exposure: bool = False,
                zero_inflated: bool = False,
                dropout: float = 0.1,
                task_weight: float = 1.0):
        """
        Initialize CountDataHead.
        
        Parameters
        ----------
        name : str
            Name of the task
            
        input_dim : int
            Dimension of the input representation
            
        distribution : str, default='poisson'
            Distribution to use ('poisson' or 'negative_binomial')
            
        exposure : bool, default=False
            Whether to model exposure (offset)
            
        zero_inflated : bool, default=False
            Whether to use a zero-inflated model
            
        dropout : float, default=0.1
            Dropout rate for the prediction network
            
        task_weight : float, default=1.0
            Weight of this task in the multi-task loss
        """
        super().__init__(name, input_dim, task_weight)

        # --- Input Validation ---
        assert isinstance(name, str) and name, "'name' must be a non-empty string"
        assert isinstance(input_dim, int) and input_dim > 0, "'input_dim' must be a positive integer"
        assert distribution in ['poisson', 'negative_binomial'], "distribution must be 'poisson' or 'negative_binomial'"
        assert isinstance(exposure, bool), "'exposure' must be a boolean"
        assert isinstance(zero_inflated, bool), "'zero_inflated' must be a boolean"
        assert isinstance(dropout, float) and 0.0 <= dropout < 1.0, "'dropout' must be a float between 0.0 and 1.0"
        assert isinstance(task_weight, float) and task_weight >= 0.0, "'task_weight' must be a non-negative float"
        # --- End Input Validation ---
        
        self.distribution = distribution
        self.exposure = exposure
        self.zero_inflated = zero_inflated
        
        # Determine number of output parameters
        n_outputs = 1  # Rate parameter (lambda) for basic Poisson
        
        if distribution == 'negative_binomial':
            n_outputs = 2  # Rate and dispersion parameters
            
        if zero_inflated:
            n_outputs += 1  # Additional parameter for zero-inflation probability
        
        # Architecture
        self.prediction_network = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, n_outputs)
        )
        
    def loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between model outputs and targets.
        
        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            Outputs from the forward pass
            
        targets : torch.Tensor
            Target count values
            
        Returns
        -------
        torch.Tensor
            Loss value
        """
        if 'loss' in outputs:
            return outputs['loss']
            
        # Extract relevant outputs
        rate = outputs['rate']
        
        # Compute negative log-likelihood based on distribution
        epsilon = 1e-7
        rate_safe = torch.clamp(rate, epsilon, 1e6)  # Ensure numeric stability
        
        if self.distribution == 'poisson':
            if self.zero_inflated:
                # Zero-inflated Poisson
                zero_prob = outputs['zero_prob']
                return -torch.mean(self._compute_zip_log_likelihood(rate_safe, zero_prob, targets))
            else:
                # Standard Poisson
                return -torch.mean(self._compute_poisson_log_likelihood(rate_safe, targets))
        else:  # Negative binomial
            dispersion = outputs['dispersion']
            dispersion_safe = torch.clamp(dispersion, epsilon, 1e6)
            
            if self.zero_inflated:
                # Zero-inflated negative binomial
                zero_prob = outputs['zero_prob']
                return -torch.mean(self._compute_zinb_log_likelihood(
                    rate_safe, dispersion_safe, zero_prob, targets))
            else:
                # Standard negative binomial
                return -torch.mean(self._compute_nb_log_likelihood(
                    rate_safe, dispersion_safe, targets))
        
    def forward(self, 
               x: torch.Tensor, 
               targets: Optional[torch.Tensor] = None, 
               mask: Optional[torch.Tensor] = None,
               exposure: Optional[torch.Tensor] = None,
               sample_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for count data prediction.
        
        Parameters
        ----------
        x : torch.Tensor
            Input representation [batch_size, input_dim]
            
        targets : torch.Tensor, optional
            Target count values [batch_size]
            
        mask : torch.Tensor, optional
            Mask indicating which samples have targets for this task
            [batch_size], where 1 means target is available
            
        exposure : torch.Tensor, optional
            Exposure values for rate adjustment [batch_size]
            
        sample_weights : torch.Tensor, optional
            Sample weights for weighted loss calculation [batch_size]
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'loss': Loss for this task (if targets provided)
            - 'rate': Predicted rate parameter
            - 'mean': Predicted mean count
            - 'dispersion': Predicted dispersion parameter (if applicable)
            - 'zero_prob': Predicted zero-inflation probability (if applicable)
        """
        # Get batch size and device
        batch_size = x.size(0)
        device = x.device
        
        # Compute raw predictions
        outputs = self.prediction_network(x)
        
        # Process outputs based on model type
        if self.distribution == 'poisson':
            if self.zero_inflated:
                # Zero-inflated Poisson
                rate = F.softplus(outputs[:, 0])  # Rate parameter (lambda) > 0
                zero_prob = torch.sigmoid(outputs[:, 1])  # Probability of structural zero
                
                # Predicted mean: (1 - zero_prob) * rate
                mean = (1 - zero_prob) * rate
                
                # Results dictionary
                results = {
                    'rate': rate,
                    'zero_prob': zero_prob,
                    'mean': mean,
                    'predictions': mean  # Add predictions key for compatibility
                }
            else:
                # Standard Poisson
                rate = F.softplus(outputs)  # Rate parameter (lambda) > 0
                
                # Apply exposure adjustment if needed
                if self.exposure and exposure is not None:
                    rate = rate * exposure
                
                # Predicted mean equals rate for Poisson
                mean = rate
                
                # Results dictionary
                results = {
                    'rate': rate.squeeze(-1),
                    'mean': mean.squeeze(-1),
                    'predictions': mean.squeeze(-1)  # Add predictions key for compatibility
                }
        else:  # Negative binomial
            if self.zero_inflated:
                # Zero-inflated negative binomial
                rate = F.softplus(outputs[:, 0])  # Rate parameter > 0
                dispersion = F.softplus(outputs[:, 1]) + 1e-6  # Dispersion parameter > 0
                zero_prob = torch.sigmoid(outputs[:, 2])  # Probability of structural zero
                
                # Predicted mean: (1 - zero_prob) * rate
                mean = (1 - zero_prob) * rate
                
                # Results dictionary
                results = {
                    'rate': rate,
                    'dispersion': dispersion,
                    'zero_prob': zero_prob,
                    'mean': mean,
                    'predictions': mean  # Add predictions key for compatibility
                }
            else:
                # Standard negative binomial
                rate = F.softplus(outputs[:, 0:1])  # Rate parameter > 0
                dispersion = F.softplus(outputs[:, 1:2]) + 1e-6  # Dispersion parameter > 0
                
                # Apply exposure adjustment if needed
                if self.exposure and exposure is not None:
                    rate = rate * exposure.unsqueeze(-1)
                
                # Predicted mean
                mean = rate
                
                # Results dictionary
                results = {
                    'rate': rate.squeeze(-1),
                    'dispersion': dispersion.squeeze(-1),
                    'mean': mean.squeeze(-1),
                    'predictions': mean.squeeze(-1)  # Add predictions key for compatibility
                }
        
        # Initialize loss
        loss = torch.tensor(0.0, device=device)
        
        # If targets are provided, compute the loss
        if targets is not None:
            # Apply mask if provided
            if mask is not None:
                mask = mask.float()
                valid_samples = torch.sum(mask)
            else:
                # All samples are valid
                valid_samples = float(batch_size)
                mask = torch.ones_like(targets, dtype=torch.float, device=device)
            
            # Apply sample weights if provided
            if sample_weights is not None:
                # Combine with mask
                combined_weights = mask * sample_weights
            else:
                combined_weights = mask
            
            # Skip loss computation if no valid samples
            if valid_samples > 0:
                # Compute negative log-likelihood based on distribution
                if self.distribution == 'poisson':
                    if self.zero_inflated:
                        # Zero-inflated Poisson log-likelihood
                        ll = self._compute_zip_log_likelihood(rate, zero_prob, targets)
                    else:
                        # Standard Poisson log-likelihood
                        ll = self._compute_poisson_log_likelihood(rate.squeeze(-1), targets)
                else:  # Negative binomial
                    if self.zero_inflated:
                        # Zero-inflated negative binomial log-likelihood
                        ll = self._compute_zinb_log_likelihood(rate, dispersion, zero_prob, targets)
                    else:
                        # Standard negative binomial log-likelihood
                        ll = self._compute_nb_log_likelihood(rate.squeeze(-1), dispersion.squeeze(-1), targets)
                
                # Apply combined weights and compute weighted negative log-likelihood
                nll = -torch.sum(ll * combined_weights) / torch.sum(combined_weights)
                loss = nll
        
        # Add loss to results
        results['loss'] = loss
        
        return results
    
    def predict(self, x: torch.Tensor, exposure: Optional[torch.Tensor] = None, sample_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Generate count data predictions.
        
        Parameters
        ----------
        x : torch.Tensor
            Input representation [batch_size, input_dim]
            
        exposure : torch.Tensor, optional
            Exposure values for rate adjustment [batch_size]
            
        sample_weights : torch.Tensor, optional
            Sample weights [batch_size]
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing model predictions
        """
        # Compute predictions (no loss)
        outputs = self.forward(x, exposure=exposure, sample_weights=sample_weights)
        
        # Remove loss from outputs
        outputs.pop('loss', None)
        
        return outputs
    
    def _compute_poisson_log_likelihood(self, rate: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Poisson log-likelihood.
        
        Parameters
        ----------
        rate : torch.Tensor
            Predicted rate parameter (lambda) [batch_size]
            
        targets : torch.Tensor
            Target count values [batch_size]
            
        Returns
        -------
        torch.Tensor
            Log-likelihood values [batch_size]
        """
        # Poisson log-likelihood: log(P(Y=y | lambda)) = y * log(lambda) - lambda - log(y!)
        # We omit the log(y!) term as it's constant w.r.t. lambda during optimization.
        # ll = y * log(lambda) - lambda
        log_rate = torch.log(rate + 1e-10)
        ll = targets * log_rate - rate
        
        return ll
    
    def _compute_zip_log_likelihood(self, rate: torch.Tensor, zero_prob: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute zero-inflated Poisson log-likelihood.
        
        Parameters
        ----------
        rate : torch.Tensor
            Predicted rate parameter (lambda) [batch_size]
            
        zero_prob : torch.Tensor
            Predicted probability of structural zero [batch_size]
            
        targets : torch.Tensor
            Target count values [batch_size]
            
        Returns
        -------
        torch.Tensor
            Log-likelihood values [batch_size]
        """
        # Zero-Inflated Poisson (ZIP) log-likelihood:
        # log(P(Y=y | lambda, pi)) = 
        #   I(y=0) * log(pi + (1-pi)*exp(-lambda)) + 
        #   I(y>0) * log((1-pi) * Poisson(y | lambda))
        # where pi is zero_prob.
        # log(P(Y=y | lambda, pi)) = 
        #   I(y=0) * log(pi + (1-pi)*exp(-lambda)) + 
        #   I(y>0) * [log(1-pi) + y*log(lambda) - lambda - log(y!)]
        # We omit log(y!) term.
        
        zeros = (targets == 0)
        
        # Initialize log-likelihood
        ll = torch.zeros_like(targets, dtype=torch.float, device=targets.device)
        
        # For zeros: log(zero_prob + (1 - zero_prob) * exp(-rate))
        # Compute in log space to avoid numerical issues
        poisson_zeros = torch.exp(-rate)
        ll[zeros] = torch.log(zero_prob[zeros] + (1 - zero_prob[zeros]) * poisson_zeros[zeros] + 1e-10)
        
        # For non-zeros: log((1 - zero_prob) * Poisson(y | rate))
        log_rate = torch.log(rate + 1e-10)
        poisson_ll = targets * log_rate - rate
        ll[~zeros] = torch.log(1 - zero_prob[~zeros] + 1e-10) + poisson_ll[~zeros]
        
        return ll
    
    def _compute_nb_log_likelihood(self, rate: torch.Tensor, dispersion: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute negative binomial log-likelihood.
        
        Parameters
        ----------
        rate : torch.Tensor
            Predicted rate parameter [batch_size]
            
        dispersion : torch.Tensor
            Predicted dispersion parameter [batch_size]
            
        targets : torch.Tensor
            Target count values [batch_size]
            
        Returns
        -------
        torch.Tensor
            Log-likelihood values [batch_size]
        """
        # Negative Binomial (NB) log-likelihood:
        # Using parameterization with mean mu=rate and dispersion alpha=dispersion (where variance = mu + alpha*mu^2)
        # P(Y=y | mu, alpha) = Gamma(y + 1/alpha) / (Gamma(y+1) * Gamma(1/alpha)) * (1/(1+alpha*mu))^(1/alpha) * (alpha*mu/(1+alpha*mu))^y
        # Let r = 1/alpha = 1/dispersion
        # P(Y=y | mu, r) = Gamma(y + r) / (Gamma(y+1) * Gamma(r)) * (r/(r+mu))^r * (mu/(r+mu))^y
        # log(P) = lgamma(y+r) - lgamma(y+1) - lgamma(r) + r*log(r) - r*log(r+mu) + y*log(mu) - y*log(r+mu)
        # log(P) = lgamma(y+r) - lgamma(y+1) - lgamma(r) + r*log(r) + y*log(mu) - (y+r)*log(r+mu)
        
        # Convert to torch float for calculations
        y = targets.float()
        
        # We use the parameterization where dispersion = 1/r (inverse of number of failures)
        r = 1.0 / dispersion
        
        # Compute log-likelihood using the formula derived above
        log_mu_term = y * torch.log(rate + 1e-10)
        log_r_term = r * torch.log(r + 1e-10)
        log_neg_binomial = torch.lgamma(y + r) - torch.lgamma(y + 1) - torch.lgamma(r) + log_r_term - (y + r) * torch.log(rate + r + 1e-10) + log_mu_term
        
        return log_neg_binomial
    
    def _compute_zinb_log_likelihood(self, rate: torch.Tensor, dispersion: torch.Tensor, zero_prob: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute zero-inflated negative binomial log-likelihood.
        
        Parameters
        ----------
        rate : torch.Tensor
            Predicted rate parameter [batch_size]
            
        dispersion : torch.Tensor
            Predicted dispersion parameter [batch_size]
            
        zero_prob : torch.Tensor
            Predicted probability of structural zero [batch_size]
            
        targets : torch.Tensor
            Target count values [batch_size]
            
        Returns
        -------
        torch.Tensor
            Log-likelihood values [batch_size]
        """
        # Zero-Inflated Negative Binomial (ZINB) log-likelihood:
        # log(P(Y=y | mu, alpha, pi)) = 
        #   I(y=0) * log(pi + (1-pi)*NB(0 | mu, alpha)) + 
        #   I(y>0) * log((1-pi) * NB(y | mu, alpha))
        # where pi is zero_prob, mu is rate, alpha is dispersion.
        # log(P) = 
        #   I(y=0) * log(pi + (1-pi)*(r/(r+mu))^r) + 
        #   I(y>0) * [log(1-pi) + log(NB(y | mu, r))] 
        # where r = 1/alpha.
        
        zeros = (targets == 0)
        
        # Initialize log-likelihood
        ll = torch.zeros_like(targets, dtype=torch.float, device=targets.device)
        
        # Convert parameters for NB PMF calculation
        r = 1.0 / dispersion
        nb_zero_prob = torch.pow(r / (r + rate), r)
        
        # For zeros: log(zero_prob + (1 - zero_prob) * NB(0 | rate, dispersion))
        # Compute in log space to avoid numerical issues
        ll[zeros] = torch.log(zero_prob[zeros] + (1 - zero_prob[zeros]) * nb_zero_prob[zeros] + 1e-10)
        
        # For non-zeros: log((1 - zero_prob) * NB(y | rate, dispersion))
        # Compute standard NB log-likelihood
        y = targets.float()
        log_mu_term = y * torch.log(rate + 1e-10)
        log_r_term = r * torch.log(r + 1e-10)
        log_nb = torch.lgamma(y + r) - torch.lgamma(y + 1) - torch.lgamma(r) + log_r_term - (y + r) * torch.log(rate + r + 1e-10) + log_mu_term
        
        ll[~zeros] = torch.log(1 - zero_prob[~zeros] + 1e-10) + log_nb[~zeros]
        
        return ll
    
    def compute_metrics(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute evaluation metrics for count data prediction.
        
        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            Model outputs from the forward pass
            
        targets : torch.Tensor
            Ground truth count values [batch_size]
            
        Returns
        -------
        Dict[str, float]
            Dictionary of metric names and values:
            - 'mae': Mean absolute error
            - 'mse': Mean squared error
            - 'rmse': Root mean squared error
            - 'mean_deviance': Mean Poisson or NB deviance
            - 'zero_accuracy': Accuracy of zero prediction (if zero-inflated)
        """
        # Extract predictions and targets
        mean_preds = outputs['mean'].detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        
        # Compute metrics
        metrics = {}
        
        # Mean absolute error
        metrics['mae'] = np.mean(np.abs(mean_preds - targets))
        
        # Mean squared error
        metrics['mse'] = np.mean((mean_preds - targets) ** 2)
        
        # Root mean squared error
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Deviance (based on distribution)
        if self.distribution == 'poisson':
            # Poisson deviance
            deviance = 2 * np.sum(
                targets * np.log((targets + 1e-10) / (mean_preds + 1e-10)) - (targets - mean_preds)
            )
            metrics['mean_deviance'] = deviance / len(targets)
        else:
            # Negative binomial deviance requires dispersion parameter
            dispersion = outputs['dispersion'].detach().cpu().numpy()
            r = 1.0 / dispersion
            
            # Compute deviance term for each sample
            dev_terms = 2 * (
                targets * np.log((targets + 1e-10) / (mean_preds + 1e-10)) - 
                (targets + r) * np.log((targets + r + 1e-10) / (mean_preds + r + 1e-10))
            )
            
            metrics['mean_deviance'] = np.mean(dev_terms)
        
        # Zero prediction metrics (for zero-inflated models)
        if self.zero_inflated and 'zero_prob' in outputs:
            zero_probs = outputs['zero_prob'].detach().cpu().numpy()
            is_zero = targets == 0
            
            # Compute zero prediction accuracy
            zero_preds = zero_probs > 0.5
            metrics['zero_accuracy'] = np.mean(zero_preds == is_zero)
            
            # Additional metrics for zero-inflation
            metrics['zero_precision'] = np.sum(zero_preds & is_zero) / (np.sum(zero_preds) + 1e-10)
            metrics['zero_recall'] = np.sum(zero_preds & is_zero) / (np.sum(is_zero) + 1e-10)
        
        return metrics
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration dictionary for serialization.
        
        Returns
        -------
        Dict[str, Any]
            Configuration dictionary
        """
        config = super().get_config()
        config.update({
            'distribution': self.distribution,
            'exposure': self.exposure,
            'zero_inflated': self.zero_inflated
        })
        return config
