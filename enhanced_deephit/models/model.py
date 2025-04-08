import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import os
import json
import logging
from pathlib import Path

from enhanced_deephit.models.encoder import TabularTransformer
from enhanced_deephit.models.tasks.base import MultiTaskManager, TaskHead
from enhanced_deephit.data.processing import DataProcessor, build_category_info


class EnhancedDeepHit(nn.Module):
    """
    Enhanced DeepHit model for multi-task tabular survival analysis.
    
    This model extends the original DeepHit approach to support:
    - Multiple target types (survival, competing risks, classification, regression)
    - Tabular transformer architecture with categorical and continuous features
    - Missing data handling
    - Masked loss for incomplete targets
    - Variational methods for uncertainty quantification
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
        Initialize EnhancedDeepHit model.
        
        Parameters
        ----------
        num_continuous : int
            Number of continuous features
            
        targets : List[TaskHead]
            List of task-specific heads
            
        cat_feat_info : List[Dict], optional
            List of dictionaries with categorical feature information
            
        encoder_dim : int, default=128
            Dimension of the encoder representation
            
        encoder_depth : int, default=4
            Number of transformer layers in the encoder
            
        encoder_heads : int, default=8
            Number of attention heads in the encoder
            
        encoder_ff_dim : int, default=512
            Feed-forward hidden dimension in the encoder
            
        encoder_dropout : float, default=0.1
            Dropout rate in the encoder
            
        encoder_feature_interaction : bool, default=True
            Whether to use explicit feature interaction in the encoder
            
        include_variational : bool, default=False
            Whether to include variational component for uncertainty
            
        variational_beta : float, default=0.1
            Weight of KL divergence in variational loss
            
        device : str, default='cpu'
            Device to use for computation ('cpu' or 'cuda')
        """
        super().__init__()
        
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
        Forward pass through the EnhancedDeepHit model.
        
        Parameters
        ----------
        continuous : torch.Tensor
            Continuous features [batch_size, num_continuous]
            
        targets : Dict[str, torch.Tensor], optional
            Dictionary mapping task names to target tensors
            
        masks : Dict[str, torch.Tensor], optional
            Dictionary mapping task names to mask tensors
            
        categorical : torch.Tensor, optional
            Categorical features [batch_size, num_categorical]
            
        missing_mask : torch.Tensor, optional
            Missing value mask [batch_size, num_features]
            
        sample_weights : torch.Tensor, optional
            Sample weights for weighted loss calculation [batch_size]
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'loss': Combined loss if targets provided
            - 'task_losses': Task-specific losses if targets provided
            - 'task_outputs': Task-specific outputs
            - 'encoder_output': Encoder representation
            - 'attention_maps': Attention maps from encoder
        """
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
        Generate predictions from the model.
        
        Parameters
        ----------
        continuous : torch.Tensor
            Continuous features [batch_size, num_continuous]
            
        categorical : torch.Tensor, optional
            Categorical features [batch_size, num_categorical]
            
        missing_mask : torch.Tensor, optional
            Missing value mask [batch_size, num_features]
            
        sample_weights : torch.Tensor, optional
            Sample weights [batch_size]
            
        return_representations : bool, default=False
            Whether to return encoder representations
            
        return_attention : bool, default=False
            Whether to return attention maps
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing task predictions
        """
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
        Train the model.
        
        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader for training data
            
        val_loader : torch.utils.data.DataLoader, optional
            DataLoader for validation data
            
        learning_rate : float, default=1e-3
            Learning rate for optimizer
            
        weight_decay : float, default=1e-4
            Weight decay for optimizer
            
        num_epochs : int, default=100
            Maximum number of training epochs
            
        patience : int, default=10
            Patience for early stopping
            
        optimize_metric : str, optional
            Metric to optimize for early stopping
            
        callbacks : List[Any], optional
            Callbacks for training process
            
        use_sample_weights : bool, default=False
            Whether to use sample weights from the batches (should be provided in the 'sample_weights' key)
            
        Returns
        -------
        Dict[str, List[float]]
            Training history
        """
        # Set model to training mode
        self.train()
        
        # Initialize optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
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
                
                # Backward pass
                loss.backward()
                
                # Update weights
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
        Evaluate the model on a dataset.
        
        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader for evaluation data
            
        use_sample_weights : bool, default=False
            Whether to use sample weights from the batches
            
        Returns
        -------
        Tuple[float, Dict[str, float]]
            Tuple containing:
            - Average loss
            - Dictionary of evaluation metrics
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
    
    def save(self, path: str, save_processor: bool = True, processor: Optional[DataProcessor] = None):
        """
        Save the model to disk.
        
        Parameters
        ----------
        path : str
            Path to save the model
            
        save_processor : bool, default=True
            Whether to save the data processor
            
        processor : DataProcessor, optional
            Data processor to save
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model parameters
        torch.save(self.state_dict(), f"{path}.pt")
        
        # Save model configuration
        config = {
            'num_continuous': self.num_continuous,
            'encoder_dim': self.encoder_dim,
            'include_variational': self.include_variational,
            'cat_feat_info': self.cat_feat_info,
            'task_config': [task.get_config() for task in self.task_manager.task_heads]
        }
        
        with open(f"{path}.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save data processor if provided
        if save_processor and processor is not None:
            import pickle
            with open(f"{path}_processor.pkl", 'wb') as f:
                pickle.dump(processor, f)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu', load_processor: bool = True) -> Tuple['EnhancedDeepHit', Optional[DataProcessor]]:
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
            
        Returns
        -------
        Tuple[EnhancedDeepHit, Optional[DataProcessor]]
            Tuple containing:
            - Loaded model
            - Data processor (if available)
        """
        # Load model configuration
        with open(f"{path}.json", 'r') as f:
            config = json.load(f)
        
        # Load task configurations
        from enhanced_deephit.models.tasks.survival import SingleRiskHead, CompetingRisksHead
        from enhanced_deephit.models.tasks.standard import ClassificationHead, RegressionHead, CountDataHead
        
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
            num_continuous=config['num_continuous'],
            targets=tasks,
            cat_feat_info=config.get('cat_feat_info'),
            encoder_dim=config['encoder_dim'],
            include_variational=config.get('include_variational', False),
            device=device
        )
        
        # Load model parameters
        model.load_state_dict(torch.load(f"{path}.pt", map_location=device))
        
        # Load data processor if requested
        processor = None
        if load_processor and os.path.exists(f"{path}_processor.pkl"):
            import pickle
            with open(f"{path}_processor.pkl", 'rb') as f:
                processor = pickle.load(f)
        
        return model, processor