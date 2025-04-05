import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from abc import ABC, abstractmethod


class TaskHead(nn.Module, ABC):
    """
    Abstract base class for all task-specific prediction heads.
    
    This class defines the interface that all task-specific heads must implement.
    """
    
    def __init__(self, 
                 name: str, 
                 input_dim: int,
                 task_weight: float = 1.0,
                 **kwargs):
        """
        Initialize a task head.
        
        Parameters
        ----------
        name : str
            Name of the task
            
        input_dim : int
            Dimension of the input representation
            
        task_weight : float, default=1.0
            Weight of this task in the multi-task loss
        """
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.task_weight = task_weight
        
    @abstractmethod
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the task head.
        
        Parameters
        ----------
        x : torch.Tensor
            Input representation [batch_size, input_dim]
            
        targets : torch.Tensor, optional
            Target values for this task
            
        mask : torch.Tensor, optional
            Mask indicating which samples have targets for this task
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'loss': Loss for this task (if targets provided)
            - 'predictions': Model predictions
            - Other task-specific outputs
        """
        pass
    
    @abstractmethod
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate predictions from the task head.
        
        Parameters
        ----------
        x : torch.Tensor
            Input representation [batch_size, input_dim]
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing predictions and additional outputs
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute evaluation metrics for this task.
        
        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            Model outputs from the forward pass
            
        targets : torch.Tensor
            Ground truth targets
            
        Returns
        -------
        Dict[str, float]
            Dictionary of metric names and values
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration dictionary for serialization.
        
        Returns
        -------
        Dict[str, Any]
            Configuration dictionary
        """
        return {
            'name': self.name,
            'task_type': self.__class__.__name__,
            'input_dim': self.input_dim,
            'task_weight': self.task_weight
        }


class MultiTaskManager(nn.Module):
    """
    Manages multiple task heads in a unified multi-task learning framework.
    
    This module coordinates multiple task-specific heads, computes combined
    losses, and handles incomplete target data through masked losses.
    """
    
    def __init__(self, 
                 encoder_dim: int,
                 task_heads: List[TaskHead],
                 include_variational: bool = False,
                 variational_beta: float = 0.1):
        """
        Initialize multi-task manager.
        
        Parameters
        ----------
        encoder_dim : int
            Dimension of the encoder representation
            
        task_heads : List[TaskHead]
            List of task-specific heads
            
        include_variational : bool, default=False
            Whether to include a variational component
            
        variational_beta : float, default=0.1
            Weight of the KL divergence in the variational loss
        """
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.task_heads = nn.ModuleList(task_heads)
        self.include_variational = include_variational
        self.variational_beta = variational_beta
        
        # Create a dictionary of task names for easy access
        self.task_dict = {head.name: head for head in task_heads}
        
        # Variational components if enabled
        if include_variational:
            # Encoder for variational parameters (mean and log variance)
            self.variational_encoder = nn.Linear(encoder_dim, encoder_dim * 2)
            
            # Decoder from latent space back to representation
            self.variational_decoder = nn.Sequential(
                nn.Linear(encoder_dim, encoder_dim),
                nn.LayerNorm(encoder_dim),
                nn.GELU()
            )
    
    def forward(self, 
               x: torch.Tensor, 
               targets: Dict[str, torch.Tensor],
               masks: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Forward pass through all task heads.
        
        Parameters
        ----------
        x : torch.Tensor
            Input representation from the encoder [batch_size, encoder_dim]
            
        targets : Dict[str, torch.Tensor]
            Dictionary mapping task names to target tensors
            
        masks : Dict[str, torch.Tensor], optional
            Dictionary mapping task names to mask tensors
            (1 for samples with targets, 0 for samples without)
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'loss': Combined loss from all tasks
            - 'task_losses': Dictionary of individual task losses
            - 'task_outputs': Dictionary of all task outputs
            - 'variational_loss': KL divergence loss (if enabled)
        """
        batch_size = x.size(0)
        device = x.device
        
        # Apply variational component if enabled
        variational_loss = torch.tensor(0.0, device=device)
        
        if self.include_variational:
            # Compute variational parameters
            var_params = self.variational_encoder(x)
            mean, logvar = var_params.chunk(2, dim=1)
            
            # Sample from the latent distribution using reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + eps * std
            
            # Reconstruct the representation
            x_recon = self.variational_decoder(z)
            
            # For prediction, use a mix of original and reconstructed representation
            x = 0.5 * (x + x_recon)
            
            # Compute KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            variational_loss = self.variational_beta * kl_loss / batch_size
            
        # Pass through each task head and collect losses and outputs
        task_losses = {}
        task_outputs = {}
        
        for task_head in self.task_heads:
            task_name = task_head.name
            task_target = targets.get(task_name, None)
            task_mask = masks.get(task_name, None) if masks else None
            
            # Skip if no targets provided for this task
            if task_target is None:
                continue
                
            # Forward pass through task head
            outputs = task_head(x, task_target, task_mask)
            
            # Extract and scale task loss
            task_loss = outputs.get('loss', torch.tensor(0.0, device=device))
            weighted_loss = task_loss * task_head.task_weight
            
            # Collect results
            task_losses[task_name] = weighted_loss
            task_outputs[task_name] = outputs
            
        # Compute combined loss
        combined_loss = sum(task_losses.values()) + variational_loss
        
        return {
            'loss': combined_loss,
            'task_losses': task_losses,
            'task_outputs': task_outputs,
            'variational_loss': variational_loss
        }
        
    def predict(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate predictions from all task heads.
        
        Parameters
        ----------
        x : torch.Tensor
            Input representation from the encoder [batch_size, encoder_dim]
            
        Returns
        -------
        Dict[str, Dict[str, torch.Tensor]]
            Dictionary mapping task names to their prediction outputs
        """
        # Apply variational component during prediction if enabled
        if self.include_variational:
            # Compute variational parameters
            var_params = self.variational_encoder(x)
            mean, _ = var_params.chunk(2, dim=1)
            
            # For prediction, just use the mean (no sampling)
            z = mean
            
            # Reconstruct the representation
            x_recon = self.variational_decoder(z)
            
            # For prediction, use a mix of original and reconstructed representation
            x = 0.5 * (x + x_recon)
            
        # Generate predictions from each task head
        predictions = {}
        
        for task_head in self.task_heads:
            task_name = task_head.name
            task_preds = task_head.predict(x)
            predictions[task_name] = task_preds
            
        return predictions
    
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
        return self.task_dict.get(name, None)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration dictionary for serialization.
        
        Returns
        -------
        Dict[str, Any]
            Configuration dictionary
        """
        return {
            'encoder_dim': self.encoder_dim,
            'task_heads': [head.get_config() for head in self.task_heads],
            'include_variational': self.include_variational,
            'variational_beta': self.variational_beta
        }