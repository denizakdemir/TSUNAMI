import pytest
import torch
import numpy as np
from enhanced_deephit.models.tasks.standard import ClassificationHead, RegressionHead, CountDataHead

def test_classification_head_binary():
    """Test binary classification head"""
    # Initialize head
    config = {
        'name': 'binary_classification',
        'input_dim': 32,
        'num_classes': 1,  # Binary classification
        'task_weight': 1.0
    }
    
    head = ClassificationHead(**config)
    
    # Create dummy input
    batch_size = 8
    x = torch.randn(batch_size, config['input_dim'])
    
    # Forward pass (prediction mode)
    outputs = head(x)
    
    # Check output shapes
    assert 'predictions' in outputs
    assert 'probabilities' in outputs
    # For binary classification, outputs shape can be [batch] or [batch, 1]
    assert (outputs['predictions'].shape == torch.Size([batch_size]) or 
            outputs['predictions'].shape == torch.Size([batch_size, 1]))
    assert (outputs['probabilities'].shape == torch.Size([batch_size]) or 
            outputs['probabilities'].shape == torch.Size([batch_size, 1]))
    
    # Check probability properties
    probas = outputs['probabilities']
    assert torch.all(probas >= 0)
    assert torch.all(probas <= 1)
    
    # Test loss calculation
    targets = torch.randint(0, 2, (batch_size, 1)).float()
    outputs_with_loss = head(x, targets)
    
    # Check loss
    assert 'loss' in outputs_with_loss
    assert outputs_with_loss['loss'].numel() == 1
    assert not torch.isnan(outputs_with_loss['loss']).any()

def test_classification_head_multiclass():
    """Test multiclass classification head"""
    # Initialize head
    config = {
        'name': 'multiclass',
        'input_dim': 32,
        'num_classes': 4,  # Multiclass classification
        'task_weight': 1.0
    }
    
    head = ClassificationHead(**config)
    
    # Create dummy input
    batch_size = 8
    x = torch.randn(batch_size, config['input_dim'])
    
    # Forward pass (prediction mode)
    outputs = head(x)
    
    # Check output shapes
    assert 'predictions' in outputs
    assert 'probabilities' in outputs
    assert outputs['predictions'].shape == (batch_size,)  # Class indices
    assert outputs['probabilities'].shape == (batch_size, config['num_classes'])
    
    # Check probability properties
    probas = outputs['probabilities']
    assert torch.all(probas >= 0)
    assert torch.all(probas <= 1)
    assert torch.allclose(torch.sum(probas, dim=1), torch.ones(batch_size))
    
    # Test loss calculation
    targets = torch.randint(0, config['num_classes'], (batch_size,))
    outputs_with_loss = head(x, targets)
    
    # Check loss
    assert 'loss' in outputs_with_loss
    assert outputs_with_loss['loss'].numel() == 1
    assert not torch.isnan(outputs_with_loss['loss']).any()

def test_regression_head():
    """Test regression head"""
    # Initialize head
    config = {
        'name': 'regression',
        'input_dim': 32,
        'output_dim': 2,  # Multiple regression outputs
        'task_weight': 1.0
    }
    
    head = RegressionHead(**config)
    
    # Create dummy input
    batch_size = 8
    x = torch.randn(batch_size, config['input_dim'])
    
    # Forward pass (prediction mode)
    outputs = head(x)
    
    # Check output shapes
    assert 'predictions' in outputs
    assert outputs['predictions'].shape == (batch_size, config['output_dim'])
    
    # Test loss calculation
    targets = torch.randn(batch_size, config['output_dim'])
    outputs_with_loss = head(x, targets)
    
    # Check loss
    assert 'loss' in outputs_with_loss
    assert outputs_with_loss['loss'].numel() == 1
    assert not torch.isnan(outputs_with_loss['loss']).any()
    
    # Test with mask
    mask = torch.ones(batch_size)
    mask[0] = 0  # Mask first sample
    outputs_masked = head(x, targets, mask)
    
    # Check that masked loss is different
    assert outputs_masked['loss'] != outputs_with_loss['loss']

def test_count_data_head():
    """Test count data head"""
    # Initialize head for Poisson regression
    config = {
        'name': 'count',
        'input_dim': 32,
        'distribution': 'poisson',
        'task_weight': 1.0
    }
    
    head = CountDataHead(**config)
    
    # Create dummy input
    batch_size = 8
    x = torch.randn(batch_size, config['input_dim'])
    
    # Forward pass (prediction mode)
    outputs = head(x)
    
    # Check output shapes for the CountDataHead
    assert 'mean' in outputs  # CountDataHead returns 'mean' instead of 'predictions'
    assert 'rate' in outputs
    assert outputs['mean'].shape == torch.Size([batch_size])
    assert outputs['rate'].shape == torch.Size([batch_size])
    
    # Rates should be positive
    assert torch.all(outputs['rate'] > 0)
    
    # Test loss calculation with Poisson data
    targets = torch.randint(0, 10, (batch_size,)).float()
    outputs_with_loss = head(x, targets)
    
    # Check loss
    assert 'loss' in outputs_with_loss
    assert outputs_with_loss['loss'].numel() == 1
    assert not torch.isnan(outputs_with_loss['loss']).any()
    
    # Test with negative binomial distribution
    config['distribution'] = 'negative_binomial'
    head_nb = CountDataHead(**config)
    outputs_nb = head_nb(x)
    
    # Should have dispersion parameter
    assert 'dispersion' in outputs_nb
    assert outputs_nb['dispersion'].shape == torch.Size([batch_size])
    assert torch.all(outputs_nb['dispersion'] > 0)