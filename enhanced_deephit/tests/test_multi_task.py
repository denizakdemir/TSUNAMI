import pytest
import torch
import numpy as np
from enhanced_deephit.models.tasks.survival import SingleRiskHead
from enhanced_deephit.models.tasks.standard import ClassificationHead, RegressionHead
from enhanced_deephit.models.tasks.base import MultiTaskManager
from enhanced_deephit.models import EnhancedDeepHit

def test_multi_task_manager():
    """Test multi-task manager with multiple task heads"""
    # Create task heads
    heads = [
        SingleRiskHead(
            name='survival',
            input_dim=64,
            num_time_bins=10,
            task_weight=1.0
        ),
        ClassificationHead(
            name='classification',
            input_dim=64,
            num_classes=3,
            task_weight=0.5
        ),
        RegressionHead(
            name='regression',
            input_dim=64,
            output_dim=1,
            task_weight=0.3
        )
    ]
    
    # Initialize multi-task manager
    manager = MultiTaskManager(
        encoder_dim=64,
        task_heads=heads
    )
    
    # Create dummy inputs
    batch_size = 4
    x = torch.randn(batch_size, 64)
    
    # Create dummy targets
    targets = {
        'survival': torch.zeros(batch_size, 2 + 10),  # [event, time, one_hot]
        'classification': torch.randint(0, 3, (batch_size,)),
        'regression': torch.randn(batch_size, 1)
    }
    
    # Add events to survival targets
    targets['survival'][0, 0] = 1  # Event
    targets['survival'][0, 1] = 5  # Time
    targets['survival'][0, 2 + 5] = 1  # One-hot
    
    # Forward pass
    outputs = manager(x, targets)
    
    # Check outputs
    assert 'loss' in outputs
    assert 'task_losses' in outputs
    assert 'task_outputs' in outputs
    
    # Check task losses
    assert 'survival' in outputs['task_losses']
    assert 'classification' in outputs['task_losses']
    assert 'regression' in outputs['task_losses']
    
    # Check weighted sum of losses equals total loss
    # Note: The task_losses already include the task weights in the MultiTaskManager
    # So don't multiply by weights again in the test
    task_loss_sum = sum([
        outputs['task_losses']['survival'],
        outputs['task_losses']['classification'],
        outputs['task_losses']['regression']
    ])
    # Use a larger tolerance for floating point comparison
    assert torch.isclose(outputs['loss'], task_loss_sum, rtol=1e-4, atol=1e-4)
    
    # Test get_task method
    survival_head = manager.get_task('survival')
    assert survival_head is not None
    assert survival_head.name == 'survival'
    
    # Test task masking
    masks = {
        'survival': torch.ones(batch_size),
        'classification': torch.ones(batch_size),
        'regression': torch.ones(batch_size)
    }
    
    # Mask out one sample for each task
    masks['survival'][1] = 0
    masks['classification'][2] = 0
    masks['regression'][3] = 0
    
    outputs_masked = manager(x, targets, masks)
    
    # Masked loss should be different
    assert outputs_masked['loss'] != outputs['loss']

def test_model_integration():
    """Test full model integration with multiple tasks"""
    # Define task heads
    heads = [
        SingleRiskHead(
            name='survival',
            input_dim=32,
            num_time_bins=5,
            task_weight=1.0
        ),
        ClassificationHead(
            name='classification',
            input_dim=32,
            num_classes=2,
            task_weight=0.5
        )
    ]
    
    # Initialize model
    model = EnhancedDeepHit(
        num_continuous=10,
        targets=heads,
        encoder_dim=32,
        encoder_depth=2,
        encoder_heads=4
    )
    
    # Create dummy batch
    batch_size = 6
    continuous = torch.randn(batch_size, 10)
    
    # Create dummy targets
    targets = {
        'survival': torch.zeros(batch_size, 2 + 5),  # [event, time, one_hot]
        'classification': torch.randint(0, 2, (batch_size,))
    }
    
    # Add events to survival targets
    targets['survival'][0, 0] = 1  # Event
    targets['survival'][0, 1] = 2  # Time
    targets['survival'][0, 2 + 2] = 1  # One-hot
    
    targets['survival'][1, 0] = 1  # Event
    targets['survival'][1, 1] = 4  # Time
    targets['survival'][1, 2 + 4] = 1  # One-hot
    
    # Forward pass with targets (training)
    outputs = model(continuous, targets)
    
    # Check outputs
    assert 'loss' in outputs
    assert 'task_losses' in outputs
    assert 'task_outputs' in outputs
    assert 'encoder_output' in outputs
    assert 'attention_maps' in outputs
    
    # Check shapes
    assert outputs['encoder_output'].shape == (batch_size, 32)
    assert len(outputs['attention_maps']) == 2  # One map per layer
    
    # Forward pass without targets (prediction)
    predictions = model.predict(continuous)
    
    # Check predictions
    assert 'task_outputs' in predictions
    assert 'survival' in predictions['task_outputs']
    assert 'classification' in predictions['task_outputs']
    
    # Check survival outputs
    survival_outputs = predictions['task_outputs']['survival']
    assert 'hazard' in survival_outputs
    assert 'survival' in survival_outputs
    assert 'risk_score' in survival_outputs
    
    # Check classification outputs
    classification_outputs = predictions['task_outputs']['classification']
    assert 'predictions' in classification_outputs
    assert 'probabilities' in classification_outputs
    
    # Test uncertainty quantification
    uncertainty = model.compute_uncertainty(continuous, num_samples=3)
    
    # Check uncertainty outputs
    assert 'survival' in uncertainty
    assert 'classification' in uncertainty
    assert 'mean' in uncertainty['survival']
    assert 'std' in uncertainty['survival']
    assert 'samples' in uncertainty['survival']