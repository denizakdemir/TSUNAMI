import pytest
import torch
import numpy as np
from enhanced_deephit.models.tasks.survival import SingleRiskHead
# CompetingRisksHead will be implemented in a future update

def test_single_risk_head_forward():
    """Test forward pass for SingleRiskHead"""
    # Initialize head
    config = {
        'name': 'survival',
        'input_dim': 64,
        'num_time_bins': 10,
        'alpha_rank': 0.1,
        'alpha_calibration': 0.1
    }
    
    head = SingleRiskHead(**config)
    
    # Create dummy input
    batch_size = 8
    x = torch.randn(batch_size, config['input_dim'])
    
    # Forward pass (prediction mode)
    outputs = head(x)
    
    # Check output shapes
    assert 'hazard' in outputs
    assert 'survival' in outputs
    assert 'risk_score' in outputs
    assert outputs['hazard'].shape == (batch_size, config['num_time_bins'])
    assert outputs['survival'].shape == (batch_size, config['num_time_bins'])
    assert outputs['risk_score'].shape == (batch_size,)
    
    # Check survival curve properties
    survival_curves = outputs['survival']
    
    # Survival curve should be monotonically decreasing
    assert torch.all(survival_curves[:, :-1] >= survival_curves[:, 1:])
    
    # Survival probabilities should be between 0 and 1
    assert torch.all(survival_curves >= 0)
    assert torch.all(survival_curves <= 1)
    
    # First time point should have highest probability
    assert torch.all(survival_curves[:, 0] >= survival_curves[:, -1])

def test_single_risk_head_loss():
    """Test loss calculation for SingleRiskHead"""
    # Initialize head
    config = {
        'name': 'survival',
        'input_dim': 32,
        'num_time_bins': 5,
        'alpha_rank': 0.1
    }
    
    head = SingleRiskHead(**config)
    
    # Create dummy input and targets
    batch_size = 4
    x = torch.randn(batch_size, config['input_dim'])
    
    # Create targets with different event patterns
    # Format: [event_indicator, time_bin, one_hot_encoding]
    targets = torch.zeros(batch_size, 2 + config['num_time_bins'])
    
    # Sample 1: Event at time 2
    targets[0, 0] = 1  # Event occurred
    targets[0, 1] = 2  # At time bin 2
    targets[0, 2 + 2] = 1  # One-hot encoding
    
    # Sample 2: Censored at time 3
    targets[1, 0] = 0  # Censored
    targets[1, 1] = 3  # At time bin 3
    targets[1, 2 + 3:] = -1  # Mark as unknown after censoring
    
    # Sample 3: Event at time 0
    targets[2, 0] = 1
    targets[2, 1] = 0
    targets[2, 2] = 1
    
    # Sample 4: Censored at time 4
    targets[3, 0] = 0
    targets[3, 1] = 4
    targets[3, 2 + 4:] = -1
    
    # Forward pass with targets
    outputs = head(x, targets)
    
    # Check loss is computed and is a scalar
    assert 'loss' in outputs
    assert outputs['loss'].numel() == 1
    assert not torch.isnan(outputs['loss']).any()
    assert outputs['loss'] > 0  # Loss should be positive

def test_competing_risks_head_forward():
    """Test forward pass for CompetingRisksHead"""
    # Initialize head
    config = {
        'name': 'competing_risks',
        'input_dim': 64,
        'num_time_bins': 10,
        'num_risks': 3,
        'alpha_rank': 0.1,
        'alpha_calibration': 0.1,
        'use_softmax': True
    }
    
    head = CompetingRisksHead(**config)
    
    # Create dummy input
    batch_size = 8
    x = torch.randn(batch_size, config['input_dim'])
    
    # Forward pass (prediction mode)
    outputs = head(x)
    
    # Check output shapes
    assert 'cause_hazards' in outputs
    assert 'overall_survival' in outputs
    assert 'cif' in outputs
    assert 'risk_scores' in outputs
    
    assert outputs['cause_hazards'].shape == (batch_size, config['num_risks'], config['num_time_bins'])
    assert outputs['overall_survival'].shape == (batch_size, config['num_time_bins'])
    assert outputs['cif'].shape == (batch_size, config['num_risks'], config['num_time_bins'])
    assert outputs['risk_scores'].shape == (batch_size, config['num_risks'])
    
    # Check CIF properties
    cif = outputs['cif']
    
    # CIF should start at 0 for all risks
    assert torch.all(cif[:, :, 0] >= 0)
    assert torch.allclose(torch.sum(cif[:, :, 0], dim=1), 
                         1 - outputs['overall_survival'][:, 0], 
                         atol=1e-5)
    
    # CIF should be monotonically increasing
    for i in range(config['num_risks']):
        assert torch.all(cif[:, i, 1:] >= cif[:, i, :-1])
    
    # CIF probabilities should be between 0 and 1
    assert torch.all(cif >= 0)
    assert torch.all(cif <= 1)
    
    # Sum of CIFs and survival probability should equal 1
    for t in range(config['num_time_bins']):
        sum_cif_survival = torch.sum(cif[:, :, t], dim=1) + outputs['overall_survival'][:, t]
        assert torch.allclose(sum_cif_survival, torch.ones_like(sum_cif_survival), atol=1e-5)

def test_competing_risks_head_loss():
    """Test loss calculation for CompetingRisksHead"""
    # Initialize head
    config = {
        'name': 'competing_risks',
        'input_dim': 32,
        'num_time_bins': 5,
        'num_risks': 2,
        'alpha_rank': 0.1
    }
    
    head = CompetingRisksHead(**config)
    
    # Create dummy input and targets
    batch_size = 4
    x = torch.randn(batch_size, config['input_dim'])
    
    # Create targets with different event patterns
    # Format: [event_indicator, time_bin, cause_index, one_hot_encoding]
    targets = torch.zeros(batch_size, 3 + config['num_risks'] * config['num_time_bins'])
    
    # Sample 1: Event at time 2, cause 0
    targets[0, 0] = 1  # Event occurred
    targets[0, 1] = 2  # At time bin 2
    targets[0, 2] = 0  # Cause 0
    
    # Sample 2: Censored at time 3
    targets[1, 0] = 0  # Censored
    targets[1, 1] = 3  # At time bin 3
    targets[1, 2] = -1  # No cause
    
    # Sample 3: Event at time 1, cause 1
    targets[2, 0] = 1
    targets[2, 1] = 1
    targets[2, 2] = 1
    
    # Sample 4: Event at time 4, cause 0
    targets[3, 0] = 1
    targets[3, 1] = 4
    targets[3, 2] = 0
    
    # Forward pass with targets
    outputs = head(x, targets)
    
    # Check loss is computed and is a scalar
    assert 'loss' in outputs
    assert outputs['loss'].numel() == 1
    assert not torch.isnan(outputs['loss']).any()
    assert outputs['loss'] > 0  # Loss should be positive