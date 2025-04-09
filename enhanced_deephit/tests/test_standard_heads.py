import pytest
import torch
import numpy as np
from unittest.mock import patch
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


# --- Loss Calculation Verification Tests ---

def test_classification_binary_bce_loss():
    """Verify BCE loss calculation for binary ClassificationHead."""
    config = {'name': 'binary', 'input_dim': 2, 'num_classes': 2} # num_classes=2 for binary
    head = ClassificationHead(**config)

    # Dummy input
    x = torch.randn(2, config['input_dim'])

    # Targets: S0=0, S1=1
    targets = torch.tensor([0., 1.])
    mask = torch.tensor([1., 1.])

    # Known logits (output of linear layer)
    logits = torch.tensor([-0.5, 1.0]) # S0 logit, S1 logit

    # --- Expected BCEWithLogitsLoss Calculation ---
    # L = -[y * log(sigmoid(x)) + (1 - y) * log(1 - sigmoid(x))]
    # S0 (y=0): -[0*log(sig(-0.5)) + 1*log(1-sig(-0.5))] = -log(1-0.3775) = -log(0.6225) = 0.4740
    # S1 (y=1): -[1*log(sig(1.0)) + 0*log(1-sig(1.0))] = -log(0.7311) = 0.3133
    # Average Loss = (0.4740 + 0.3133) / 2 = 0.7873 / 2 = 0.39365
    expected_loss = 0.39365

    # Mock the forward method
    # The head internally squeezes the output for binary classification if num_classes=2
    # So the mock should return the raw output of the linear layer.
    # Check the actual output dim of the final layer
    final_layer_output_dim = head.prediction_network[-1].out_features
    mock_return_value = logits.unsqueeze(-1) if final_layer_output_dim == 1 else logits

    with patch.object(head.prediction_network, 'forward', return_value=mock_return_value) as mock_forward:
        # Compute loss
        outputs = head(x, targets, mask) # Pass targets directly, head handles shape
        computed_loss = outputs['loss'].item()
        mock_forward.assert_called_once_with(x)

    assert pytest.approx(computed_loss, abs=1e-4) == expected_loss


def test_count_poisson_nll_loss():
    """Verify Poisson NLL loss calculation for CountDataHead."""
    config = {'name': 'count_poi', 'input_dim': 2, 'distribution': 'poisson'}
    head = CountDataHead(**config)

    x = torch.randn(2, config['input_dim'])
    targets = torch.tensor([2., 5.]) # Target counts
    mask = torch.tensor([1., 1.])
    # Known rate (lambda) predictions
    rates = torch.tensor([3.0, 4.0])
    # The network outputs raw values before softplus
    mock_output = torch.log(torch.expm1(rates)).unsqueeze(-1)

    # --- Expected Poisson NLL Calculation ---
    # NLL = -(y * log(lambda) - lambda) = lambda - y * log(lambda)
    # S0 (y=2, lambda=3): 3.0 - 2 * log(3.0) = 3.0 - 2 * 1.0986 = 0.8028
    # S1 (y=5, lambda=4): 4.0 - 5 * log(4.0) = 4.0 - 5 * 1.3863 = -2.9315
    # Average NLL = (0.8028 + (-2.9315)) / 2 = -2.1287 / 2 = -1.06435
    expected_loss = -1.06435 # Final correction for assertion

    # Mock the forward method
    with patch.object(head.prediction_network, 'forward', return_value=mock_output) as mock_forward:
        outputs = head(x, targets, mask)
        computed_loss = outputs['loss'].item()
        mock_forward.assert_called_once_with(x)

    assert pytest.approx(computed_loss, abs=1e-4) == expected_loss


def test_count_nb_nll_loss():
    """Verify Negative Binomial NLL loss calculation for CountDataHead."""
    config = {'name': 'count_nb', 'input_dim': 2, 'distribution': 'negative_binomial'}
    head = CountDataHead(**config)

    x = torch.randn(2, config['input_dim'])
    targets = torch.tensor([2., 5.]) # Target counts
    mask = torch.tensor([1., 1.])
    # Known rate (mu) and dispersion (alpha) predictions
    rates = torch.tensor([3.0, 4.0])
    dispersions = torch.tensor([0.5, 1.0]) # alpha

    # Need shape [batch, 2] for network output
    mock_output = torch.stack([torch.log(torch.expm1(rates)), torch.log(torch.expm1(dispersions))], dim=1)

    # --- Expected NB NLL Calculation ---
    # NLL = - [lgamma(y+r) - lgamma(y+1) - lgamma(r) + r*log(r) + y*log(mu) - (y+r)*log(r+mu)]
    # where r = 1/alpha
    # S0 (y=2, mu=3, alpha=0.5 -> r=2):
    #   NLL0 = - [lgamma(4) - lgamma(3) - lgamma(2) + 2*log(2) + 2*log(3) - 4*log(5)]
    #        = - [log(6) - log(2) - log(1) + 2*0.6931 + 2*1.0986 - 4*1.6094]
    #        = - [1.7918 - 0.6931 - 0 + 1.3862 + 2.1972 - 6.4376] = - [-1.7555] = 1.7555
    # S1 (y=5, mu=4, alpha=1.0 -> r=1):
    #   NLL1 = - [lgamma(6) - lgamma(6) - lgamma(1) + 1*log(1) + 5*log(4) - 6*log(5)]
    #        = - [0 - 0 - 0 + 0 + 5*1.3863 - 6*1.6094]
    #        = - [6.9315 - 9.6564] = - [-2.7249] = 2.7249
    # Average NLL = (1.7555 + 2.7249) / 2 = 4.4804 / 2 = 2.2402
    expected_loss = 2.2402

    # Mock the forward method
    with patch.object(head.prediction_network, 'forward', return_value=mock_output) as mock_forward:
        outputs = head(x, targets, mask)
        computed_loss = outputs['loss'].item()
        mock_forward.assert_called_once_with(x)

    # Loosen tolerance slightly due to lgamma approximations
    assert pytest.approx(computed_loss, abs=1e-3) == expected_loss


def test_count_zip_nll_loss():
    """Verify Zero-Inflated Poisson NLL loss calculation."""
    config = {'name': 'count_zip', 'input_dim': 2, 'distribution': 'poisson', 'zero_inflated': True}
    head = CountDataHead(**config)

    x = torch.randn(3, config['input_dim'])
    targets = torch.tensor([0., 0., 3.]) # Target counts (two zeros)
    mask = torch.tensor([1., 1., 1.])
    # Known rate (lambda) and zero_prob (pi) predictions
    rates = torch.tensor([0.5, 2.0, 4.0])
    zero_probs = torch.tensor([0.1, 0.6, 0.1]) # pi

    # Need shape [batch, 2] for network output
    zero_logits = torch.log(zero_probs / (1 - zero_probs)) # Inverse sigmoid
    mock_output = torch.stack([torch.log(torch.expm1(rates)), zero_logits], dim=1)

    # --- Expected ZIP NLL Calculation ---
    # NLL = - log(P(Y=y | lambda, pi))
    # If y=0: NLL = - log(pi + (1-pi)*exp(-lambda))
    # If y>0: NLL = - [log(1-pi) + y*log(lambda) - lambda] (omitting log(y!))
    # S0 (y=0, lambda=0.5, pi=0.1): NLL0 = - log(0.1 + 0.9*exp(-0.5)) = - log(0.64585) = 0.4371
    # S1 (y=0, lambda=2.0, pi=0.6): NLL1 = - log(0.6 + 0.4*exp(-2.0)) = - log(0.65412) = 0.4245
    # S2 (y=3, lambda=4.0, pi=0.1): NLL2 = - [log(0.9) + 3*log(4.0) - 4.0] = - [-0.1054 + 4.1589 - 4.0] = -0.0535
    # Average NLL = (0.4371 + 0.4245 - 0.0535) / 3 = 0.8081 / 3 = 0.26937
    expected_loss = 0.26937 # Final correction for assertion

    # Mock the forward method
    with patch.object(head.prediction_network, 'forward', return_value=mock_output) as mock_forward:
        outputs = head(x, targets, mask)
        computed_loss = outputs['loss'].item()
        mock_forward.assert_called_once_with(x)

    assert pytest.approx(computed_loss, abs=1e-4) == expected_loss


def test_count_zinb_nll_loss():
    """Verify Zero-Inflated Negative Binomial NLL loss calculation."""
    config = {'name': 'count_zinb', 'input_dim': 2, 'distribution': 'negative_binomial', 'zero_inflated': True}
    head = CountDataHead(**config)

    x = torch.randn(3, config['input_dim'])
    targets = torch.tensor([0., 0., 3.]) # Target counts
    mask = torch.tensor([1., 1., 1.])
    # Known rate (mu), dispersion (alpha), zero_prob (pi)
    rates = torch.tensor([0.5, 2.0, 4.0]) # mu
    dispersions = torch.tensor([1.0, 0.5, 2.0]) # alpha
    zero_probs = torch.tensor([0.1, 0.6, 0.1]) # pi

    # Need shape [batch, 3] for network output
    zero_logits = torch.log(zero_probs / (1 - zero_probs)) # Inverse sigmoid
    mock_output = torch.stack([
        torch.log(torch.expm1(rates)),
        torch.log(torch.expm1(dispersions)),
        zero_logits
    ], dim=1)

    # --- Expected ZINB NLL Calculation ---
    # NLL = - log(P(Y=y | mu, alpha, pi))
    # If y=0: NLL = - log(pi + (1-pi)*NB(0 | mu, alpha))
    # If y>0: NLL = - [log(1-pi) + log(NB(y | mu, alpha))]
    # NB(0 | mu, alpha) = (r/(r+mu))^r where r = 1/alpha
    # log(NB(y | mu, r)) = lgamma(y+r) - lgamma(y+1) - lgamma(r) + r*log(r) + y*log(mu) - (y+r)*log(r+mu)
    # S0 (y=0, mu=0.5, alpha=1.0->r=1, pi=0.1): NLL0 = - log(0.1 + 0.9*(1/1.5)^1) = -log(0.7) = 0.3567
    # S1 (y=0, mu=2.0, alpha=0.5->r=2, pi=0.6): NLL1 = - log(0.6 + 0.4*(2/4)^2) = -log(0.7) = 0.3567
    # S2 (y=3, mu=4.0, alpha=2.0->r=0.5, pi=0.1):
    #   logNB = lgamma(3.5)-lgamma(4)-lgamma(0.5) + 0.5*log(0.5) + 3*log(4) - 3.5*log(4.5) = -2.6154
    #   NLL2 = - [log(0.9) + logNB] = - [-0.1054 - 2.6154] = 2.7208
    # Average NLL = (0.3567 + 0.3567 + 2.7208) / 3 = 3.4342 / 3 = 1.1447
    expected_loss = 1.1447

    # Mock the forward method
    with patch.object(head.prediction_network, 'forward', return_value=mock_output) as mock_forward:
        outputs = head(x, targets, mask)
        computed_loss = outputs['loss'].item()
        mock_forward.assert_called_once_with(x)

    # Loosen tolerance slightly due to lgamma approximations
    assert pytest.approx(computed_loss, abs=1e-3) == expected_loss
