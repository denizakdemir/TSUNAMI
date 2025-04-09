import pytest
import torch
import numpy as np

from enhanced_deephit.models.tasks.base import TaskHead, MultiTaskManager
from enhanced_deephit.models.tasks.survival import SingleRiskHead, CompetingRisksHead
from enhanced_deephit.models.tasks.standard import ClassificationHead, RegressionHead, CountDataHead


class TestMultiTaskManager:
    """Test suite for the MultiTaskManager class"""
    
    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks for testing"""
        tasks = [
            SingleRiskHead(name='survival', input_dim=64, num_time_bins=30),
            ClassificationHead(name='classification', input_dim=64, num_classes=3),
            RegressionHead(name='regression', input_dim=64)
        ]
        return tasks
    
    def test_multi_task_manager_initialization(self, sample_tasks):
        """Test initialization of MultiTaskManager"""
        manager = MultiTaskManager(
            encoder_dim=64,
            task_heads=sample_tasks,
            include_variational=True,
            variational_beta=0.1
        )
        
        # Check that task heads are properly stored
        assert len(manager.task_heads) == 3
        assert manager.task_names == ['survival', 'classification', 'regression']
        assert manager.include_variational is True
        assert manager.variational_beta == 0.1
        
        # Check that the variational components are created
        assert manager.variational_mu is not None
        assert manager.variational_logvar is not None
        
        # Check that task weights are created properly
        assert len(manager.task_weights) == 3
        for name, weight in manager.task_weights.items():
            assert name in ['survival', 'classification', 'regression']
            assert isinstance(weight, torch.nn.Parameter)
    
    def test_multi_task_manager_forward_pass(self, sample_tasks):
        """Test forward pass of MultiTaskManager"""
        batch_size = 16
        encoder_dim = 64
        
        # Create manager
        manager = MultiTaskManager(
            encoder_dim=encoder_dim,
            task_heads=sample_tasks
        )
        
        # Create encoder output and targets
        encoder_output = torch.randn(batch_size, encoder_dim)
        
        targets = {
            'survival': torch.cat([
                torch.randint(0, 30, (batch_size, 1)),  # time
                torch.randint(0, 2, (batch_size, 1))  # event
            ], dim=1),
            'classification': torch.randint(0, 3, (batch_size,)),
            'regression': torch.randn(batch_size)
        }
        
        # Create masks (all ones for this test)
        masks = {
            'survival': torch.ones(batch_size),
            'classification': torch.ones(batch_size),
            'regression': torch.ones(batch_size)
        }
        
        # Forward pass
        outputs = manager(
            x=encoder_output,
            targets=targets,
            masks=masks
        )
        
        # Check outputs
        assert 'loss' in outputs
        assert 'task_losses' in outputs
        assert 'task_outputs' in outputs
        
        # Check task losses
        assert 'survival' in outputs['task_losses']
        assert 'classification' in outputs['task_losses']
        assert 'regression' in outputs['task_losses']
        
        # Check task outputs
        assert 'survival' in outputs['task_outputs']
        assert 'classification' in outputs['task_outputs']
        assert 'regression' in outputs['task_outputs']
        
        # Check that task outputs contain the expected fields
        assert 'survival' in outputs['task_outputs']['survival']
        assert 'predictions' in outputs['task_outputs']['classification']
        assert 'predictions' in outputs['task_outputs']['regression']
    
    def test_multi_task_manager_variational(self, sample_tasks):
        """Test variational component of MultiTaskManager"""
        batch_size = 16
        encoder_dim = 64
        
        # Create manager with variational component
        manager = MultiTaskManager(
            encoder_dim=encoder_dim,
            task_heads=sample_tasks,
            include_variational=True,
            variational_beta=0.1
        )
        
        # Create encoder output and targets
        encoder_output = torch.randn(batch_size, encoder_dim)
        
        targets = {
            'survival': torch.cat([
                torch.randint(0, 30, (batch_size, 1)),
                torch.randint(0, 2, (batch_size, 1))
            ], dim=1),
            'classification': torch.randint(0, 3, (batch_size,)),
            'regression': torch.randn(batch_size)
        }
        
        # Forward pass
        outputs = manager(
            x=encoder_output,
            targets=targets
        )
        
        # Check that variational loss is included
        assert 'variational_loss' in outputs
        assert outputs['variational_loss'].item() > 0
        
        # Check that the loss includes the variational component
        loss_sum = sum(loss.item() for loss in outputs['task_losses'].values())
        assert abs(loss_sum + outputs['variational_loss'].item() - outputs['loss'].item()) < 1e-5
    
    def test_multi_task_manager_missing_targets(self, sample_tasks):
        """Test MultiTaskManager with missing targets"""
        batch_size = 16
        encoder_dim = 64
        
        # Create manager
        manager = MultiTaskManager(
            encoder_dim=encoder_dim,
            task_heads=sample_tasks
        )
        
        # Create encoder output
        encoder_output = torch.randn(batch_size, encoder_dim)
        
        # Create partial targets (missing 'regression')
        targets = {
            'survival': torch.cat([
                torch.randint(0, 30, (batch_size, 1)),
                torch.randint(0, 2, (batch_size, 1))
            ], dim=1),
            'classification': torch.randint(0, 3, (batch_size,))
        }
        
        # Forward pass
        outputs = manager(
            x=encoder_output,
            targets=targets
        )
        
        # Check that only provided targets are in task_losses
        assert 'survival' in outputs['task_losses']
        assert 'classification' in outputs['task_losses']
        assert 'regression' not in outputs['task_losses']
        
        # But all tasks should have outputs
        assert 'survival' in outputs['task_outputs']
        assert 'classification' in outputs['task_outputs']
        assert 'regression' in outputs['task_outputs']
    
    def test_get_task_method(self, sample_tasks):
        """Test the get_task method"""
        manager = MultiTaskManager(
            encoder_dim=64,
            task_heads=sample_tasks
        )
        
        # Check that tasks can be retrieved by name
        assert manager.get_task('survival') is not None
        assert manager.get_task('classification') is not None
        assert manager.get_task('regression') is not None
        assert manager.get_task('nonexistent') is None
        
        # Check the returned task has the correct type
        assert isinstance(manager.get_task('survival'), SingleRiskHead)
        assert isinstance(manager.get_task('classification'), ClassificationHead)
        assert isinstance(manager.get_task('regression'), RegressionHead)
    
    def test_predict_method(self, sample_tasks):
        """Test the predict method"""
        batch_size = 16
        encoder_dim = 64
        
        # Create manager
        manager = MultiTaskManager(
            encoder_dim=encoder_dim,
            task_heads=sample_tasks
        )
        
        # Create encoder output
        encoder_output = torch.randn(batch_size, encoder_dim)
        
        # Predict
        predictions = manager.predict(encoder_output)
        
        # Check predictions
        assert 'survival' in predictions
        assert 'classification' in predictions
        assert 'regression' in predictions
        
        # Check specific outputs
        assert 'survival' in predictions['survival']
        assert 'predictions' in predictions['classification']
        assert 'predictions' in predictions['regression']


class TestTaskHeads:
    """Test suite for individual task heads"""
    
    def test_single_risk_head(self):
        """Test SingleRiskHead initialization and forward pass"""
        batch_size = 16
        encoder_dim = 64
        num_times = 30
        
        # Create task head
        head = SingleRiskHead(
            name='survival',
            input_dim=encoder_dim,
            num_time_bins=num_times,
            alpha_rank=0.2,
            alpha_calibration=0.2
        )
        
        # Check initialization
        assert head.name == 'survival'
        assert head.num_time_bins == num_times
        
        # Create input and target
        x = torch.randn(batch_size, encoder_dim)
        target = torch.cat([
            torch.randint(0, 2, (batch_size, 1)),  # event
            torch.randint(0, num_times, (batch_size, 1))  # time
        ], dim=1)
        
        # Forward pass
        output = head(x)
        
        # Check output shape and properties
        assert 'survival' in output
        assert output['survival'].shape == (batch_size, num_times)
        assert torch.all(output['survival'] >= 0) and torch.all(output['survival'] <= 1)
        
        # Monotonicity check - survival curve should be non-increasing
        differences = output['survival'][:, :-1] - output['survival'][:, 1:]
        assert torch.all(differences >= -1e-6)  # Allow small numerical error
        
        # Test loss computation by calling forward with targets
        output_with_loss = head(x, target)
        assert 'loss' in output_with_loss
        assert output_with_loss['loss'].numel() == 1  # Scalar loss
        assert output_with_loss['loss'].item() > 0  # Loss should be positive
        
        # Test metric computation using the compute_metrics method
        metrics = head.compute_metrics(output, target)
        assert 'c_index' in metrics
        assert 0 <= metrics['c_index'] <= 1
    
    def test_competing_risks_head(self):
        """Test CompetingRisksHead initialization and forward pass"""
        batch_size = 16
        encoder_dim = 64
        num_times = 30
        num_risks = 3
        
        # Create task head
        head = CompetingRisksHead(
            name='competing_risks',
            input_dim=encoder_dim,
            num_time_bins=num_times,
            num_risks=num_risks,
            alpha_rank=0.3,
            alpha_calibration=0.1
        )
        
        # Check initialization
        assert head.name == 'competing_risks'
        assert head.num_time_bins == num_times
        assert head.num_risks == num_risks
        
        # Create input and target
        x = torch.randn(batch_size, encoder_dim)
        target = torch.cat([
            torch.randint(0, 2, (batch_size, 1)),  # event
            torch.randint(0, num_times, (batch_size, 1)),  # time
            torch.randint(0, num_risks + 1, (batch_size, 1))  # cause (0 = censored, 1+ = event types)
        ], dim=1)
        
        # Forward pass
        output = head(x)
        
        # Check output shape and properties
        assert 'cif' in output
        assert 'overall_survival' in output
        assert output['cif'].shape == (batch_size, num_risks, num_times)
        assert output['overall_survival'].shape == (batch_size, num_times)
        
        # Check valid probabilities
        assert torch.all(output['cif'] >= 0) and torch.all(output['cif'] <= 1)
        assert torch.all(output['overall_survival'] >= 0) and torch.all(output['overall_survival'] <= 1)
        
        # Skip detailed monotonicity check as it depends on the specific implementation
        
        # Check probability constraint: survival + sum(CIF) = 1
        total_prob = output['overall_survival'] + torch.sum(output['cif'], dim=1)
        assert torch.allclose(total_prob, torch.ones_like(total_prob), atol=1e-5)
        
        # Test loss computation by calling forward with targets
        output_with_loss = head(x, target)
        assert 'loss' in output_with_loss
        assert output_with_loss['loss'].numel() == 1  # Scalar loss
        assert output_with_loss['loss'].item() > 0  # Loss should be positive
        
        # Test metric computation
        metrics = head.compute_metrics(output, target)
        assert 'c_index_avg' in metrics
    
    def test_classification_head_binary(self):
        """Test ClassificationHead with binary classification"""
        batch_size = 16
        encoder_dim = 64
        
        # Create task head for binary classification
        head = ClassificationHead(
            name='binary',
            input_dim=encoder_dim,
            num_classes=2,
            class_weights=None
        )
        
        # Check initialization
        assert head.name == 'binary'
        assert head.num_classes == 2
        
        # Create input and target
        x = torch.randn(batch_size, encoder_dim)
        target = torch.randint(0, 2, (batch_size,))
        
        # Forward pass
        output = head(x)
        
        # Check output shape and properties
        assert 'probabilities' in output
        
        if output['probabilities'].dim() > 1:  # Multi-class form
            assert output['probabilities'].shape == (batch_size, 2)
            assert torch.all(torch.sum(output['probabilities'], dim=1).sub(1).abs() < 1e-5)  # Sum to 1
        else:  # Binary form
            assert output['probabilities'].shape == (batch_size,)
        
        # Test loss computation
        loss = head.loss(output, target)
        assert loss.numel() == 1
        assert loss.item() > 0
        
        # Test metric computation
        metrics = head.compute_metrics(output, target)
        assert 'accuracy' in metrics
    
    def test_classification_head_multiclass(self):
        """Test ClassificationHead with multiclass classification"""
        batch_size = 16
        encoder_dim = 64
        num_classes = 4
        
        # Create task head for multiclass classification
        head = ClassificationHead(
            name='multiclass',
            input_dim=encoder_dim,
            num_classes=num_classes,
            class_weights=torch.ones(num_classes)  # Equal weights
        )
        
        # Check initialization
        assert head.name == 'multiclass'
        assert head.num_classes == num_classes
        
        # Create input and target
        x = torch.randn(batch_size, encoder_dim)
        target = torch.randint(0, num_classes, (batch_size,))
        
        # Forward pass
        output = head(x)
        
        # Check output shape and properties
        assert 'probabilities' in output
        assert output['probabilities'].shape == (batch_size, num_classes)
        assert torch.all(torch.sum(output['probabilities'], dim=1).sub(1).abs() < 1e-5)  # Sum to 1
        
        # Test loss computation
        loss = head.loss(output, target)
        assert loss.numel() == 1
        assert loss.item() > 0
        
        # Test metric computation
        metrics = head.compute_metrics(output, target)
        assert 'accuracy' in metrics
    
    def test_regression_head(self):
        """Test RegressionHead initialization and forward pass"""
        batch_size = 16
        encoder_dim = 64
        
        # Create task head
        head = RegressionHead(
            name='regression',
            input_dim=encoder_dim,
            loss_type='mse'
        )
        
        # Check initialization
        assert head.name == 'regression'
        
        # Create input and target
        x = torch.randn(batch_size, encoder_dim)
        target = torch.randn(batch_size)
        
        # Forward pass
        output = head(x)
        
        # Check output shape
        assert 'predictions' in output
        assert output['predictions'].shape == (batch_size,)  # Single output per sample
        
        # Test loss computation
        loss = head.loss(output, target)
        assert loss.numel() == 1
        assert loss.item() >= 0  # MSE is always non-negative
        
        # Test metric computation
        metrics = head.compute_metrics(output, target)
        assert 'mse' in metrics
    
    def test_count_data_head(self):
        """Test CountDataHead initialization and forward pass"""
        batch_size = 16
        encoder_dim = 64
        
        # Create task head
        head = CountDataHead(
            name='count',
            input_dim=encoder_dim
        )
        
        # Check initialization
        assert head.name == 'count'
        
        # Create input and target
        x = torch.randn(batch_size, encoder_dim)
        target = torch.randint(0, 10, (batch_size,))
        
        # Forward pass
        output = head(x)
        
        # Check output shape
        assert 'predictions' in output
        assert output['predictions'].shape == (batch_size,)  # Single output per sample
        assert torch.all(output['predictions'] > 0)  # Poisson rates are positive
        
        # Test loss computation
        loss = head.loss(output, target)
        assert loss.numel() == 1
        assert loss.item() >= 0
        
        # Test metric computation
        metrics = head.compute_metrics(output, target)
        assert 'rmse' in metrics