import pytest
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from tsunami.models.model import EnhancedDeepHit
from tsunami.models.tasks.survival import SingleRiskHead, CompetingRisksHead
from tsunami.models.tasks.standard import ClassificationHead, RegressionHead
from tsunami.data.processing import DataProcessor


class TestModelIntegration:
    """Integration tests for the TSUNAMI library components"""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a small pandas dataset for integration testing"""
        np.random.seed(42)
        n_samples = 100
        
        # Create continuous features
        continuous_data = {
            f'cont_{i}': np.random.randn(n_samples) for i in range(5)
        }
        
        # Create categorical features
        categorical_data = {
            'cat_1': np.random.choice(['A', 'B', 'C'], n_samples),
            'cat_2': np.random.choice(['X', 'Y', 'Z', 'W'], n_samples)
        }
        
        # Create target variables
        survival_time = np.random.randint(1, 30, n_samples)
        survival_event = np.random.randint(0, 2, n_samples)
        
        binary_target = np.random.randint(0, 2, n_samples)
        regression_target = np.random.randn(n_samples)
        
        # Combine into a dataframe
        df = pd.DataFrame({
            **continuous_data,
            **categorical_data,
            'time': survival_time,
            'event': survival_event,
            'binary_target': binary_target,
            'regression_target': regression_target
        })
        
        # Add some missing values
        for col in ['cont_0', 'cont_1', 'cat_1']:
            mask = np.random.rand(n_samples) < 0.1
            df.loc[mask, col] = np.nan
            
        return df
    
    def test_data_processing_to_model_pipeline(self, sample_dataset):
        """Test the full pipeline from data processing to model prediction"""
        df = sample_dataset
        
        # Create a data processor with correct parameter names
        processor = DataProcessor(
            num_impute_strategy='mean',
            cat_impute_strategy='most_frequent',
            normalize='robust',
            create_missing_indicators=True
        )
        
        # Fit the processor and transform the data
        processed_data = processor.fit_transform(df)
        
        # Extract the processed tensors
        continuous = torch.tensor(processed_data.values, dtype=torch.float32)
        
        # Create targets
        survival_target = torch.tensor(np.column_stack([
            df['time'].values, df['event'].values
        ]), dtype=torch.float32)
        
        binary_target = torch.tensor(df['binary_target'].values, dtype=torch.long)
        regression_target = torch.tensor(df['regression_target'].values, dtype=torch.float32)
        
        # Create a simple dataloader
        dataset = TensorDataset(
            continuous, 
            survival_target, binary_target, regression_target
        )
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Set encoder dimension for all components
        encoder_dim = 32
        
        # Create tasks for the model
        tasks = [
            SingleRiskHead(
                name='survival',
                input_dim=encoder_dim,  # Match the encoder dimension
                num_time_bins=30,
                alpha_rank=0.2,
                alpha_calibration=0.2
            ),
            ClassificationHead(
                name='binary',
                input_dim=encoder_dim,  # Match the encoder dimension
                num_classes=2
            ),
            RegressionHead(
                name='regression',
                input_dim=encoder_dim  # Match the encoder dimension
            )
        ]
        
        # Create the model
        model = EnhancedDeepHit(
            num_continuous=continuous.shape[1],
            targets=tasks,
            encoder_dim=encoder_dim,  # Use the same dimension
            encoder_depth=2,
            encoder_heads=2,
            include_variational=True
        )
        
        # Do a single training step to test the pipeline
        for batch in dataloader:
            continuous_batch, surv_target, bin_target, reg_target = batch
            
            # Create the targets dictionary
            targets = {
                'survival': surv_target,
                'binary': bin_target,
                'regression': reg_target
            }
            
            # Forward pass with targets
            outputs = model(
                continuous=continuous_batch,
                targets=targets
            )
            
            # Check that the model produces the expected outputs
            assert 'loss' in outputs
            assert 'task_outputs' in outputs
            assert 'survival' in outputs['task_outputs']
            assert 'binary' in outputs['task_outputs']
            assert 'regression' in outputs['task_outputs']
            
            # Check survival task outputs
            surv_output = outputs['task_outputs']['survival']
            assert 'survival' in surv_output
            assert surv_output['survival'].shape == (continuous_batch.shape[0], 30)
            
            # Break after one batch to keep the test short
            break
    
    def test_end_to_end_multi_task_workflow(self, sample_dataset):
        """Test an end-to-end workflow for multi-task learning with the model"""
        df = sample_dataset
        
        # Create a data processor with correct parameter names
        processor = DataProcessor(
            num_impute_strategy='mean',
            cat_impute_strategy='most_frequent',
            normalize='robust',
            create_missing_indicators=True
        )
        
        # Fit the processor and transform the data
        processed_data = processor.fit_transform(df)
        
        # Create tensors for features and targets
        continuous = torch.tensor(processed_data.values, dtype=torch.float32)
        
        # Extract targets
        surv_target = torch.tensor(np.column_stack([
            df['time'].values, df['event'].values
        ]), dtype=torch.float32)
        
        bin_target = torch.tensor(df['binary_target'].values, dtype=torch.long)
        reg_target = torch.tensor(df['regression_target'].values, dtype=torch.float32)
        
        # Split data into train/test
        n_samples = len(df)
        train_idx = np.random.choice(n_samples, int(0.8 * n_samples), replace=False)
        test_idx = np.setdiff1d(np.arange(n_samples), train_idx)
        
        # Create train/test datasets
        train_continuous = continuous[train_idx]
        train_surv = surv_target[train_idx]
        train_bin = bin_target[train_idx]
        train_reg = reg_target[train_idx]
        
        test_continuous = continuous[test_idx]
        
        # Create dataloaders
        train_dataset = TensorDataset(
            train_continuous, train_surv, train_bin, train_reg
        )
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        # Set encoder dimension for all components
        encoder_dim = 32
        
        # Create tasks with correct parameter names
        tasks = [
            SingleRiskHead(
                name='survival',
                input_dim=encoder_dim,  # Match the encoder dimension
                num_time_bins=30
            ),
            ClassificationHead(
                name='binary',
                input_dim=encoder_dim,  # Match the encoder dimension
                num_classes=2
            ),
            RegressionHead(
                name='regression',
                input_dim=encoder_dim  # Match the encoder dimension
            )
        ]
        
        # Create model
        model = EnhancedDeepHit(
            num_continuous=continuous.shape[1],
            targets=tasks,
            encoder_dim=encoder_dim,  # Use the same dimension
            encoder_depth=2,
            encoder_heads=2
        )
        
        # Train for just a few epochs for testing
        for epoch in range(2):
            for batch in train_loader:
                continuous_batch, surv_target, bin_target, reg_target = batch
                
                # Process batch and get loss
                outputs = model(
                    continuous=continuous_batch,
                    targets={
                        'survival': surv_target,
                        'binary': bin_target,
                        'regression': reg_target
                    }
                )
                
                # Extract and check loss
                loss = outputs['loss']
                assert not torch.isnan(loss)
                assert loss.item() > 0
                
            # After the second epoch, break
            if epoch == 1:
                break
        
        # Test prediction on test data
        model.eval()
        with torch.no_grad():
            predictions = model.predict(
                continuous=test_continuous
            )
        
        # Check predictions
        assert 'task_outputs' in predictions
        assert 'survival' in predictions['task_outputs']
        assert 'binary' in predictions['task_outputs']
        assert 'regression' in predictions['task_outputs']
        
        # Check survival predictions shape
        surv_preds = predictions['task_outputs']['survival']['survival']
        assert surv_preds.shape == (len(test_idx), 30)
        
        # Check binary predictions
        bin_preds = predictions['task_outputs']['binary']['predictions']
        assert bin_preds.shape[0] == len(test_idx)
        
        # Check regression predictions
        reg_preds = predictions['task_outputs']['regression']['predictions']
        assert reg_preds.shape[0] == len(test_idx)
        
        # Test uncertainty quantification
        uncertainty = model.compute_uncertainty(
            continuous=test_continuous[:5],  # Use a small subset for speed
            num_samples=3
        )
        
        # Check uncertainty outputs
        assert 'survival' in uncertainty
        assert 'mean' in uncertainty['survival']
        assert 'std' in uncertainty['survival']
        assert 'samples' in uncertainty['survival']
        
        # Test applying a threshold to binary predictions
        binary_decisions = (bin_preds > 0.5).float()
        assert torch.all((binary_decisions == 0) | (binary_decisions == 1))


class TestCompetingRisksIntegration:
    """Integration tests focusing on competing risks functionality"""
    
    @pytest.fixture
    def competing_risks_data(self):
        """Create a dataset with competing risks outcomes"""
        np.random.seed(42)
        n_samples = 80
        
        # Features
        continuous_data = np.random.randn(n_samples, 4)
        
        # Time to event
        times = np.random.randint(1, 20, n_samples)
        
        # Event types: 0 = censored, 1 = event type 1, 2 = event type 2
        events = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3])
        
        return {
            'continuous': continuous_data,
            'times': times,
            'events': events,
            'n_samples': n_samples
        }
    
    def test_competing_risks_model_workflow(self, competing_risks_data):
        """Test the competing risks model workflow"""
        # Convert data to tensors
        continuous = torch.tensor(competing_risks_data['continuous'], dtype=torch.float32)
        
        # Adjust event causes to be 0-based for indexing (0 = censored, 1 = cause 1, 2 = cause 2)
        # We need to adjust because CompetingRisksHead expects cause indices to be 0-based
        # where 0 is the first cause, 1 is the second cause, etc.
        events = competing_risks_data['events'].copy()
        # Convert events: 0->censored (no change), 1->cause 0, 2->cause 1
        events_adjusted = np.zeros_like(events)
        events_adjusted[events > 0] = events[events > 0] - 1
        
        # Create competing risks target with three columns: time, event, cause
        cr_target = torch.tensor(np.column_stack([
            competing_risks_data['times'],
            events > 0,  # Convert to binary event indicator (1 = any event occurred)
            events_adjusted  # Event type/cause (0-based: -1 = censored, 0 = first cause, 1 = second cause)
        ]), dtype=torch.float32)
        
        # Create dataloader
        dataset = TensorDataset(continuous, cr_target)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Set encoder dimension for all components
        encoder_dim = 32
        
        # Create competing risks task
        cr_task = CompetingRisksHead(
            name='competing_risks',
            input_dim=encoder_dim,  # Match the encoder dimension
            num_time_bins=20,
            num_risks=2,  # This means we have 2 competing events/causes
            alpha_rank=0.2,
            alpha_calibration=0.2
        )
        
        # Create model
        model = EnhancedDeepHit(
            num_continuous=4,
            targets=[cr_task],
            encoder_dim=encoder_dim,  # Use the same dimension
            encoder_depth=2
        )
        
        # Test a forward pass
        for batch in dataloader:
            cont_batch, target_batch = batch
            
            # Forward pass
            outputs = model(
                continuous=cont_batch,
                targets={'competing_risks': target_batch}
            )
            
            # Check outputs
            assert 'task_outputs' in outputs
            assert 'competing_risks' in outputs['task_outputs']
            
            cr_output = outputs['task_outputs']['competing_risks']
            assert 'cif' in cr_output
            assert 'overall_survival' in cr_output
            
            # Check shapes
            batch_size = cont_batch.shape[0]
            assert cr_output['cif'].shape == (batch_size, 2, 20)  # [batch_size, num_risks, num_times]
            assert cr_output['overall_survival'].shape == (batch_size, 20)  # [batch_size, num_times]
            
            # Check valid probabilities
            assert torch.all(cr_output['cif'] >= 0) and torch.all(cr_output['cif'] <= 1)
            assert torch.all(cr_output['overall_survival'] >= 0) and torch.all(cr_output['overall_survival'] <= 1)
            
            # Check that survival + sum(CIF) is approximately 1
            total_prob = cr_output['overall_survival'] + torch.sum(cr_output['cif'], dim=1)
            assert torch.allclose(total_prob, torch.ones_like(total_prob), atol=1e-5)
            
            # Only test one batch
            break
            
        # Test prediction mode
        model.eval()
        with torch.no_grad():
            predictions = model.predict(continuous=continuous[:10])
            
            # Check prediction outputs
            assert 'task_outputs' in predictions
            assert 'competing_risks' in predictions['task_outputs']
            
            cr_preds = predictions['task_outputs']['competing_risks']
            assert 'cif' in cr_preds
            assert 'overall_survival' in cr_preds
            
            # Check shape and validity
            assert cr_preds['cif'].shape == (10, 2, 20)
            assert cr_preds['overall_survival'].shape == (10, 20)
            
            # Test uncertainty quantification
            uncertainty = model.compute_uncertainty(
                continuous=continuous[:5],
                num_samples=3
            )
            
            assert 'competing_risks' in uncertainty
            assert 'mean' in uncertainty['competing_risks']
            assert 'std' in uncertainty['competing_risks']
            assert 'samples' in uncertainty['competing_risks']