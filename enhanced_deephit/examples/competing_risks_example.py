import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from data.processing import DataProcessor
from models import EnhancedDeepHit
from models.tasks.survival import CompetingRisksHead
from simulation.data_generation import generate_competing_risks_data


class CompetingRisksDataset(Dataset):
    """Dataset for competing risks analysis"""
    
    def __init__(self, df, target, processor=None):
        """
        Initialize dataset
        
        Parameters
        ----------
        df : pd.DataFrame
            Input features
            
        target : np.ndarray
            Target values [n_samples, 3 + n_risks * num_bins]
            
        processor : DataProcessor, optional
            Data processor for transforming features
        """
        self.target = torch.tensor(target, dtype=torch.float32)
        
        if processor is not None:
            # Transform features
            self.df_processed = processor.transform(df)
            
            # Get categorical feature info for the encoder
            self.cat_feat_info = []
            for col, info in processor.cat_embed_info.items():
                self.cat_feat_info.append({
                    'name': col,
                    'cardinality': info['cardinality'],
                    'embed_dim': info['embed_dim']
                })
        else:
            # Use raw features
            self.df_processed = df
            self.cat_feat_info = []
        
        # Extract continuous and categorical features
        self.continuous_cols = [col for col in self.df_processed.columns 
                                if not col.startswith('cat_') and 
                                not col.endswith('_missing')]
        
        self.missing_indicator_cols = [col for col in self.df_processed.columns 
                                       if col.endswith('_missing')]
        
        # Convert to tensors
        self.continuous = torch.tensor(
            self.df_processed[self.continuous_cols].values, 
            dtype=torch.float32
        )
        
        if self.missing_indicator_cols:
            self.missing_mask = torch.tensor(
                self.df_processed[self.missing_indicator_cols].values,
                dtype=torch.float32
            )
        else:
            self.missing_mask = None
    
    def __len__(self):
        return len(self.continuous)
    
    def __getitem__(self, idx):
        item = {
            'continuous': self.continuous[idx],
            'targets': {
                'competing_risks': self.target[idx]
            }
        }
        
        # Let's skip the missing mask for simplicity in this example
        # In a real implementation, you would want to make sure the mask
        # is properly shaped to match what the model expects
        
        return item


def main():
    # Generate synthetic data
    np.random.seed(42)
    data, target, num_bins, n_risks = generate_competing_risks_data(
        n_samples=500,
        n_features=10,
        num_risks=3,
        include_categorical=True,
        missing_rate=0.1
    )
    
    # Split data into train, validation, and test sets
    train_data, test_data, train_target, test_target = train_test_split(
        data, target, test_size=0.2, random_state=42
    )
    
    train_data, val_data, train_target, val_target = train_test_split(
        train_data, train_target, test_size=0.2, random_state=42
    )
    
    # Initialize data processor
    processor = DataProcessor(
        num_impute_strategy='mean',
        cat_impute_strategy='most_frequent',
        normalize='robust',
        time_features=None,
        cat_embed_dim='auto',
        create_missing_indicators=True
    )
    
    # Fit processor on training data
    processor.fit(train_data)
    
    # Create datasets
    train_dataset = CompetingRisksDataset(train_data, train_target, processor)
    val_dataset = CompetingRisksDataset(val_data, val_target, processor)
    test_dataset = CompetingRisksDataset(test_data, test_target, processor)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )
    
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize competing risks task head
    competing_risks_head = CompetingRisksHead(
        name='competing_risks',
        input_dim=128,
        num_time_bins=num_bins,
        num_risks=n_risks,
        alpha_rank=0.1,
        alpha_calibration=0.0,
        task_weight=1.0,
        use_softmax=True,
        use_cause_specific=True
    )
    
    # Initialize model
    model = EnhancedDeepHit(
        num_continuous=len(train_dataset.continuous_cols),
        targets=[competing_risks_head],
        cat_feat_info=train_dataset.cat_feat_info,
        encoder_dim=128,
        encoder_depth=4,
        encoder_heads=8,
        encoder_feature_interaction=True,
        include_variational=False,
        device=device
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-3,
        weight_decay=1e-4,
        num_epochs=5,  # Reduced epochs for faster testing
        patience=2
    )
    
    # Evaluate on test set
    test_loss, test_metrics = model.evaluate(test_loader)
    print(f"\nTest Loss: {test_loss:.4f}")
    print("Test Metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig('competing_risks_training_history.png')
    
    # Save model
    model_path = 'model_outputs/competing_risks_model'
    os.makedirs('model_outputs', exist_ok=True)
    model.save(model_path, save_processor=True, processor=processor)
    print(f"Model saved to '{model_path}.pt'")

    
    # Generate predictions for a sample
    sample_idx = 0
    sample = test_dataset[sample_idx]
    
    continuous = sample['continuous'].unsqueeze(0).to(device)
    
    # Get predictions (without missing mask)
    predictions = model.predict(continuous)
    
    # Extract cumulative incidence functions and overall survival
    cr_output = predictions['task_outputs']['competing_risks']
    cif = cr_output['cif'].squeeze().cpu().numpy()  # [n_risks, num_bins]
    overall_survival = cr_output['overall_survival'].squeeze().cpu().numpy()  # [num_bins]
    
    # Plot results
    time_points = np.arange(num_bins)
    
    # Plot overall survival
    plt.figure(figsize=(10, 6))
    plt.step(time_points, overall_survival, where='post', label='Overall Survival')
    plt.xlabel('Time Bin')
    plt.ylabel('Probability')
    plt.title('Overall Survival Function')
    plt.grid(True)
    plt.legend()
    plt.savefig('competing_risks_overall_survival.png')
    
    # Plot cumulative incidence functions
    plt.figure(figsize=(10, 6))
    
    risk_colors = ['blue', 'red', 'green', 'purple', 'orange']
    for i in range(n_risks):
        plt.step(
            time_points, 
            cif[i], 
            where='post', 
            label=f'Risk {i+1}',
            color=risk_colors[i % len(risk_colors)]
        )
    
    plt.xlabel('Time Bin')
    plt.ylabel('Cumulative Incidence')
    plt.title('Cumulative Incidence Functions')
    plt.grid(True)
    plt.legend()
    plt.savefig('competing_risks_cif.png')
    
    # Compute uncertainty estimates
    print("\nComputing uncertainty estimates...")
    uncertainty = model.compute_uncertainty(
        continuous, num_samples=10  # Reduced samples for faster testing
    )
    
    cr_uncertainty = uncertainty.get('competing_risks', {})
    if 'std' in cr_uncertainty and 'mean' in cr_uncertainty:
        cif_std = cr_uncertainty['std'].squeeze().cpu().numpy()
        
        # Plot CIF with uncertainty for the first cause
        plt.figure(figsize=(10, 6))
        plt.step(time_points, cif[0], where='post', label=f'Risk 1 CIF', color='blue')
        plt.fill_between(
            time_points, 
            np.clip(cif[0] - 2 * cif_std[0], 0, 1), 
            np.clip(cif[0] + 2 * cif_std[0], 0, 1),
            alpha=0.3, 
            color='blue',
            label='95% Confidence Interval'
        )
        plt.xlabel('Time Bin')
        plt.ylabel('Cumulative Incidence')
        plt.title('CIF with Uncertainty (Risk 1)')
        plt.grid(True)
        plt.legend()
        plt.savefig('competing_risks_cif_uncertainty.png')
    
    print("Done!")


if __name__ == "__main__":
    main()