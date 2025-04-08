import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from enhanced_deephit.data.processing import DataProcessor
from enhanced_deephit.models import EnhancedDeepHit
from enhanced_deephit.models.tasks.survival import SingleRiskHead

# Try to import CompetingRisksHead if available
try:
    from enhanced_deephit.models.tasks.survival import CompetingRisksHead
    COMPETING_RISKS_AVAILABLE = True
except ImportError:
    COMPETING_RISKS_AVAILABLE = False
    print("CompetingRisksHead is not available. Competing risks demonstration will use simulated data.")
from enhanced_deephit.visualization.survival_plots import (
    plot_survival_curve,
    plot_cumulative_incidence,
    plot_calibration_curve
)
from enhanced_deephit.visualization.feature_effects import (
    plot_partial_dependence,
    plot_ice_curves,
    plot_feature_interaction
)

def generate_synthetic_data(n_samples=500, n_features=5, competing_risks=False, seed=42):
    """Generate synthetic survival data."""
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate survival times with first and third features being most important
    risk_scores = 2 * X[:, 0] + 1.5 * X[:, 2]
    survival_times = np.random.exponential(scale=np.exp(-risk_scores))
    censoring_times = np.random.exponential(scale=2)
    
    # Determine observed time and event indicator
    times = np.minimum(survival_times, censoring_times)
    events = (survival_times <= censoring_times).astype(np.float32)
    
    # Discretize time into bins
    num_bins = 10
    max_time = np.percentile(times, 99)
    bin_edges = np.linspace(0, max_time, num_bins + 1)
    time_bins = np.digitize(times, bin_edges) - 1
    time_bins = np.clip(time_bins, 0, num_bins - 1)
    
    # Create dataframe
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['time'] = times
    df['event'] = events
    df['time_bin'] = time_bins
    
    # For competing risks, add cause
    if competing_risks:
        # Create two causes (0 and 1)
        causes = np.random.choice([0, 1], size=n_samples)
        df['cause'] = np.where(events, causes, -1)
    
    return df, num_bins, bin_edges

def prepare_data(df, num_bins, competing_risks=False):
    """Prepare data for the model."""
    # Create data processor
    processor = DataProcessor(
        num_impute_strategy='mean',
        normalize='robust'
    )
    
    # Only use feature columns for fitting, not target columns (time, event, time_bin)
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    processor.fit(df[feature_cols])
    
    # Process features
    df_processed = processor.transform(df[feature_cols])
    
    # Extract features and convert to tensor
    X_tensor = torch.tensor(df_processed.values, dtype=torch.float32)
    
    if not competing_risks:
        # Create target format for single risk
        # [event_indicator, time_bin, one_hot_encoding]
        target = np.zeros((len(df), 2 + num_bins))
        target[:, 0] = df['event'].values
        target[:, 1] = df['time_bin'].values
        
        # One-hot encoding of time
        for i in range(len(df)):
            if df['event'].iloc[i]:
                # For events, mark the event time
                target[i, 2 + int(df['time_bin'].iloc[i])] = 1
            else:
                # For censored, mark all times after censoring as unknown (-1)
                target[i, 2 + int(df['time_bin'].iloc[i]):] = -1
    else:
        # Create target format for competing risks
        # [event_indicator, time_bin, cause_index, one_hot_encoding]
        target = np.zeros((len(df), 3 + 2 * num_bins))
        target[:, 0] = df['event'].values
        target[:, 1] = df['time_bin'].values
        target[:, 2] = df['cause'].values  # Cause index (-1 for censored)
    
    # Convert to tensor
    target_tensor = torch.tensor(target, dtype=torch.float32)
    
    return X_tensor, target_tensor, processor

def create_and_train_models(X_tensor, target_tensor, num_bins, competing_risks=False):
    """Create and train the DeepHit model(s)."""
    # Create dataset and dataloader with proper formatting
    class SurvivalDataset(torch.utils.data.Dataset):
        def __init__(self, X, targets, task_name='survival'):
            self.X = X
            self.targets = targets
            self.task_name = task_name
            
        def __len__(self):
            return len(self.X)
            
        def __getitem__(self, idx):
            return {
                'continuous': self.X[idx],
                'targets': {
                    self.task_name: self.targets[idx]
                }
            }
    
    models = {}
    
    # Single risk model
    if not competing_risks or competing_risks == 'both':
        # Create single risk model
        task_head = SingleRiskHead(
            name='survival',
            input_dim=64,
            num_time_bins=num_bins,
            alpha_rank=0.1
        )
        
        single_model = EnhancedDeepHit(
            num_continuous=X_tensor.shape[1],
            targets=[task_head],
            encoder_dim=64,
            encoder_depth=2,
            encoder_heads=4,
            include_variational=True,
            device='cpu'
        )
        
        # Create dataset and dataloader
        dataset = SurvivalDataset(X_tensor, target_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train model
        print("Training single risk model...")
        single_model.fit(
            train_loader=loader,
            num_epochs=5,
            learning_rate=0.001
        )
        
        models['single'] = single_model
    
    # Competing risks model
    if competing_risks and COMPETING_RISKS_AVAILABLE:
        # Create competing risks model
        cr_task_head = CompetingRisksHead(
            name='competing_risks',
            input_dim=64,
            num_time_bins=num_bins,
            num_risks=2,
            alpha_rank=0.1
        )
        
        cr_model = EnhancedDeepHit(
            num_continuous=X_tensor.shape[1],
            targets=[cr_task_head],
            encoder_dim=64,
            encoder_depth=2,
            encoder_heads=4,
            include_variational=True,
            device='cpu'
        )
        
        # Create dataset and dataloader
        cr_dataset = SurvivalDataset(X_tensor, target_tensor, task_name='competing_risks')
        cr_loader = torch.utils.data.DataLoader(cr_dataset, batch_size=32, shuffle=True)
        
        # Train model
        print("Training competing risks model...")
        cr_model.fit(
            train_loader=cr_loader,
            num_epochs=5,
            learning_rate=0.001
        )
        
        models['competing'] = cr_model
    elif competing_risks:
        print("CompetingRisksHead is not available. Skipping competing risks model training.")
        
        # Create simulated data for visualization
        print("Creating simulated competing risks data for visualization...")
        # Create simulated CIF and survival curves
        time_points = np.linspace(0, 10, num_bins)
        
        # Function to create simulated monotonic CIF
        def create_simulated_cif(n_patients=5, n_risks=2, n_timepoints=num_bins):
            # Create base hazard rates that increase over time
            base_hazards = np.zeros((n_risks, n_timepoints))
            for r in range(n_risks):
                # Different progression rate for each risk
                rate = 0.05 + r * 0.03
                base_hazards[r] = rate * (1 + np.arange(n_timepoints) * 0.15)
            
            # Convert to cumulative incidence using cumulative sum
            cif = np.zeros((n_patients, n_risks, n_timepoints))
            for p in range(n_patients):
                # Vary the progression rate for each patient
                patient_factor = 0.7 + 0.6 * np.random.random()
                for r in range(n_risks):
                    # Cumulative sum and clip to ensure monotonic increase
                    cif[p, r] = np.clip(np.cumsum(base_hazards[r] * patient_factor), 0, 0.9 / n_risks)
            
            # Calculate overall survival as 1 - sum of CIFs
            survival = 1 - np.sum(cif, axis=1)
            
            return cif, survival
        
        # Create simulated data
        class SimulatedCompetingRisksModel:
            def __init__(self, n_patients=100, n_risks=2, n_timepoints=num_bins):
                self.n_patients = n_patients
                self.n_risks = n_risks
                self.n_timepoints = n_timepoints
                self.time_points = np.linspace(0, 10, n_timepoints)
                
                # Create simulated data
                self.all_cif, self.all_survival = create_simulated_cif(
                    n_patients=n_patients, 
                    n_risks=n_risks, 
                    n_timepoints=n_timepoints
                )
            
            def predict(self, X):
                # Return simulated predictions
                batch_size = X.shape[0]
                indices = np.random.choice(self.n_patients, size=batch_size)
                
                return {
                    'task_outputs': {
                        'competing_risks': {
                            'cif': self.all_cif[indices],
                            'overall_survival': self.all_survival[indices]
                        }
                    }
                }
                
            def compute_uncertainty(self, X, num_samples=5):
                # Return simulated uncertainty
                batch_size = X.shape[0]
                # Create varying predictions
                cif_samples = []
                survival_samples = []
                
                for _ in range(num_samples):
                    cif, survival = create_simulated_cif(
                        n_patients=batch_size, 
                        n_risks=self.n_risks, 
                        n_timepoints=self.n_timepoints
                    )
                    cif_samples.append(cif)
                    survival_samples.append(survival)
                
                # Calculate standard deviation
                cif_std = np.std(np.array(cif_samples), axis=0)
                survival_std = np.std(np.array(survival_samples), axis=0)
                
                return {
                    'competing_risks': {
                        'std': cif_std,
                        'survival_std': survival_std
                    }
                }
        
        # Create simulated model
        models['competing'] = SimulatedCompetingRisksModel(
            n_patients=100, 
            n_risks=2, 
            n_timepoints=num_bins
        )
    
    return models

def visualize_survival_and_cif(models, X_tensor, bin_edges):
    """Visualize survival curves and cumulative incidence functions."""
    # Directory for saving plots
    os.makedirs('plots', exist_ok=True)
    
    # Generate predictions
    if 'single' in models:
        print("\nGenerating predictions for survival curves...")
        model = models['single']
        
        # Get predictions
        with torch.no_grad():
            preds = model.predict(X_tensor)
            uncertainty = model.compute_uncertainty(X_tensor, num_samples=5)
        
        # Survival curves
        survival_curves = preds['task_outputs']['survival']['survival'].numpy()
        time_points = bin_edges[:-1]  # Use bin start points as time points
        
        # Plot for a single patient
        print("Plotting survival curve for single patient...")
        fig1 = plot_survival_curve(
            survival_curves[0], 
            time_points=time_points,
            title="Survival Curve for Patient 0"
        )
        fig1.savefig("plots/survival_curve_single.png")
        plt.close(fig1)
        
        # Plot for multiple patients
        print("Plotting survival curves for multiple patients...")
        fig2 = plot_survival_curve(
            survival_curves[:5], 
            time_points=time_points,
            labels=[f'Patient {i}' for i in range(5)],
            title="Survival Curves for Multiple Patients"
        )
        fig2.savefig("plots/survival_curves_multiple.png")
        plt.close(fig2)
        
        # Plot with uncertainty
        print("Plotting survival curve with uncertainty...")
        uncertainty_std = uncertainty['survival']['std'].numpy()[0]
        fig3 = plot_survival_curve(
            survival_curves[0],
            time_points=time_points,
            uncertainty=uncertainty_std,
            title="Survival Curve with Uncertainty"
        )
        fig3.savefig("plots/survival_curve_uncertainty.png")
        plt.close(fig3)
    
    # Cumulative incidence functions
    if 'competing' in models:
        print("\nGenerating predictions for cumulative incidence functions...")
        model = models['competing']
        
        # Get predictions
        with torch.no_grad():
            cr_preds = model.predict(X_tensor)
            cr_uncertainty = model.compute_uncertainty(X_tensor, num_samples=5)
        
        # Cumulative incidence functions
        if isinstance(cr_preds['task_outputs']['competing_risks']['cif'], torch.Tensor):
            cif = cr_preds['task_outputs']['competing_risks']['cif'].numpy()
        else:
            # Already numpy array
            cif = cr_preds['task_outputs']['competing_risks']['cif']
        
        # Plot for a single patient
        print("Plotting cumulative incidence for single patient...")
        fig4 = plot_cumulative_incidence(
            cif[0], 
            time_points=time_points,
            risk_names=['Cause 1', 'Cause 2'],
            title="Cumulative Incidence for Patient 0"
        )
        fig4.savefig("plots/cif_single.png")
        plt.close(fig4)
        
        # Plot with uncertainty
        print("Plotting cumulative incidence with uncertainty...")
        if isinstance(cr_uncertainty['competing_risks']['std'], torch.Tensor):
            uncertainty_std = cr_uncertainty['competing_risks']['std'].numpy()[0]
        else:
            uncertainty_std = cr_uncertainty['competing_risks']['std'][0]
            
        fig5 = plot_cumulative_incidence(
            cif[0],
            time_points=time_points,
            risk_names=['Cause 1', 'Cause 2'],
            uncertainty=uncertainty_std,
            title="Cumulative Incidence with Uncertainty"
        )
        fig5.savefig("plots/cif_uncertainty.png")
        plt.close(fig5)
        
        # Stacked CIF plot
        print("Plotting stacked cumulative incidence...")
        fig6 = plot_cumulative_incidence(
            cif[0],
            time_points=time_points,
            risk_names=['Cause 1', 'Cause 2'],
            stacked=True,
            title="Stacked Cumulative Incidence"
        )
        fig6.savefig("plots/cif_stacked.png")
        plt.close(fig6)

def visualize_feature_effects(model, X_tensor, feature_names):
    """Visualize feature effects."""
    # Partial dependence plot
    print("\nGenerating partial dependence plot...")
    fig1 = plot_partial_dependence(
        model,
        X_tensor,
        feature_idx=0,  # Feature 0 (one of the most important)
        feature_name=feature_names[0],
        target='risk_score',
        title=f"Partial Dependence of Risk Score on {feature_names[0]}"
    )
    fig1.savefig("plots/partial_dependence.png")
    plt.close(fig1)
    
    # ICE curves
    print("Generating ICE curves...")
    fig2 = plot_ice_curves(
        model,
        X_tensor[:20],  # Use first 20 samples for clarity
        feature_idx=0,
        feature_name=feature_names[0],
        target='risk_score',
        title=f"Individual Conditional Expectation Curves for {feature_names[0]}"
    )
    fig2.savefig("plots/ice_curves.png")
    plt.close(fig2)
    
    # Feature interaction plot
    print("Generating feature interaction plot...")
    fig3 = plot_feature_interaction(
        model,
        X_tensor,
        feature1_idx=0,
        feature2_idx=2,  # The two most important features
        feature1_name=feature_names[0],
        feature2_name=feature_names[2],
        n_points=10,
        target='risk_score',
        title=f"Interaction Effect between {feature_names[0]} and {feature_names[2]}"
    )
    fig3.savefig("plots/feature_interaction.png")
    plt.close(fig3)

def main():
    print("Survival Visualization Example")
    print("=============================\n")
    
    # Step 1: Generate synthetic data
    print("Generating synthetic data...")
    df, num_bins, bin_edges = generate_synthetic_data(n_samples=500, n_features=5)
    df_cr, _, _ = generate_synthetic_data(n_samples=500, n_features=5, competing_risks=True)
    print(f"Generated data with {len(df)} samples and {df.shape[1]-3} features")
    
    # Step 2: Prepare data for the models
    print("\nPreparing data...")
    X_tensor, target_tensor, processor = prepare_data(df, num_bins)
    X_tensor_cr, target_tensor_cr, _ = prepare_data(df_cr, num_bins, competing_risks=True)
    
    # Feature names
    feature_names = [f'feature_{i}' for i in range(X_tensor.shape[1])]
    print(f"Features: {feature_names}")
    
    # Step 3: Create and train models
    models = create_and_train_models(X_tensor, target_tensor, num_bins)
    
    if COMPETING_RISKS_AVAILABLE:
        cr_models = create_and_train_models(X_tensor_cr, target_tensor_cr, num_bins, competing_risks=True)
        # Combine models
        all_models = {**models, **cr_models}
    else:
        print("Using simulated competing risks model instead")
        # Create simulated competing risks model
        cr_models = create_and_train_models(X_tensor_cr, target_tensor_cr, num_bins, competing_risks=True)
        all_models = {**models, **cr_models}
    
    # Step 4: Visualize survival curves and cumulative incidence functions
    visualize_survival_and_cif(all_models, X_tensor, bin_edges)
    
    # Step 5: Visualize feature effects
    visualize_feature_effects(models['single'], X_tensor, feature_names)
    
    print("\nVisualization example completed successfully.")
    print("Output images saved in the 'plots' directory")

if __name__ == "__main__":
    main()