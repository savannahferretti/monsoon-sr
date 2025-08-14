import os
import torch
import h5py
import json
import warnings
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')

# Directories
DATA_DIR = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/processed'
MODELS_DIR = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/models/v4'
RESULTS_DIR = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def create_model_architecture(inputsize):
    """Create the same model architecture used in training"""
    model = torch.nn.Sequential(
        torch.nn.Linear(inputsize, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.GELU(),
        torch.nn.Linear(256, 128),
        torch.nn.BatchNorm1d(128),
        torch.nn.GELU(),
        torch.nn.Linear(128, 64),
        torch.nn.BatchNorm1d(64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 32),
        torch.nn.BatchNorm1d(32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 1)
    )
    return model

def load_test_data():
    """Load test data and metadata"""
    datapath = os.path.join(DATA_DIR, 'datasplits.h5')
    metadatapath = os.path.join(DATA_DIR, 'metadata.json')
    
    with open(metadatapath, 'r') as f:
        metadata = json.load(f)
    
    with h5py.File(datapath, 'r') as f:
        inputs_test = torch.tensor(f['X_test'][:], dtype=torch.float32)
        targets_test = torch.tensor(f['y_test'][:], dtype=torch.float32)
    
    # Remove rows with NaN values
    test_mask = ~torch.isnan(inputs_test).any(dim=1)
    inputs_test = inputs_test[test_mask]
    targets_test = targets_test[test_mask]
    
    print(f"Test samples: {inputs_test.shape[0]:,} (removed {(~test_mask).sum():,} NaN rows)")
    
    return inputs_test, targets_test, metadata

def select_features_for_prediction(inputs, inputmapping, selectedfeatures):
    """Select the same features used during training"""
    selectedcols = []
    for feature in selectedfeatures:
        info = inputmapping[feature]
        selectedcols.extend(info['columns'])
    inputs_selected = inputs[:, selectedcols]
    return inputs_selected

def load_trained_model(experiment_num, inputsize):
    """Load a trained model"""
    model_path = os.path.join(MODELS_DIR, f'experiment_{experiment_num}_best_model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Create model with same architecture
    model = create_model_architecture(inputsize)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model

def predict_with_model(model, inputs, batch_size=32000):
    """Generate predictions using trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    dataset = TensorDataset(inputs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch_inputs, in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_pred = model(batch_inputs).squeeze()
            predictions.append(batch_pred.cpu())
    
    return torch.cat(predictions, dim=0)

def calculate_test_loss(predictions, targets, criterion):
    """Calculate test loss"""
    with torch.no_grad():
        loss = criterion(predictions, targets).item()
    return loss

def evaluate_all_experiments():
    """Evaluate all trained experiments on test data"""
    print("Loading test data...")
    inputs_test, targets_test, metadata = load_test_data()
    
    inputmapping = metadata['feature_mapping']
    
    # Define experiments (same as in training)
    experiments = {
        '1': {'features': ['bl'], 'description': 'BL'},
        '2': {'features': ['cape', 'subsat'], 'description': 'CAPE and SUBSAT'},
        # '3': {'features': ['capeprofile'], 'description': 'CAPE Profile'},
        # '4': {'features': ['subsatprofile'], 'description': 'SUBSAT Profile'},
        '5': {'features': ['capeprofile', 'subsatprofile'], 'description': 'CAPE and SUBSAT Profiles'},
        # '6': {'features': ['t', 'q'], 'description': 'Temperature and Specific Humidity Profiles'},
    }
    
    results = {}
    criterion = torch.nn.L1Loss()  # Same as training
    
    for expnum, expconfig in experiments.items():
        selectedfeatures = expconfig['features']
        description = expconfig['description']
        
        print(f"\nEvaluating Experiment {expnum}: {description}")
        
        try:
            # Select features for this experiment
            inputs_test_exp = select_features_for_prediction(inputs_test, inputmapping, selectedfeatures)
            inputsize = inputs_test_exp.shape[1]
            
            print(f"  Input size: {inputsize}")
            print(f"  Features: {selectedfeatures}")
            
            # Load trained model
            model = load_trained_model(expnum, inputsize)
            
            # Count parameters
            nparams = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {nparams:,}")
            
            # Generate predictions
            predictions = predict_with_model(model, inputs_test_exp)
            
            # Calculate test loss
            testloss = calculate_test_loss(predictions, targets_test, criterion)
            print(f"  Test Loss (MAE): {testloss:.6f}")
            
            # Store results
            results[expnum] = {
                'experiment_number': expnum,
                'description': description,
                'features': selectedfeatures,
                'input_size': inputsize,
                'n_parameters': nparams,
                'test_loss': testloss,
                'predictions': predictions.numpy(),
                'targets': targets_test.numpy()
            }
            
        except Exception as e:
            print(f"  Error evaluating experiment {expnum}: {e}")
            results[expnum] = {
                'experiment_number': expnum,
                'description': description,
                'features': selectedfeatures,
                'error': str(e)
            }
    
    return results

def save_results(results):
    """Save results to file"""
    results_path = os.path.join(RESULTS_DIR, 'v4_experiment_results.pkl')
    
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to: {results_path}")
    
    # Also save a summary CSV
    summary_data = []
    for expnum, result in results.items():
        if 'error' not in result:
            summary_data.append({
                'experiment': expnum,
                'description': result['description'],
                'features': ', '.join(result['features']),
                'input_size': result['input_size'],
                'n_parameters': result['n_parameters'],
                'test_loss_mae': result['test_loss']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(RESULTS_DIR, 'v4_experiment_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")
    
    return summary_df

def analyze_results(results):
    """Print analysis of results"""
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*60)
    
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not successful_results:
        print("No successful experiments found!")
        return
    
    # Sort by test loss
    sorted_results = sorted(successful_results.items(), key=lambda x: x[1]['test_loss'])
    
    print(f"{'Exp':<4} {'Description':<35} {'Features':<8} {'Params':<10} {'Test MAE':<10}")
    print("-" * 75)
    
    for expnum, result in sorted_results:
        # Fix: Format the parameter count separately
        params_str = f"{result['n_parameters']:,}"
        print(f"{expnum:<4} {result['description']:<35} {result['input_size']:<8} {params_str:<10} {result['test_loss']:<10.6f}")
    
    best_exp = sorted_results[0]
    print(f"\nBest performing experiment: {best_exp[0]} ({best_exp[1]['description']})")
    print(f"Test MAE: {best_exp[1]['test_loss']:.6f}")
    print(f"Parameters: {best_exp[1]['n_parameters']:,}")

    
if __name__ == '__main__':
    print("Evaluating all trained experiments...")
    
    # Evaluate all experiments
    results = evaluate_all_experiments()
    
    # Save results
    summary_df = save_results(results)
    
    # Analyze results
    analyze_results(results)
    
    print("\nEvaluation complete!")
    print(f"Results available in: {RESULTS_DIR}")