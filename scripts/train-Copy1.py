import wandb
import torch
import h5py
import json
import warnings
import numpy as np
import os
from torch.utils.data import TensorDataset,DataLoader

warnings.filterwarnings('ignore')

# Create models directory structure
MODELS_DIR = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/models/v4'
os.makedirs(MODELS_DIR, exist_ok=True)

def load_hdf5_data():
    datapath = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/processed/datasplits.h5'
    metadatapath = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/processed/metadata.json'
    
    with open(metadatapath, 'r') as f:
        metadata = json.load(f)
    
    with h5py.File(datapath, 'r') as f:
        inputs_train = torch.tensor(f['X_train'][:], dtype=torch.float32)
        inputs_valid = torch.tensor(f['X_valid'][:], dtype=torch.float32)
        targets_train = torch.tensor(f['y_train'][:], dtype=torch.float32)
        targets_valid = torch.tensor(f['y_valid'][:], dtype=torch.float32)

    # Remove rows with NaN values
    train_mask = ~torch.isnan(inputs_train).any(dim=1)
    valid_mask = ~torch.isnan(inputs_valid).any(dim=1)
    
    inputs_train = inputs_train[train_mask]
    targets_train = targets_train[train_mask]
    inputs_valid = inputs_valid[valid_mask]
    targets_valid = targets_valid[valid_mask]
    
    print(f"After removing NaN rows:")
    print(f"Training samples: {inputs_train.shape[0]:,} (removed {(~train_mask).sum():,})")
    print(f"Validation samples: {inputs_valid.shape[0]:,} (removed {(~valid_mask).sum():,})")
    
    return inputs_train, inputs_valid, targets_train, targets_valid, metadata

def select_variables_for_experiment(inputs, inputmapping, selectedfeatures):
    selectedcols = []
    selectedinputnames = []
    for feature in selectedfeatures:
        info = inputmapping[feature]
        selectedcols.extend(info['columns'])
        selectedinputnames.extend(info['feature_names'])
        print(f'  Selected {feature}: {info["n_features"]} features (columns {info["columns"][0]}-{info["columns"][-1]})')
    inputs_selected = inputs[:, selectedcols]
    print(f'  Total: {len(selectedfeatures)} variables, {len(selectedcols)} features, shape: {inputs_selected.shape}')
    return inputs_selected, selectedinputnames, selectedcols

def get_data_loaders(inputs_train, inputs_valid, targets_train, targets_valid, batchsize):
    traindataset = TensorDataset(inputs_train, targets_train)
    validdataset = TensorDataset(inputs_valid, targets_valid)
    trainloader = DataLoader(traindataset, batch_size=batchsize, shuffle=True, num_workers=8, pin_memory=True)
    validloader = DataLoader(validdataset, batch_size=batchsize, shuffle=False, num_workers=8, pin_memory=True)
    return trainloader, validloader

def train_loop(trainloader, model, optimizer, criterion, device):
    model.train()
    epochtrainloss = 0.0
    for inputs_batch, targets_batch in trainloader:
        inputs_batch, targets_batch = inputs_batch.to(device, non_blocking=True), targets_batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        targets_pred = model(inputs_batch).squeeze()
        loss = criterion(targets_pred, targets_batch)
        loss.backward()
        optimizer.step()
        epochtrainloss += loss.item() * inputs_batch.size(0)
    return epochtrainloss / len(trainloader.dataset)

def valid_loop(validloader, model, criterion, device):
    model.eval()
    epochvalidloss = 0.0
    with torch.no_grad():
        for inputs_batch, targets_batch in validloader:
            inputs_batch, targets_batch = inputs_batch.to(device, non_blocking=True), targets_batch.to(device, non_blocking=True)
            targets_pred = model(inputs_batch).squeeze()
            loss = criterion(targets_pred, targets_batch)
            epochvalidloss += loss.item() * inputs_batch.size(0)
    return epochvalidloss / len(validloader.dataset)

def run_experiment(expnum, expname, selectedfeatures, inputmapping, inputs_train, inputs_valid, targets_train, targets_valid, nepochs=30, learningrate=0.0001, patience=3):
    wandb.login()
    print(f'\nSelecting features for {expname}:')
    inputs_train_exp, selectedinputnames, _ = select_variables_for_experiment(inputs_train, inputmapping, selectedfeatures)
    inputs_valid_exp, _, _ = select_variables_for_experiment(inputs_valid, inputmapping, selectedfeatures)
    inputsize = inputs_train_exp.shape[1]
    
    if inputsize == 1:
        batchsize = 100000
    elif inputsize <= 10:
        batchsize = 64000
    elif inputsize <= 50:
        batchsize = 32000
    else:
        batchsize = 16000
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model first to get parameter count
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
    ).to(device)
    
    # Calculate number of parameters
    n_parameters = sum(p.numel() for p in model.parameters())
    
    run = wandb.init(
        project='MAE Deep',
        name=f'Experiment {expnum}',
        config={
            'experiment_number':expnum,
            'variable_names':selectedfeatures,
            'n_variables': len(selectedfeatures),
            'n_inputs': inputsize,
            'input_names':selectedinputnames,
            'learning_rate': learningrate,
            'batch_size': batchsize,
            'patience': patience,
            'n_parameters': n_parameters})
    
    trainloader, validloader = get_data_loaders(inputs_train_exp, inputs_valid_exp, targets_train, targets_valid, batchsize)
    
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True)
    
    # Early stopping variables
    best_valid_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(MODELS_DIR, f'experiment_{expnum}_best_model.pth')
    
    for epoch in range(nepochs):
        trainloss = train_loop(trainloader, model, optimizer, criterion, device)
        validloss = valid_loop(validloader, model, criterion, device)
        
        # Step the scheduler
        scheduler.step(validloss)
        
        print(f'[Epoch {epoch+1}/{nepochs}] Train Loss: {trainloss:.6f}; Valid Loss: {validloss:.6f}')
        wandb.log({'train_loss': trainloss, 'valid_loss': validloss, 'learning_rate': optimizer.param_groups[0]['lr']})
        
        # Early stopping and model saving
        if validloss < best_valid_loss:
            best_valid_loss = validloss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), best_model_path)
            print(f'  New best model saved with validation loss: {validloss:.6f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    # Load the best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f'Loaded best model with validation loss: {best_valid_loss:.6f}')
    
    wandb.log({
        'final_train_loss': trainloss,
        'final_valid_loss': validloss,
        'best_valid_loss': best_valid_loss,
        'n_parameters': n_parameters
    })
    wandb.finish()
    
    return model

if __name__ == '__main__':
    print('Loading HDF5 data...')
    inputs_train, inputs_valid, targets_train, targets_valid, metadata = load_hdf5_data()

    inputmapping = metadata['feature_mapping']
    availablefeatures = list(inputmapping.keys())
    
    print(f'Training samples: {inputs_train.shape[0]:,} points')
    print(f'Validation samples: {inputs_valid.shape[0]:,} points')
    print(f'Total features: {inputs_train.shape[1]}')
    print(f'Available variables: {availablefeatures}')
    print(f'\nVariable feature mapping:')
    
    for feature, info in inputmapping.items():
        print(f'{feature}: columns {info["columns"][0]}-{info["columns"][-1]} ({info["n_features"]} features)')
    
    experiments = {
        '1': {'features': ['bl'], 'description': 'BL'},
        '2': {'features': ['cape', 'subsat'], 'description': 'CAPE and SUBSAT'},
        # '3': {'features': ['capeprofile'], 'description': 'CAPE Profile'},
        # '4': {'features': ['subsatprofile'], 'description': 'SUBSAT Profile'},
        '5': {'features': ['capeprofile', 'subsatprofile'], 'description': 'CAPE and SUBSAT Profiles'},
        # '6': {'features': ['t', 'q'], 'description': 'Temperature and Specific Humidity Profiles'},
    }
    
    for expnum, expconfig in experiments.items():
        selectedfeatures = expconfig['features']
        description = expconfig['description']
        expname = f'Experiment {expnum}'
        print(f'\n{"="*60}')
        print(f'Running {expname}: {description}')
        print(f'Features: {selectedfeatures}')
        model = run_experiment(expnum, expname, selectedfeatures, inputmapping, inputs_train, inputs_valid, targets_train, targets_valid)
        print(f'Completed {expname}')
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print('\nAll experiments completed!')
    print(f'Best models saved in: {MODELS_DIR}')