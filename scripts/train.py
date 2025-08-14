import wandb
import torch
import h5py
import json
import warnings
import numpy as np
from torch.utils.data import TensorDataset,DataLoader

warnings.filterwarnings('ignore')

def load_hdf5_data():
    datapath     = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/processed/datasplits.h5'
    metadatapath = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/processed/metadata.json'
    with open(metadatapath,'r') as f:
        metadata = json.load(f)
    with h5py.File(datapath,'r') as f:
        Xtrain = torch.tensor(f['X_train'][:],dtype=torch.float32)  
        Xvalid = torch.tensor(f['X_valid'][:],dtype=torch.float32)  
        # Xtest  = torch.tensor(f['X_test'][:],dtype=torch.float32)  
        ytrain = torch.tensor(f['y_train'][:],dtype=torch.float32)  
        yvalid = torch.tensor(f['y_valid'][:],dtype=torch.float32)  
        # ytest  = torch.tensor(f['y_test'][:],dtype=torch.float32)  

    # Remove rows with NaN values
    train_mask = ~torch.isnan(Xtrain).any(dim=1)
    valid_mask = ~torch.isnan(Xvalid).any(dim=1)
    
    Xtrain = Xtrain[train_mask]
    ytrain = ytrain[train_mask]
    Xvalid = Xvalid[valid_mask]
    yvalid = yvalid[valid_mask]
    
    print(f"After removing NaN rows:")
    print(f"Training samples: {Xtrain.shape[0]:,} (removed {(~train_mask).sum():,})")
    print(f"Validation samples: {Xvalid.shape[0]:,} (removed {(~valid_mask).sum():,})")
    
    return Xtrain,Xvalid,ytrain,yvalid,metadata

def select_variables_for_experiment(X,featuremapping,selectedfeatures):
    selectedcols   = []
    selectedXnames = []
    for feature in selectedfeatures:
        info = featuremapping[feature]
        selectedcols.extend(info['columns'])
        selectedXnames.extend(info['feature_names'])
        print(f'  Selected {feature}: {info["n_features"]} features (columns {info["columns"][0]}-{info["columns"][-1]})')
    Xselected = X[:,selectedcols]
    print(f'  Total: {len(selectedfeatures)} variables, {len(selectedcols)} features, shape: {Xselected.shape}')
    return Xselected,selectedXnames,selectedcols

def get_data_loaders(Xtrain,Xvalid,ytrain,yvalid,batchsize):
    traindataset = TensorDataset(Xtrain,ytrain)
    validdataset = TensorDataset(Xvalid,yvalid)
    trainloader  = DataLoader(traindataset,batch_size=batchsize,shuffle=True,num_workers=8,pin_memory=True)
    validloader  = DataLoader(validdataset,batch_size=batchsize,shuffle=False,num_workers=8,pin_memory=True)
    return trainloader,validloader

def train_loop(trainloader,model,optimizer,criterion,device):
    model.train()
    epochtrainloss = 0.0
    for Xtrue,ytrue in trainloader:
        Xtrue,ytrue = Xtrue.to(device,non_blocking=True),ytrue.to(device,non_blocking=True)
        optimizer.zero_grad()
        ypred = model(Xtrue).squeeze()
        loss  = criterion(ypred,ytrue)
        loss.backward()
        optimizer.step()
        epochtrainloss += loss.item()*Xtrue.size(0)
    return epochtrainloss/len(trainloader.dataset)

def valid_loop(validloader,model,criterion,device):
    model.eval()
    epochvalidloss = 0.0
    with torch.no_grad():
        for Xtrue,ytrue in validloader:
            Xtrue,ytrue = Xtrue.to(device,non_blocking=True),ytrue.to(device,non_blocking=True)
            ypred = model(Xtrue).squeeze()
            loss = criterion(ypred,ytrue)
            epochvalidloss += loss.item()*Xtrue.size(0)
    return epochvalidloss/len(validloader.dataset)

def run_experiment(expnum,expname,selectedfeatures,featuremapping,Xtrain,Xvalid,ytrain,yvalid,nepochs=30,learningrate=0.0001):
    wandb.login()
    print(f'\nSelecting features for {expname}:')
    Xtrainexp,selectedXnames,_ = select_variables_for_experiment(Xtrain,featuremapping,selectedfeatures)
    Xvalidexp, _,_ = select_variables_for_experiment(Xvalid,featuremapping,selectedfeatures)
    inputsize = Xtrainexp.shape[1]
    if inputsize==1:
        batchsize = 100000
    elif inputsize<=10:
        batchsize = 64000
    elif inputsize<=50:
        batchsize = 32000
    else:
        batchsize = 16000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run = wandb.init(
        project='8-13-2025',
        name=f'Experiment {expnum} (MAE Deep)',
        config={
            'experiment_number':expnum,
            'selected_variables':selectedfeatures,
            'feature_names':selectedXnames,
            'n_variables':len(selectedfeatures),
            'n_features':inputsize,
            'n_epochs':nepochs,
            'learning_rate':learningrate,
            'batch_size':batchsize,
            'architecture':'MAE Deep'})
    trainloader,validloader = get_data_loaders(Xtrainexp,Xvalidexp,ytrain,yvalid,batchsize)
    model = torch.nn.Sequential(
        torch.nn.Linear(inputsize,256),
        torch.nn.BatchNorm1d(256),
        torch.nn.GELU(),
        torch.nn.Linear(256,128),
        torch.nn.BatchNorm1d(128),
        torch.nn.GELU(),
        torch.nn.Linear(128,64),
        torch.nn.BatchNorm1d(64),
        torch.nn.GELU(),
        torch.nn.Linear(64,32),
        torch.nn.BatchNorm1d(32),
        torch.nn.GELU(),
        torch.nn.Linear(32,1)
    ).to(device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learningrate)
    for epoch in range(nepochs):
        trainloss = train_loop(trainloader,model,optimizer,criterion,device)
        validloss = valid_loop(validloader,model,criterion,device)
        print(f'[Epoch {epoch+1}/{nepochs}] Train Loss: {trainloss:.6f}; Valid Loss: {validloss:.6f}')
        wandb.log({'train_loss':trainloss,'valid_loss':validloss})
    wandb.log({'final_train_loss':trainloss,'final_valid_loss':validloss,'n_parameters': sum(p.numel() for p in model.parameters())})
    wandb.finish()
    return model

if __name__=='__main__':
    print('Loading HDF5 data...')
    Xtrain,Xvalid,ytrain,yvalid,metadata = load_hdf5_data()

    featuremapping    = metadata['feature_mapping']
    availablefeatures = list(featuremapping.keys())
    print(f'Training samples: {Xtrain.shape[0]:,} points')
    print(f'Validation samples: {Xvalid.shape[0]:,} points')
    print(f'Total features: {Xtrain.shape[1]}')
    print(f'Available variables: {availablefeatures}')
    print(f'\nVariable feature mapping:')
    for feature,info in featuremapping.items():
        print(f'{feature}: columns {info["columns"][0]}-{info["columns"][-1]} ({info["n_features"]} features)')
    experiments = {
        '1':{'features':['bl'],'description':'BL'},
        '2':{'features':['cape','subsat'],'description':'CAPE and SUBSAT'},
        '3':{'features':['capeprofile'],'description':'CAPE Profile'},
        '4':{'features':['subsatprofile'],'description':'SUBSAT Profile'},
        '5':{'features':['capeprofile','subsatprofile'],'description':'CAPE and SUBSAT Profiles'},
        '6':{'features':['t','q'],'description':'Temperature and Specific Humidity Profiles'},
    }
    for expnum,expconfig in experiments.items():
        selectedfeatures = expconfig['features']
        description      = expconfig['description']
        expname = f'Experiment {expnum}'
        print(f'\n{"="*60}')
        print(f'Running {expname}: {description}')
        print(f'Features: {selectedfeatures}')
        model = run_experiment(expnum,expname,selectedfeatures,featuremapping,Xtrain,Xvalid,ytrain,yvalid)
        print(f'Completed {expname}')
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print('\nAll experiments completed!')