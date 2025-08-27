#!/usr/bin/env python

import os
import h5py
import torch
import wandb
import pickle
import logging
import warnings
import numpy as np
from torch.utils.data import TensorDataset,DataLoader

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

FILEDIR  = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/processed'
SAVEDIR  = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/results'
MODELDIR = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/models/nn'
CONFIGS  = [
    {'name':'exp_1','inputvars':['bl'],'description':'BL only'},
    {'name':'exp_2','inputvars':['cape','subsat'],'description':'CAPE + SUBSAT'},
    {'name':'exp_3','inputvars':['capeprofile'],'description':'CAPE Profile'},
    {'name':'exp_4','inputvars':['subsatprofile'],'description':'SUBSAT Profile'},
    {'name':'exp_5','inputvars':['capeprofile','subsatprofile'],'description':'Both Profiles'},
    {'name':'exp_6','inputvars':['t','q'],'description':'$T$ + $q$'}]
        
class NNMODEL:
    
    def __init__(self,inputsize,batchsize=64000,epochs=30,criterion=torch.nn.L1Loss(),learningrate=0.0001,patience=3):
        '''
        Purpose: Initialize a NN model for precipitation prediction.
        Args:
        - inputsize (int): number of input features
        - batchsize (int): the batch size for training (defaults to 64,000)
        - epochs (int): maximum number of training epochs (defaults to 30)
        - criterion (torch.nn.Module): the loss function (defaults to L1/MAE loss)
        - learningrate (float): learning rate (defaults to 0.0001)
        - patience (int): early stopping patience (defaults to 3)
        '''
        self.inputsize    = inputsize
        self.batchsize    = batchsize
        self.epochs       = epochs
        self.criterion    = criterion
        self.learningrate = learningrate
        self.patience     = patience
        self.device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model        = torch.nn.Sequential(
            torch.nn.Linear(self.inputsize,256),torch.nn.BatchNorm1d(256),torch.nn.GELU(),
            torch.nn.Linear(256,128),torch.nn.BatchNorm1d(128),torch.nn.GELU(),
            torch.nn.Linear(128,64),torch.nn.BatchNorm1d(64),torch.nn.GELU(),
            torch.nn.Linear(64,32),torch.nn.BatchNorm1d(32),torch.nn.GELU(),
            torch.nn.Linear(32,1)).to(self.device)
        self.bestloss     = float('inf')
        self.bestepoch    = 0
            
    def fit(self,Xtrain,Xvalid,ytrain,yvalid):
        '''
        Purpose: Train a NN model.
        Args:
        - Xtrain (torch.Tensor): training input features
        - Xvalid (torch.Tensor): validation input features
        - ytrain (torch.Tensor): training target precipitation values
        - yvalid (torch.Tensor): validation target precipitation values
        '''
        trainloader = DataLoader(TensorDataset(Xtrain,ytrain),batch_size=self.batchsize,shuffle=True,num_workers=8,pin_memory=True)
        validloader = DataLoader(TensorDataset(Xvalid,yvalid),batch_size=self.batchsize,shuffle=False,num_workers=8,pin_memory=True)
        optimizer   = torch.optim.Adam(self.model.parameters(),lr=self.learningrate)
        scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=self.patience)
        counter     = 0
        for epoch in range(self.epochs):
            self.model.train()
            trainloss = 0.0
            for Xbatch,ybatch in trainloader:
                Xbatch = Xbatch.to(self.device)
                ybatch = ybatch.to(self.device)
                optimizer.zero_grad()
                loss   = self.criterion(self.model(Xbatch).squeeze(),ybatch.squeeze())
                loss.backward()
                optimizer.step()
                trainloss += loss.item()*Xbatch.size(0)
            trainloss /= len(trainloader.dataset)
            self.model.eval()
            validloss = 0.0
            with torch.no_grad():
                for Xbatch,ybatch in validloader:
                    Xbatch = Xbatch.to(self.device)
                    ybatch = ybatch.to(self.device)
                    loss   = self.criterion(self.model(Xbatch).squeeze(),ybatch.squeeze())
                    validloss += loss.item()*Xbatch.size(0)
            validloss /= len(validloader.dataset)
            scheduler.step(validloss)
            wandb.log({
                'epoch':epoch+1,
                'train_loss':trainloss,
                'valid_loss':validloss,
                'learning_rate':optimizer.param_groups[0]['lr']})
            if validloss<self.bestloss:
                counter = 0
                self.bestloss  = validloss
                self.bestepoch = epoch+1
                self.bestmodel = self.model.state_dict().copy()
                wandb.log({
                    'best_epoch':self.bestepoch,
                    'best_valid_loss':self.bestloss})
            else:
                counter += 1
                if counter>=self.patience:
                    logger.info(f'   Early stopping at epoch {epoch+1}')
                    break
        self.model.load_state_dict(self.bestmodel)
    
    def predict(self,X):
        '''
        Purpose: Generate precipitation predictions using the trained model.
        Args:
        - X (torch.Tensor): input features for prediction
        Returns:
        - numpy.ndarray: predicted precipitation values
        '''
        self.model.eval()
        with torch.no_grad():
            X     = X.to(self.device)
            ypred = self.model(X).squeeze()
        return ypred.cpu().numpy()
    
    def save(self,name,modeldir=MODELDIR):
        '''
        Purpose: Save the best trained model to a PyTorch state dictionary in the specified directory.
        Args:
        - name (str): name of the configuration
        - modeldir (str): directory where the model should be saved (defaults to MODELDIR)
        '''
        filename = f'{name}_best.pth'
        filepath = os.path.join(modeldir,filename)
        torch.save(self.model.state_dict(),filepath)
        logger.info(f'   Model saved to {filepath}')

def load(inputvars,filename,filedir=FILEDIR):
    '''
    Purpose: Load NN data splits from an HDF5 file for specific input variables.
    Args:
    - inputvars (list): list of input variables to load
    - filename (str): name of the HDF5 file
    - filedir (str): directory containing the HDF5 file (defaults to FILEDIR)
    Returns:
    - (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor): inputs and target tensors organized by data split
    '''
    filepath = os.path.join(filedir,filename)
    with h5py.File(filepath,'r') as f:
        Xtraindata = []
        Xvaliddata = []
        Xtestdata  = []
        for inputvar in inputvars:
            Xtrainarray = f[f'{inputvar}_train'][:]
            Xvalidarray = f[f'{inputvar}_valid'][:]
            Xtestarray  = f[f'{inputvar}_test'][:]
            Xtraindata.append(Xtrainarray)
            Xvaliddata.append(Xvalidarray)
            Xtestdata.append(Xtestarray)
        Xtrain = torch.tensor(np.concatenate(Xtraindata,axis=1),dtype=torch.float32)
        Xvalid = torch.tensor(np.concatenate(Xvaliddata,axis=1),dtype=torch.float32)
        Xtest  = torch.tensor(np.concatenate(Xtestdata,axis=1),dtype=torch.float32)
        ytrain = torch.tensor(f['pr_train'][:],dtype=torch.float32)
        yvalid = torch.tensor(f['pr_valid'][:],dtype=torch.float32)
        ytest  = torch.tensor(f['pr_test'][:],dtype=torch.float32)
    return Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest

def process(filename,configs=CONFIGS):
    '''
    Purpose: Train and evaluate NN models with multiple input variable configurations.
    Args:
    - filename (str): name of the HDF5 file
    - configs (list): model configurations specifying input variables and descriptions (defaults to CONFIGS)
    Returns:
    - dict: dictionary containing NN model results
    '''
    results = {}
    for config in configs:
        name        = config['name']
        inputvars   = config['inputvars']
        description = config['description']
        logger.info(f'   Running {name}')
        Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest = load(inputvars,filename)
        model = NNMODEL(Xtrain.shape[1])
        wandb.init(
            project='Standard NNs',
            name=name,
            config={
                'name':name,
                'input_variables':inputvars,
                'loss_function':str(model.criterion.__class__.__name__),
                'n_params':sum(p.numel() for p in model.model.parameters()),
                'batch_size':model.batchsize,
                'patience':model.patience})
        model.fit(Xtrain,Xvalid,ytrain,yvalid)
        model.save(name)
        results[name] = {
            'description':description,
            'n_params':sum(p.numel() for p in model.model.parameters()),
            'y_pred':model.predict(Xtest)}
        wandb.finish()
        del model,Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results

if __name__=='__main__':
    try:
        logger.info('Training NN models...')
        results = process('ml_data_subset.h5')
        logger.info('Saving results...')
        with open(f'{SAVEDIR}/nn_standard_subset_results.pkl', 'wb') as f:
            pickle.dump(results,f)
        del results
        logger.info('Script execution completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')