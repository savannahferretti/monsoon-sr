#!/usr/bin/env python

import os
import time
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
    {'name':'exp_1','inputvars':['bl'],'description':'Experiment 1'},
    # {'name':'exp_2','inputvars':['cape','subsat'],'description':'Experiment 2'},
    # {'name':'exp_3','inputvars':['capeprofile'],'description':'Experiment 3'},
    # {'name':'exp_4','inputvars':['subsatprofile'],'description':'Experiment 4'},
    # {'name':'exp_5','inputvars':['capeprofile','subsatprofile'],'description':'Experiment 5'},
    # {'name':'exp_6','inputvars':['t','q'],'description':'Experiment 6'}
]

NORMTARGET  = True
LOG1PTARGET = True
        
class NNMODEL:
    
    def __init__(self,inputsize,batchsize=64000,epochs=30,criterion=torch.nn.MSELoss(),learningrate=0.0001,patience=3):
        '''
        Purpose: Initialize a NN model for precipitation prediction.
        Args:
        - inputsize (int): number of input features
        - batchsize (int): the batch size for training (defaults to 64,000)
        - epochs (int): maximum number of training epochs (defaults to 50)
        - criterion (torch.nn.Module): the loss function (defaults to MSE)
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
        self.beststate    = None
            
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
        starttime   = time.time()
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
            if not np.isfinite(trainloss):
                logger.warning('   Training loss is non-finite. Stopping...')
                break
            self.model.eval()
            validloss = 0.0
            with torch.no_grad():
                for Xbatch,ybatch in validloader:
                    Xbatch = Xbatch.to(self.device)
                    ybatch = ybatch.to(self.device)
                    loss   = self.criterion(self.model(Xbatch).squeeze(),ybatch.squeeze())
                    validloss += loss.item()*Xbatch.size(0)
            validloss /= len(validloader.dataset)
            if np.isfinite(validloss):
                scheduler.step(validloss)
            else:
                logger.warning('   Validation loss is non-finite. Stopping...')
                break
            wandb.log({
                'Epoch':epoch+1,
                'Training Loss':trainloss,
                'Validation Loss':validloss,
                'Learning Rate':optimizer.param_groups[0]['lr']})
            if validloss<self.bestloss:
                counter = 0
                self.bestloss  = validloss
                self.bestepoch = epoch+1
                self.beststate = self.model.state_dict().copy()
            else:
                counter += 1
                if counter>self.patience:
                    logger.info(f'   Early stopping at epoch {epoch+1}!')
                    break
        self.model.load_state_dict(self.beststate)
        trainingtime = time.time()-starttime
        wandb.run.summary.update({
            'Best Model at Epoch':self.bestepoch,
            'Best Validation Loss':self.bestloss,
            'Total Training Epochs':epoch+1,
            'Training Duration (s)':trainingtime,
            'Stopped Early':counter>self.patience})
    
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
        filename = f'{name}_best_normtarget_NEW.pth'
        filepath = os.path.join(modeldir,filename)
        torch.save(self.model.state_dict(),filepath)
        logger.info(f'   Model saved to {filepath}')

def load(inputvars, filename, filedir=FILEDIR):
    def _stack_with_masks(f, split, inputvars):
        blocks = []
        for inputvar in inputvars:
            X = f[f'{inputvar}_{split}'][:]
            if inputvar in ['t','q','capeprofile','subsatprofile']:
                mask = np.isfinite(X).astype(np.float32)
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                X = X * mask
                blocks.append(np.concatenate([X, mask], axis=1))
            else:
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                blocks.append(X)
        return torch.tensor(np.concatenate(blocks, axis=1), dtype=torch.float32)

    filepath = os.path.join(filedir, filename)
    with h5py.File(filepath, 'r') as f:
        Xtrain = _stack_with_masks(f, 'train', inputvars)
        Xvalid = _stack_with_masks(f, 'valid', inputvars)
        Xtest  = _stack_with_masks(f, 'test',  inputvars)

        # keep as numpy until normalization is done
        ytrain = f['pr_train'][:].astype(np.float32).squeeze()
        yvalid = f['pr_valid'][:].astype(np.float32).squeeze()
        ytest  = f['pr_test'][:].astype(np.float32).squeeze()

    # define normparams regardless
    normparams = None
    if NORMTARGET:
        if LOG1PTARGET:
            ytrain = np.log1p(ytrain)
            yvalid = np.log1p(yvalid)
            ytest  = np.log1p(ytest)
        ymean = ytrain.mean()
        ystd  = ytrain.std() + 1e-8
        ytrain = (ytrain - ymean) / ystd
        yvalid = (yvalid - ymean) / ystd
        ytest  = (ytest  - ymean) / ystd
        normparams = {'mean': ymean, 'std': ystd, 'log1p': LOG1PTARGET}

    # convert to tensors once
    ytrain = torch.tensor(ytrain, dtype=torch.float32)
    yvalid = torch.tensor(yvalid, dtype=torch.float32)
    ytest  = torch.tensor(ytest,  dtype=torch.float32)

    return Xtrain, Xvalid, Xtest, ytrain, yvalid, ytest, normparams

def invert_normalization(y_norm, normparams):
    if normparams is None:
        return y_norm
    y = y_norm * normparams['std'] + normparams['mean']
    if normparams.get('log1p', False):
        y = np.expm1(y)
    return y
        
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
        logger.info(f'   Running {description}')
        Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest,normparams = load(inputvars,filename)
        model = NNMODEL(Xtrain.shape[1])
        wandb.init(
            project='NNs',
            name=description,
            config={
                'Input Variables':inputvars,
                'Model Parameter Count':sum(p.numel() for p in model.model.parameters()),
                'Training Batch Size':model.batchsize,
                'Maximum Training Epochs':model.epochs,
                'Network Architecture':'256→128→64→32→1 (GELU + BatchNorm)',
                'Optimizer':'Adam',
                'Loss Function':'Mean Squared Error',
                'Initial Learning Rate':model.learningrate,
                'Learning Rate Scheduler':'ReduceLROnPlateau (factor=0.5)',
                'Early Stopping Patience':model.patience})
        model.fit(Xtrain,Xvalid,ytrain,yvalid)
        model.save(name)
        results[name] = {
            'description':description,
            'n_params':sum(p.numel() for p in model.model.parameters()),
            'y_pred':invert_normalization(model.predict(Xtest),normparams)}
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
        with open(f'{SAVEDIR}/nn_normtarget_NEW_subset_results.pkl', 'wb') as f:
            pickle.dump(results,f)
        del results
        logger.info('Script execution completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')