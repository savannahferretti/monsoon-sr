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

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

FILEPATH = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/processed/data.h5'
SAVEDIR  = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/results'
MODELDIR = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/models/nn'
CONFIGS  = [
    {'name':'exp_1','inputvars':['bl'],'description':'Experiment 1'},
    {'name':'exp_2','inputvars':['cape','subsat'],'description':'Experiment 2'},
    {'name':'exp_3','inputvars':['capeprofile'],'description':'Experiment 3'},
    {'name':'exp_4','inputvars':['subsatprofile'],'description':'Experiment 4'},
    {'name':'exp_5','inputvars':['capeprofile','subsatprofile'],'description':'Experiment 5'},
    {'name':'exp_6','inputvars':['t','q'],'description':'Experiment 6'}]



import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self,x):

        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = NN()




class NNMODEL:
    
    def __init__(self,inputsize,batchsize=64000,epochs=30,criterion=torch.nn.MSELoss(),learningrate=1e-4,patience=3):
        '''
        Purpose: Initialize a NN model for precipitation prediction.
        Args:
        - inputsize (int): number of input features
        - batchsize (int): the batch size for training (defaults to 64,000)
        - epochs (int): maximum number of training epochs (defaults to 30)
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
        self.bestmodel    = None
        
    def fit(self,Xtrain,Xvalid,ytrain,yvalid):
        '''
        Purpose: Train a NN model.
        Args:
        - Xtrain (torch.Tensor): training input features
        - Xvalid (torch.Tensor): validation input features
        - ytrain (torch.Tensor): training target precipitation values
        - yvalid (torch.Tensor): validation target precipitation values
        '''
        traindataset = torch.utils.data.TensorDataset(Xtrain,ytrain)
        validdataset = torch.utils.data.TensorDataset(Xvalid,yvalid)
        trainloader = torch.utils.data.DataLoader(traindataset,batch_size=self.batchsize,shuffle=True,num_workers=8,pin_memory=True)
        validloader = torch.utlis.data.DataLoader(validdataset,batch_size=self.batchsize,shuffle=False,num_workers=8,pin_memory=True)
        optimizer   = torch.optim.Adam(self.model.parameters(),lr=self.learningrate)
        scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=self.patience)
        counter     = 0
        starttime   = time.time()
        for epoch in range(self.epochs):
            self.model.train()
            sumbatchloss = 0.0
            samplesseen  = 0
            for Xbatch, ybatch in trainloader:
                Xbatch = Xbatch.to(self.device)
                ybatch = ybatch.to(self.device)
                optimizer.zero_grad()
                meanbatchloss = self.criterion(self.model(Xbatch).squeeze(),ybatch.squeeze())
                meanbatchloss.backward()
                optimizer.step()
                sumbatchloss += float(meanbatchloss)*Xbatch.size(0)
                samplesseen  += Xbatch.size(0)
            trainloss = sumbatchloss/samplesseen
            if not np.isfinite(trainloss):
                logger.warning('   Training loss is non-finite. Stopping...')
                break
            self.model.eval()
            sumbatchloss = 0.0
            samplesseen  = 0
            with torch.no_grad():
                for Xbatch,ybatch in validloader:
                    Xbatch = Xbatch.to(self.device)
                    ybatch = ybatch.to(self.device)
                    meanbatchloss = self.criterion(self.model(Xbatch).squeeze(),ybatch.squeeze())
                    sumbatchloss += float(meanbatchloss)*Xbatch.size(0)
                    samplesseen  += Xbatch.size(0)
            validloss = sumbatchloss/samplesseen
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
                self.bestmodel = self.model.state_dict().copy()
            else:
                counter += 1
                if counter>self.patience:
                    logger.info(f'   Early stopping at epoch {epoch+1}!')
                    break
        if self.bestmodel is not None:
            self.model.load_state_dict(self.bestmodel)
        trainingtime = time.time()-starttime
        wandb.run.summary.update({
            'Best Model at Epoch':self.bestepoch,
            'Best Validation Loss':self.bestloss,
            'Total Training Epochs':epoch+1,
            'Training Duration (s)':trainingtime,
            'Stopped Early':counter>self.patience})

    # def predict(self,X):
    #     '''
    #     Purpose: Generate precipitation predictions using the trained model.
    #     Args:
    #     - X (torch.Tensor): input features for prediction
    #     Returns:
    #     - numpy.ndarray: predicted precipitation values
    #     '''
    #     self.model.eval()
    #     with torch.no_grad():
    #         X     = X.to(self.device)
    #         ypred = self.model(X)
    #         ypred = ypred.squeeze().cpu().numpy()
    #     return ypred

    def save(self,name,modeldir=MODELDIR):
        '''
        Purpose: Save the best model state to a PyTorch model checkpoint file.
        Args:
        - name (str): model name
        - modeldir (str): output directory (defaults to MODELDIR)
        '''        
        filename = f'{name}_best.pth'
        filepath = os.path.join(modeldir,filename)
        torch.save(self.model.state_dict(),filepath)
    




    PROFILE_LIKE = ('t', 'q', 'capeprofile', 'subsatprofile')

def normalize(train,valid,test,datatype,columnwise):
    if datatype=='input':
        if columnwise:
            mean = np.nanmean(train,axis=0)
            std  = np.nanstd(train,axis=0)
        else:
            mean = np.nanmean(train)
            std  = np.nanstd(train)
        trainorm  = (train-mean)/std
        validnorm = (valid-mean)/std
        testnorm  = (test-mean)/std
        return trainnorm,validnorm,testnorm
    if datatype == 'target':
        logtrain = np.log1p(train)
        logvalid = np.log1p(valid)
        logtest  = np.log1p(test)
        mean = np.nanmean(logtrain)
        std  = np.nanstd(logtrain)
        trainorm   = (logtrain-mean)/std
        validnorm  = (logvalid-mean)/std
        testnorm   = (logtest-mean)/std
        normparams = {
            'mean':mean,
            'std':std}
        return trainnorm,validnorm,testnorm,normparams

def denormalize(ynorm,normparams):
    y = ynorm*normparams['std']+normparams['mean']
    y = np.expm1(y)
    return y
    
def _mask_and_values(array):
    mask = np.isfinite(array)
    vals = np.nan_to_num(a,nan=0.0,posinf=0.0,neginf=0.0)
    vals = vals * mask
    return vals, mask


def feature_block(array, include_mask: bool) -> np.ndarray:
    """
    Return a single feature block:
      - values: NaNs/inf -> 0, then zeroed by finite mask
      - if include_mask: append the finite mask as extra columns
    """
    mask   = np.isfinite(array)
    values = np.nan_to_num(array,nan=0.0,posinf=0.0,neginf=0.0).astype(np.float32) * mask
    if includemask:
        return np.concatenate([vals,mask], axis=1)
    else:
        return values
    return np.concatenate([vals,mask], axis=1) if include_mask else vals
                
def load(inputvars,filepath,columnwise):

    with h5py.File(filepath,'r') as f:
        Xtrainblocks = []
        Xvalidblocks = []
        Xtestblocks  = []
        for inputvar in inputvars:
            Xtrainarray = f[f'{inputvar}_train'][:]
            Xvalidarray = f[f'{inputvar}_valid'][:]
            Xtestarray  = f[f'{inputvar}_test'] [:]
            Xtrainnorm,Xvalidnorm,Xtestnorm = normalize(Xtrainarray,Xvalidarray,Xtestarray,datatype='input',columnwise=True)
            if inputvar in ('t','q','capeprofile','subsatprofile'):
                Xtrainvals,Xtrainmask = mask_and_values(Xtrainnorm)
                Xvalidvals,Xvalidmask = mask_and_values(Xvalidnorm)
                Xtestvals,Xtestmask   = mask_and_values(Xtestnorm)
                Xtrainblocks.append(np.concatenate([Xtrainvals,trainmask],axis=1))
                Xvalidblocks.append(np.concatenate([Xvalidvals,validmask],axis=1))
                Xtestblocks.append(np.concatenate([Xtestvals,testmask],axis=1))
            else:
                Xtrainblocks.append(Xtrainvals)
                Xvalidblocks.append(Xvalidvals)
                Xtestblocks.append(Xtestvals)
        Xtrain = np.concatenate(Xtrainblocks,axis=1).astype(np.float32)
        Xvalid = np.concatenate(Xvalidblocks,axis=1).astype(np.float32)
        Xtest  = np.concatenate(Xtestblocks,axis=1).astype(np.float32)
        
        ytrainarray = f['pr_train'][:]
        yvalidarray = f['pr_valid'][:]
        ytestarray  = f['pr_test'] [:]
    ytrainnorm,yvalidnorm,ytestnorm,normparams = normalize(ytrainarray,yvalidarray,ytestarray,datatype='target',columnwise=False)
    
    Xtrain = torch.tensor(Xtrain)
    Xvalid = torch.tensor(Xvalid)
    Xtest  = torch.tensor(Xtest)
    ytrain = torch.tensor(ytrainnorm)
    yvalid = torch.tensor(yvalidnorm)
    ytest  = torch.tensor(ytestnorm)
    
    return Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest,normparams
    
# def process(configs=CONFIGS):
#     '''
#     Purpose: Train and evaluate NN models with multiple input variable configurations.
#     Args:
#     - configs (list[dict]): model configurations specifying input variables and descriptions (defaults to CONFIGS)
#     Returns:
#     - list[dict]: dictionary containing NN model results
#     '''
#     results = {}
#     for config in configs:
#         name        = config['name']
#         inputvars   = config['inputvars']
#         description = config['description']
#         logger.info(f'   Running {description}')
#         Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest,normparams = load(inputvars,filename)
#         model = NNMODEL(Xtrain.shape[1])
#         wandb.init(
#             project='Precipitation NNs',
#             name=description,
#             config={
#                 'Input Variables':inputvars,
#                 'Model Parameter Count':sum(p.numel() for p in model.model.parameters()),
#                 'Training Batch Size':model.batchsize,
#                 'Maximum Training Epochs':model.epochs,
#                 'Network Architecture':'256→128→64→32→1 (GELU + BatchNorm)',
#                 'Optimizer':'Adam',
#                 'Loss Function':'Mean Squared Error',
#                 'Initial Learning Rate':model.learningrate,
#                 'Learning Rate Scheduler':'ReduceLROnPlateau (factor=0.5)',
#                 'Early Stopping Patience':model.patience})
#         model.fit(Xtrain,Xvalid,ytrain,yvalid)
#         model.save(name)
#         results[name] = {
#             'description':description,
#             'n_params':sum(p.numel() for p in model.model.parameters()),
#             'y_pred':denormalize(model.predict(Xtest),normparams)}
#         wandb.finish()
#         del model,Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest,normparams
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#     return results

# def save(results,filename='nn_results.pkl',savedir=SAVEDIR):
#     '''
#     Purpose: Save NN model results to a pickle file in the specified directory, then verify the write by reopening.
#     Args:
#     - results (dict): NN model results to save
#     - filename (str): output file name (defaults to 'nn_results.pkl')
#     - savedir (str): output directory (defaults to SAVEDIR)
#     Returns:
#     - bool: True if write and verification succeed, otherwise False
#     '''
#     try:
#         filepath = os.path.join(savedir,filename)
#         logger.info(f'Attempting to save results to {filepath}...')
#         with open(filepath,'wb') as f:
#             pickle.dump(results,f,protocol=pickle.HIGHEST_PROTOCOL)
#         with open(filepath,'rb') as f:
#             _ = pickle.load(f)
#         logger.info('   File write successful')
#         return True
#     except Exception:
#         logger.exception('   Failed to save or verify')
#         return False

# if __name__=='__main__':
#     try:
#         logger.info('Training NN models...')
#         results = process()
#         logger.info('Saving results...')
#         save(results)
#         del results
#         logger.info('Script execution completed successfully!')
#     except Exception as e:
#         logger.error(f'An unexpected error occurred: {str(e)}')