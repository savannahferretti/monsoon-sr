#!/usr/bin/env python

import os
import json
import time
import torch
import wandb
import logging
import warnings
import numpy as np
import xarray as xr
from model import NNModel
from torch.utils.data import TensorDataset,DataLoader

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

with open('configs.json','r',encoding='utf8') as f:
    CONFIGS  = json.load(f)
FILEDIR      = CONFIGS['paths']['filedir']
MODELDIR     = CONFIGS['paths']['modeldir']
RESULTSDIR   = CONFIGS['paths']['resultsdir']
RUNCONFIGS   = CONFIGS['runs']
TARGETVAR    = 'pr'
EPOCHS       = 20
BATCHSIZE    = 66240
LEARNINGRATE = 0.005
PATIENCE     = 3
CRITERION    = torch.nn.MSELoss()
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

if DEVICE=='cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

###############################################
def evaluate(model,dataloader,device):
    model.eval()
    sumsqerr  = 0.0         
    nelements = 0              
    crit = torch.nn.MSELoss(reduction='sum')
    with torch.no_grad():
        for Xbatch,ybatch in dataloader:
            Xbatch,ybatch = Xbatch.to(device,non_blocking=True),ybatch.to(device,non_blocking=True)
            ybatchpred = model(Xbatch).squeeze()
            sumsqerr  += crit(ybatchpred.squeeze(),ybatch.squeeze()).item()
            nelements += ybatch.numel()
    return sumsqerr/nelements if nelements>0 else float('nan')
###############################################
    
def reshape(da):
    '''
    Purpose: Convert an xr.DataArray into a 2D NumPy array suitable for passing through the NN.
    Args:
    - da (xr.DataArray): 3D or 4D DataArray
    Returns:
    - np.ndarray: 2D array of shape (nsamples, nfeatures), where nfeatures equals 1 for 3D or equals 'nlev' for 4D
    '''
    if 'lev' in da.dims:
        arr = da.transpose('time','lat','lon','lev').values.reshape(-1,da.lev.size)
    else:
        arr = da.transpose('time','lat','lon').values.reshape(-1,1)
    return arr

def load(splitname,inputvars,targetvar=TARGETVAR,filedir=FILEDIR):
    '''
    Purpose: Load in a normalized training or validation split and build a 2D feature matrix for the NN. 
    Args:
    - splitname (str): 'norm_train' | 'norm_valid'
    - inputvars (list[str]): list of input variables
    - targetvar (str): target variable name (defaults to TARGETVAR)
    - filedir (str): directory containing split files (defaults to FILEDIR)
    Returns:
    - tuple[torch.FloatTensor,torch.FloatTensor]: 2D input/target tensors
    '''
    if splitname not in ('norm_train','norm_valid'):
        raise ValueError("Splitname must be 'norm_train' or 'norm_valid'.")
    filename = f'{splitname}.h5'
    filepath = os.path.join(filedir,filename)
    varlist  = list(inputvars)+[targetvar]
    ds    = xr.open_dataset(filepath,engine='h5netcdf')[varlist]
    ##########################
    if splitname=='norm_train':
        ds = ds.sel(time=slice('2011-06-01','2014-08-31'))
    elif splitname=='norm_valid':
        ds = ds.sel(time=slice('2015-06-01','2015-08-31'))
    ##########################
    Xlist = [reshape(ds[inputvar]) for inputvar in inputvars]
    X = np.concatenate(Xlist,axis=1) if len(Xlist)>1 else Xlist[0]
    y = reshape(ds[targetvar])
    X = torch.tensor(X,dtype=torch.float32)
    y = torch.tensor(y,dtype=torch.float32)
    return X,y

def fit(model,runname,Xtrain,Xvalid,ytrain,yvalid,batchsize=BATCHSIZE,device=DEVICE,learningrate=LEARNINGRATE,
        patience=PATIENCE,criterion=CRITERION,epochs=EPOCHS):
    '''
    Purpose: Train a NN model with early stopping and learning rate scheduling.
    Args:
    - model (NNModel): initialized model instance
    - runname (str): model run name
    - Xtrain (torch.Tensor): training input(s)
    - Xvalid (torch.Tensor): validation input(s)
    - ytrain (torch.Tensor): training target
    - yvalid (torch.Tensor): validation target
    - batchsize (int): number of samples per training batch (defaults to BATCHSIZE)
    - device (str): 'cuda' or 'cpu' device for model training (defaults to DEVICE)
    - learningrate (float): initial learning rate for the Adam optimizer (defaults to LEARNINGRATE)
    - patience (int): number of epochs to wait without validation loss improvement before early stopping (defaults to PATIENCE)
    - criterion (callable): loss function used for optimization (defaults to CRITERION)
    - epochs (int): maximum number of training epochs (defaults to EPOCHS)
    Returns:
    - None: trains in-place and saves the best model checkpoint
    '''
    traindataset = TensorDataset(Xtrain,ytrain)
    validdataset = TensorDataset(Xvalid,yvalid)
    trainloader  = DataLoader(traindataset,batch_size=batchsize,shuffle=True,num_workers=8,persistent_workers=True,pin_memory=True,prefetch_factor=4)
    validloader  = DataLoader(validdataset,batch_size=batchsize,shuffle=False,num_workers=8,persistent_workers=True,pin_memory=True,prefetch_factor=4)
    model      = model.to(device)
    optimizer  = torch.optim.Adam(model.parameters(),lr=learningrate)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=patience)
    wandb.init(
        project='Precipitation NNs',
        name=runname,
        config={
            'Epochs':epochs,
            'Batch size':batchsize,
            'Initial learning rate':learningrate,
            'Early stopping patience':patience})
    bestloss  = float('inf')
    bestepoch = 0
    noimprove = 0
    starttime = time.time()
    for epoch in range(1,epochs+1):
        model.train()
        runningloss = 0.0
        for Xbatch,ybatch in trainloader:
            Xbatch,ybatch = Xbatch.to(device,non_blocking=True),ybatch.to(device,non_blocking=True)
            optimizer.zero_grad()
            ybatchpred = model(Xbatch)
            loss = criterion(ybatchpred.squeeze(),ybatch.squeeze())
            loss.backward()
            optimizer.step()
            runningloss += loss.item()*Xbatch.size(0)
        trainloss = runningloss/len(trainloader.dataset)
        model.eval()
        runningloss = 0.0
        with torch.no_grad():
            for Xbatch,ybatch in validloader:
                Xbatch,ybatch = Xbatch.to(device,non_blocking=True),ybatch.to(device,non_blocking=True)
                ybatchpred = model(Xbatch)
                loss = criterion(ybatchpred.squeeze(),ybatch.squeeze())
                runningloss += loss.item()*Xbatch.size(0)
        validloss = runningloss/len(validloader.dataset)
        trainevalmse = evaluate(model,trainloader,device)
        validevalmse = evaluate(model,validloader,device)
        scheduler.step(validloss)
        if validloss<bestloss:
            bestloss  = validloss
            bestepoch = epoch
            noimprove = 0
            save(model.state_dict(),runname)
        else:
            noimprove +=1
        wandb.log({
            'Epoch':epoch,
            'Training loss':trainloss,
            'Validation loss':validloss,
            'MSE on training set after epoch finishes':trainevalmse,
            'MSE on validation set after epoch finishes':validevalmse,
            'Learning rate':optimizer.param_groups[0]['lr']})
        if noimprove>=patience:
            break
    duration = time.time()-starttime
    wandb.run.summary.update({
        'Best model at epoch':bestepoch,
        'Best validation loss':bestloss,
        'Total training epochs':epoch,
        'Training duration (s)':duration,
        'Stopped early':noimprove>=patience})
    wandb.finish()

def save(modelstate,runname,modeldir=MODELDIR):
    '''
    Purpose: Save trained model parameters for the best model (lowest validation loss) to a PyTorch checkpoint file in the specified 
    directory, then verify the write by reopening.
    Args:
    - modelstate (dict): model.state_dict() to save
    - runname (str): model run name
    - modeldir (str): output directory (defaults to MODELDIR)
    Returns:
    - bool: True if write and verification succeed, otherwise False
    '''
    os.makedirs(modeldir,exist_ok=True)
    filename = f'nn_{runname}.pth'
    filepath = os.path.join(modeldir,filename)
    logger.info(f'   Attempting to save {filename}...')
    try:
        torch.save(modelstate,filepath)
        _ = torch.load(filepath,map_location='cpu')
        logger.info('      File write successful')
        return True
    except Exception:
        logger.exception('      Failed to save or verify')
        return False

if __name__=='__main__':
    try:
        logger.info('Training and saving NN models...')
        for config in RUNCONFIGS:
            runname     = config['run_name']
            inputvars   = config['input_vars']
            description = config['description']
            logger.info(f'   Training {description}')
            Xtrain,ytrain = load('norm_train',inputvars)
            Xvalid,yvalid = load('norm_valid',inputvars)
            model = NNModel(Xtrain.shape[1])
            fit(model,runname,Xtrain,Xvalid,ytrain,yvalid)
            del model,Xtrain,Xvalid,ytrain,yvalid
        logger.info('Script completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')