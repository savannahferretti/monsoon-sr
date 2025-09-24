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

RUNNAME      = 'debug_xy'
INPUTVARS    = ['x']         
TARGETVAR    = 'y'
TRAINSPLIT   = 'debug_train'  
VALIDSPLIT   = 'debug_valid'  
EPOCHS       = 5
BATCHSIZE    = 64000
LEARNINGRATE = 0.001
PATIENCE     = 3
CRITERION    = torch.nn.MSELoss()
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

if DEVICE=='cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    
def reshape(da):
    if 'lev' in da.dims:
        arr = da.transpose('time','lat','lon','lev').values.reshape(-1,da.lev.size)
    else:
        arr = da.transpose('time','lat','lon').values.reshape(-1,1)
    return arr

def load(splitname,inputvars,targetvar=TARGETVAR,filedir=FILEDIR):
    if splitname not in ('debug_train','debug_valid'):
        raise ValueError("Splitname must be 'debug_train' or 'debug_valid'.")
    filename = f'{splitname}.h5'
    filepath = os.path.join(filedir,filename)
    varlist  = list(inputvars)+[targetvar]
    ds    = xr.open_dataset(filepath,engine='h5netcdf')[varlist]
    Xlist = [reshape(ds[inputvar]) for inputvar in inputvars]
    X = np.concatenate(Xlist,axis=1) if len(Xlist)>1 else Xlist[0]
    y = reshape(ds[targetvar])
    X = torch.tensor(X,dtype=torch.float32)
    y = torch.tensor(y,dtype=torch.float32)
    return X,y



def fit(model,runname,Xtrain,Xvalid,ytrain,yvalid,batchsize=BATCHSIZE,device=DEVICE,learningrate=LEARNINGRATE,
        patience=PATIENCE,criterion=CRITERION,epochs=EPOCHS):
    traindataset = TensorDataset(Xtrain,ytrain)
    validdataset = TensorDataset(Xvalid,yvalid)
    trainloader  = DataLoader(traindataset,batch_size=batchsize,shuffle=True,num_workers=4,persistent_workers=True,pin_memory=True)
    validloader  = DataLoader(validdataset,batch_size=batchsize,shuffle=False,num_workers=4,persistent_workers=True,pin_memory=True)
    model      = model.to(device)
    optimizer  = torch.optim.Adam(model.parameters(),lr=learningrate)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=patience)
    wandb.init(
        project='Test NNs for Debugging',
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
            loss = criterion(ybatchpred.squeeze(-1),ybatch.squeeze(-1))
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
                loss = criterion(ybatchpred.squeeze(-1),ybatch.squeeze(-1))
                runningloss += loss.item()*Xbatch.size(0)
        validloss = runningloss/len(validloader.dataset)
        scheduler.step(validloss)
        improved = validloss<bestloss
        if improved:
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
        logger.info('Training single NN model on debug XY...')
        Xtrain,ytrain = load('debug_train',INPUTVARS)
        Xvalid,yvalid = load('debug_valid',INPUTVARS)
        model = NNModel(Xtrain.shape[1])
        fit(model,RUNNAME,Xtrain,Xvalid,ytrain,yvalid)
        del model,Xtrain,Xvalid,ytrain,yvalid
        logger.info('Script completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')