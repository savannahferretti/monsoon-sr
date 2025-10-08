#!/usr/bin/env python

import os
import json
import time
import torch
import wandb
import random
import logging
import warnings
import numpy as np
import xarray as xr
from models import PODModel,NNModel,TweedieDevianceLoss
from torch.utils.data import TensorDataset,DataLoader

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

with open('configs.json','r',encoding='utf8') as f:
    CONFIGS = json.load(f)
FILEDIR     = CONFIGS['paths']['filedir']
MODELDIR    = CONFIGS['paths']['modeldir']
RUNCONFIGS  = CONFIGS['runs']

PRTHRESH     = 0.01
EPOCHS       = 20
BATCHSIZE    = 298080
LEARNINGRATE = 3e-4
PATIENCE     = 2
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOMSEED   = 42

random.seed(RANDOMSEED)
np.random.seed(RANDOMSEED)
torch.manual_seed(RANDOMSEED)
if DEVICE=='cuda':
    torch.cuda.manual_seed_all(RANDOMSEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = False

def load_pod(filtered,inputvar='bl',targetvar='pr',prthresh=PRTHRESH,filedir=FILEDIR):
    dslist = []
    for splitname in ('train','valid'):
        filename = f'{splitname}.h5'
        filepath = os.path.join(filedir,filename)
        ds = xr.open_dataset(filepath,engine='h5netcdf')[[inputvar,targetvar]]
        dslist.append(ds)
    trainds = xr.concat(dslist,dim='time')
    X = trainds[inputvar]
    y = trainds[targetvar]
    if filtered:
        mask = y>prthresh
        X = X.where(mask)
        y = y.where(mask)
    return X,y

def save_pod(model,runname,modeldir=MODELDIR):
    os.makedirs(modeldir,exist_ok=True)
    filename = f'pod_{runname}.npz'
    filepath = os.path.join(modeldir,filename)
    logger.info(f'   Attempting to save {filename}...')
    try:
        np.savez(
            filepath,
            binedges=model.binedges,
            bincenters=model.bincenters,
            binmeans=model.binmeans,
            samplethresh=np.int32(model.samplethresh),
            binwidth=np.float32(model.binwidth),
            binmin=np.float32(model.binmin),
            binmax=np.float32(model.binmax),
            nparams=np.int32(model.nparams))
        with np.load(filepath) as _:
            pass
        logger.info('      File write successful')
        return True
    except Exception:
        logger.exception('      Failed to save or verify')
        return False
        
def fit_pod(model,runname,Xtrain,ytrain,modeldir=MODELDIR):
    Xflat = Xtrain.values.ravel()
    yflat = ytrain.values.ravel()
    mask  = np.isfinite(Xflat)&np.isfinite(yflat)
    Xflat = Xflat[mask]
    yflat = yflat[mask]
    binidxs = np.digitize(Xflat,model.binedges)-1
    inrange = (binidxs>=0)&(binidxs<model.nbins)
    counts = np.bincount(binidxs[inrange],minlength=model.nbins).astype(np.int64)
    sums   = np.bincount(binidxs[inrange],weights=yflat[inrange],minlength=model.nbins).astype(np.float32)
    with np.errstate(divide='ignore',invalid='ignore'):
        means = sums/counts
    badbins  = (counts<model.samplethresh)|~np.isfinite(means)
    means    = means.astype(np.float32)
    means[badbins]   = np.nan
    model.binmeans   = means
    model.nparams    = int(np.isfinite(means).sum())
    save_pod(model,runname)
    return None

def load_nn(splitname,filtered,inputvar='bl',targetvar='pr',prthresh=PRTHRESH,filedir=FILEDIR):
    if splitname not in ('train','valid'):
        raise ValueError('splitname must be "train" or "valid".')
    filename = f'{splitname}.h5'
    filepath = os.path.join(filedir,filename)
    ds = xr.open_dataset(filepath,engine='h5netcdf')[[inputvar,targetvar]]
    X  = ds[inputvar].transpose('time','lat','lon').values.reshape(-1,1)
    y  = ds[targetvar].transpose('time','lat','lon').values.reshape(-1,1)
    mask = np.isfinite(X).all(axis=1)&np.isfinite(y).squeeze(1)
    if filtered:
        mask = mask&(y.squeeze(1)>prthresh)
    X = X[mask]
    y = y[mask]
    X = torch.tensor(X,dtype=torch.float32)
    y = torch.tensor(y,dtype=torch.float32)
    return X,y

def save_nn(modelstate,runname,modeldir=MODELDIR):
    os.makedirs(modeldir,exist_ok=True)
    filepath = os.path.join(modeldir,f'nn_{runname}.pth')
    logger.info(f'   Attempting to save nn_{runname}.pth...')
    try:
        torch.save(modelstate,filepath)
        _ = torch.load(filepath,map_location='cpu')
        logger.info('      File write successful')
        return True
    except Exception:
        logger.exception('      Failed to save or verify')
        return False

def fit_nn(model,runname,criterion,Xtrain,ytrain,Xvalid,yvalid,batchsize=BATCHSIZE,device=DEVICE,learningrate=LEARNINGRATE,patience=PATIENCE,epochs=EPOCHS,modeldir=MODELDIR):
    traindataset = TensorDataset(Xtrain,ytrain)
    validdataset = TensorDataset(Xvalid,yvalid)    
    def _dataloader_worker_args(device):
        cpucount = os.cpu_count() or 0
        nworkers = min(32,cpucount) if cpucount else 0
        kwargs = {'num_workers':nworkers,'persistent_workers':bool(nworkers),'pin_memory':device=='cuda'}
        if device=='cuda':
            kwargs['pin_memory_device'] = 'cuda'
        if kwargs['num_workers']>0:
            kwargs['prefetch_factor'] = 4
        return kwargs
    kwargs      = _dataloader_worker_args(device)
    trainloader = DataLoader(traindataset,batch_size=batchsize,shuffle=True,**kwargs)
    validloader = DataLoader(validdataset,batch_size=batchsize,shuffle=False,**kwargs)
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=learningrate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=patience)
    wandb.init(
        project='MSE vs. Tweedie NNs',
        name=runname,
        config={
            'Criterion':criterion.__class__.__name__,
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
            save_nn(model.state_dict(),runname)
        else:
            noimprove += 1
        wandb.log({
            'Training loss':trainloss,
            'Validation loss':validloss,
            'Learning rate':optimizer.param_groups[0]['lr']},
                  step=epoch)
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
    return None
    
if __name__=='__main__':
    try:
        logger.info('Training models...')
        for config in RUNCONFIGS:
            runtype     = config['run_type']
            runname     = config['run_name']
            filtered    = config['filtered']
            description = config['description']
            logger.info(f'Running {description}')
            if runtype=='pod':
                binwidth      = config['bin_width']
                Xtrain,ytrain = load_pod(filtered)
                model = PODModel(binwidth)
                fit_pod(model,runname,Xtrain,ytrain)
                del Xtrain,ytrain,model
            elif runtype=='nn':
                criterion     = torch.nn.MSELoss() if 'mse' in runname.lower() else TweedieDevianceLoss()
                Xtrain,ytrain = load_nn('train',filtered)
                Xvalid,yvalid = load_nn('valid',filtered)
                model = NNModel(Xtrain.shape[1])
                fit_nn(model,runname,criterion,Xtrain,ytrain,Xvalid,yvalid)
                del Xtrain,ytrain,Xvalid,yvalid,model
            else:
                logger.error(f'Unknown run type {runtype}. Skipping...')
        logger.info('Script execution completed successfully!')
    except Exception as e:
        logger.exception(f'An unexpected error occurred: {e}')