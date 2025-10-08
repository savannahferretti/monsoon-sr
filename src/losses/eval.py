#!/usr/bin/env python

import os
import json
import time
import torch
import logging
import warnings
import argparse
import numpy as np
import xarray as xr
from models import PODModel,NNModel,TweedieDevianceLoss
from torch.utils.data import TensorDataset,DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

with open('configs.json','r',encoding='utf8') as f:
    CONFIGS = json.load(f)
FILEDIR     = CONFIGS['paths']['filedir']
MODELDIR    = CONFIGS['paths']['modeldir']
RESULTSDIR  = CONFIGS['paths']['resultsdir']
RUNCONFIGS  = CONFIGS['runs']

PRTHRESH  = 0.01
BATCHSIZE = 72864
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

if DEVICE=='cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

def load_pod(splitname,filtered,inputvar='bl',targetvar='pr',prthresh=PRTHRESH,filedir=FILEDIR):
    filename = f'{splitname}.h5'
    filepath = os.path.join(filedir,filename)
    evalds = xr.open_dataset(filepath,engine='h5netcdf')[[inputvar,targetvar]]
    X = evalds[inputvar]
    y = evalds[targetvar]
    if filtered:
        mask = y>prthresh
        X = X.where(mask)
        y = y.where(mask)
    return X,y,y

def fetch_pod(runname,modeldir=MODELDIR):
    filename = f'pod_{runname}.npz'
    filepath = os.path.join(modeldir,filename)
    with np.load(filepath) as data:
        model = PODModel(float(data['binwidth']),
                         binmin=float(data['binmin']),
                         binmax=float(data['binmax']),
                         samplethresh=int(data['samplethresh']))
        model.binedges   = data['binedges'].astype(np.float32)
        model.bincenters = data['bincenters'].astype(np.float32)
        model.binmeans   = data['binmeans'].astype(np.float32)
        model.nparams    = int(data['nparams'])
        model.nbins      = int(model.bincenters.size)
    return model

def predict_pod(model,X,template):
    ypredflat = model.forward(X)
    da = xr.DataArray(ypredflat.reshape(template.shape),dims=template.dims,coords=template.coords,name='pr')
    da.attrs = dict(long_name='POD-predicted precipitation',units='mm/hr')
    return da
        
def load_nn(splitname,filtered,inputvar='bl',targetvar='pr',prthresh=PRTHRESH,filedir=FILEDIR):
    filename = f'{splitname}.h5'
    filepath = os.path.join(filedir,filename)
    evalds   = xr.open_dataset(filepath,engine='h5netcdf')[[inputvar,targetvar]]
    X3D = evalds[inputvar].transpose('time','lat','lon')
    y3D = evalds[targetvar].transpose('time','lat','lon')
    X = X3D.values.reshape(-1,1)
    y = y3D.values.reshape(-1,1)
    mask = np.isfinite(X).all(axis=1)&np.isfinite(y).squeeze(1)
    if filtered:
        mask = mask&(y.squeeze(1)>prthresh)
    maskfull = mask  
    X = X[mask]
    y = y[mask]
    X = torch.tensor(X,dtype=torch.float32)
    y = torch.tensor(y,dtype=torch.float32)
    return X,y,y3D,maskfull

def fetch_nn(runname,inputsize,device=DEVICE,modeldir=MODELDIR):
    filename = f'nn_{runname}.pth'
    filepath = os.path.join(modeldir,filename)
    model = NNModel(inputsize).to(device)
    state = torch.load(filepath,map_location='cpu')
    model.load_state_dict(state)
    return model

def predict_nn(model,X,template,mask,batchsize=BATCHSIZE,device=DEVICE):
    evaldataset = TensorDataset(X)
    evalloader  = DataLoader(evaldataset,batch_size=batchsize,shuffle=False,pin_memory=True)
    ypredlist = []
    model.eval()
    with torch.no_grad():
        for (Xbatch,) in evalloader:
            Xbatch = Xbatch.to(device,non_blocking=True)
            ybatchpred = model(Xbatch)
            ypredlist.append(ybatchpred.squeeze(-1).cpu().numpy())
    ypredflat = np.concatenate(ypredlist,axis=0)
    full = np.full(mask.shape,np.nan,dtype=np.float32)
    full[mask] = ypredflat
    da = xr.DataArray(full.reshape(template.shape),dims=template.dims,coords=template.coords,name='pr')
    da.attrs = dict(long_name='NN-predicted precipitation',units='mm/hr')
    return da

def save(ypred,runtype,runname,splitname,resultsdir=RESULTSDIR):
    os.makedirs(resultsdir,exist_ok=True)
    filename = f'{runtype}_{runname}_{splitname}_pr.nc'
    filepath = os.path.join(resultsdir,filename)
    logger.info(f'Attempting to save {filename}...')
    try:
        ypred.to_netcdf(filepath,engine='h5netcdf')
        with xr.open_dataset(filepath,engine='h5netcdf') as _:
            pass
        logger.info('   File write successful')
        return True
    except Exception:
        logger.exception('   Failed to save or verify')
        return False
        
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate models on a chosen split.')
    parser.add_argument('--split',required=True,choices=['valid','test'],help='Which split to evaluate: "valid" or "test".')
    args = parser.parse_args()
    try:
        logger.info('Evaluating models...')
        for config in RUNCONFIGS:
            runtype     = config['run_type']
            runname     = config['run_name']
            filtered    = config['filtered']
            description = config['description']
            logger.info(f'Running {description}')
            if runtype=='pod':
                X,y,template = load_pod(args.split,filtered)
                model = fetch_pod(runname)
                ypred = predict_pod(model,X,template)
                save(ypred,runtype,runname,args.split)
                del X,y,template,model,ypred
            elif runtype=='nn':
                X,y,template,mask = load_nn(args.split,filtered)
                model = fetch_nn(runname,X.shape[1])
                ypred = predict_nn(model,X,template,mask)
                save(ypred,runtype,runname,args.split)
                del X,y,template,mask,model,ypred
            else:
                logger.error(f'Unknown run type {runtype}. Skipping...')
        logger.info('Script execution completed successfully!')
    except Exception as e:
        logger.exception(f'An unexpected error occurred: {e}')