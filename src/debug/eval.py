#!/usr/bin/env python

import os
import json
import torch
import logging
import warnings
import argparse
import numpy as np
import xarray as xr
from model import NNModel
from torch.utils.data import TensorDataset,DataLoader

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

with open('configs.json','r',encoding='utf8') as f:
    CONFIGS = json.load(f)
FILEDIR     = CONFIGS['paths']['filedir']
MODELDIR    = CONFIGS['paths']['modeldir']
RESULTSDIR  = CONFIGS['paths']['resultsdir']
RUNNAME     = 'debug_xy'
INPUTVARS   = ['x']
TARGETVAR   = 'y'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCHSIZE   = 64000

def reshape(da):
    if 'lev' in da.dims:
        arr = da.transpose('time','lat','lon','lev').values.reshape(-1,da.lev.size)
    else:
        arr = da.transpose('time','lat','lon').values.reshape(-1,1)
    return arr
    
def load(splitname,inputvars,targetvar=TARGETVAR,filedir=FILEDIR):
    if splitname not in ('debug_valid','debug_test'):
        raise ValueError("Split must be 'debug_valid' or 'debug_test'.")
    filename = f'{splitname}.h5'
    filepath = os.path.join(filedir,filename)
    varlist  = list(inputvars)+[targetvar]
    ds = xr.open_dataset(filepath,engine='h5netcdf')[varlist]
    Xlist = [reshape(ds[inputvar]) for inputvar in inputvars]
    X = np.concatenate(Xlist,axis=1) if len(Xlist)>1 else Xlist[0]
    X = torch.tensor(X,dtype=torch.float32)
    ytemplate = ds[targetvar]
    return X,ytemplate

def fetch(runname,inputsize,device=DEVICE,modeldir=MODELDIR):
    filename = f'nn_{runname}.pth'
    filepath = os.path.join(modeldir,filename)
    model = NNModel(inputsize).to(device)
    state = torch.load(filepath,map_location=device)
    model.load_state_dict(state)
    return model

def predict(model,X,ytemplate,batchsize=BATCHSIZE,device=DEVICE):
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
    da = xr.DataArray(ypredflat.reshape(ytemplate.shape),dims=ytemplate.dims,coords=ytemplate.coords,name='ypred')
    da.attrs = dict(long_name='NN-predicted y',units='arb')
    return da

def save(ypred,runname,splitname,resultsdir=RESULTSDIR):
    os.makedirs(resultsdir,exist_ok=True)
    filename = f'nn_{runname}_{splitname}_y.nc'
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
    parser = argparse.ArgumentParser(description='Evaluate NN models on a chosen split.')
    parser.add_argument('--split',required=True,choices=['debug_valid','debug_test'],help="Which split to evaluate: 'debug_valid' or 'debug_test'.")
    args = parser.parse_args() 
    try:
        logger.info(f"Evaluating NN model '{RUNNAME}' on split {args.split} ...")
        X,ytemplate = load(args.split,INPUTVARS)
        model = fetch(RUNNAME,X.shape[1])
        ypred = predict(model,X,ytemplate)
        save(ypred,RUNNAME,args.split)
        del X,ytemplate,model,ypred
        logger.info('Script completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')