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
RUNCONFIGS  = CONFIGS['runs']
TARGETVAR   = 'pr'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCHSIZE   = 33120

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
    - splitname (str): 'norm_valid' | 'norm_test'
    - inputvars (list[str]): list of input variables
    - targetvar (str): target variable name (defaults to TARGETVAR)
    - filedir (str): directory containing split files (defaults to FILEDIR)
    Returns:
    - tuple[torch.FloatTensor,torch.FloatTensor]: 2D input tensor and target DataArray (for reshaping predictions)
    '''
    if splitname not in ('norm_valid','norm_test'):
        raise ValueError("Split must be 'norm_valid' or 'norm_test'.")
    filename = f'{splitname}.h5'
    filepath = os.path.join(filedir,filename)
    varlist  = list(inputvars)+[targetvar]
    ds = xr.open_dataset(filepath,engine='h5netcdf')[varlist]
    ################
    ds = ds.sel(time=slice('2015-06-01','2015-08-31'))
    ###############
    Xlist = [reshape(ds[inputvar]) for inputvar in inputvars]
    X = np.concatenate(Xlist,axis=1) if len(Xlist)>1 else Xlist[0]
    X = torch.tensor(X,dtype=torch.float32)
    ytemplate = ds[targetvar]
    return X,ytemplate

def fetch(runname,inputsize,device=DEVICE,modeldir=MODELDIR):
    '''
    Purpose: Rebuild a trained NN model.
    Args:
    - runname (str): model run name
    - inputsize (int): number of input features to initialize NNModel
    - device (str): 'cuda' or 'cpu' device for model evaluation (defaults to DEVICE)
    - modeldir (str): directory with saved models (defaults to MODELDIR)
    Returns:
    - NNModel: model on 'device' with loaded state_dict (weights)
    '''
    filename = f'nn_{runname}.pth'
    filepath = os.path.join(modeldir,filename)
    model = NNModel(inputsize).to(device)
    state = torch.load(filepath,map_location=device)
    model.load_state_dict(state)
    return model

def denormalize(ynormflat,targetvar=TARGETVAR,filedir=FILEDIR):
    '''
    Purpose: Convert normalized precipitation predictions back to physical units by undoing z-score normalization and log1p transformation. 
    Args:
    - ynormflat (np.ndarray): vector of normalized predictions
    - targetvar (str): target variable name (defaults to TARGETVAR)
    - filedir (str): directory where JSON file is stored (defaults to FILEDIR)
    Returns:
    - np.ndarray: vector of denormalized predictions in original units
    '''
    with open(os.path.join(filedir,'stats.json'),'r',encoding='utf-8') as f:
        stats = json.load(f)
    mean = float(stats[f'{targetvar}_mean'])
    std  = float(stats[f'{targetvar}_std'])
    ylog = ynormflat*std+mean
    y = np.expm1(ylog)
    return y
    
def predict(model,X,ytemplate,batchsize=BATCHSIZE,device=DEVICE):
    '''
    Purpose: Run the NN forward pass in batches and return precipitation predictions as an xr.DataArray.
    Args:
    - model (NNModel): trained/loaded NN model
    - X (torch.Tensor): 2D input tensor
    - ytemplate (xr.DataArray): template with dimension/coordinates to reshape predictions
    - batchsize (int): inference batch size (defaults to BATCHSIZE)
    - device (str): 'cuda' or 'cpu' device for model evaluation (defaults to DEVICE)
    Returns:
    - xr.DataArray: 3D DataArray of predicted precipitation 
    '''
    evaldataset = TensorDataset(X)
    evalloader  = DataLoader(evaldataset,batch_size=batchsize,shuffle=False,pin_memory=True)
    ypredlist = []
    model.eval()
    with torch.no_grad():
        for (Xbatch,) in evalloader:
            Xbatch = Xbatch.to(device)
            ybatchpred = model(Xbatch)
            ypredlist.append(ybatchpred.squeeze().cpu().numpy())
    ynormflat = np.concatenate(ypredlist,axis=0)
    ypredflat = denormalize(ynormflat)
    da = xr.DataArray(ypredflat.reshape(ytemplate.shape),dims=ytemplate.dims,coords=ytemplate.coords,name='predpr')
    da.attrs = dict(long_name='NN-predicted precipitation',units='mm/day')
    return da

def save(ypred,runname,splitname,resultsdir=RESULTSDIR):
    '''
    Purpose: Save an xr.DataArray of predicted precipitation to a NetCDF file, then verify the write by reopening.
    Args:
    - ypred (xr.DataArray): 3D DataArray of predicted precipitation
    - runname (str): model run name
    - splitname (str): evaluated split label
    - resultsdir (str): output directory (defaults to RESULTSDIR)
    Returns:
    - bool: True if writing and verification succeed, otherwise False
    '''
    os.makedirs(resultsdir,exist_ok=True)
    filename = f'nn_{runname}_{splitname}_pr.nc'
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
    parser.add_argument('--split',required=True,choices=['norm_valid','norm_test'],help="Which split to evaluate: 'norm_valid' or 'norm_test'.")
    args = parser.parse_args()
    try:
        logger.info('Evaluating NN models...')
        for config in RUNCONFIGS:
            runname     = config['run_name']
            inputvars   = config['input_vars']
            description = config['description']
            logger.info(f'   Evaluating {description}')
            X,ytemplate = load(args.split,inputvars)
            model = fetch(runname,X.shape[1])
            ypred = predict(model,X,ytemplate)
            save(ypred,runname,args.split)
            del X,ytemplate,model,ypred
        logger.info('Script completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')