#!/usr/bin/env python

import os
import json
import logging
import warnings
import argparse
import numpy as np
import xarray as xr
from model import PODModel

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

with open('configs.json','r',encoding='utf8') as f:
    CONFIGS = json.load(f)
FILEDIR     = CONFIGS['paths']['filedir']
MODELDIR    = CONFIGS['paths']['modeldir']
RESULTSDIR  = CONFIGS['paths']['resultsdir']
RUNCONFIGS  = CONFIGS['runs']
INPUTVAR    = 'bl'
TARGETVAR   = 'pr'

def load(splitname,inputvar=INPUTVAR,targetvar=TARGETVAR,filedir=FILEDIR):
    '''
    Purpose: Load in the chosen data split for evaluation.
    Args:
    - splitname (str): 'valid' | 'test'
    - inputvar (str): input variable name (defaults to INPUTVAR)
    - targetvar (str): target variable name (defaults to TARGETVAR)
    - filedir (str): directory containing split files (defaults to FILEDIR)
    Returns:
    - tuple[xr.DataArray,xr.DataArray]: 3D BL/precipitation DataArrays for evaluation
    '''
    if splitname not in ('valid','test'):
        raise ValueError("Splitname must be 'valid' or 'test'.")
    filename = f'{splitname}.h5'
    filepath = os.path.join(filedir,filename)
    evalds   = xr.open_dataset(filepath,engine='h5netcdf')[[inputvar,targetvar]]
    X,y = evalds[inputvar].load(),evalds[targetvar].load()
    return X,y

def fetch(runname,modeldir=MODELDIR):
    '''
    Purpose: Rebuild the trained POD model.
    Args:
    - runname (str): model run name
    - modeldir (str): directory with saved models (defaults to MODELDIR)
    Returns:
    - PODModel: model with loaded parameters
    '''
    filename = f'pod_{runname}.npz'
    filepath = os.path.join(modeldir,filename)
    with np.load(filepath) as data:
        model  = PODModel(
            float(data['binwidth']),
            binmin=float(data['binmin']),
            binmax=float(data['binmax']),
            samplethresh=int(data['samplethresh']))
        model.binedges   = data['binedges'].astype(np.float32)
        model.bincenters = data['bincenters'].astype(np.float32)
        model.binmeans   = data['binmeans'].astype(np.float32)
        model.nparams    = int(data['nparams'])
    return model

def predict(model,X):
    '''
    Purpose: Run the POD forward pass and return precipitation predictions as an xr.DataArray.
    Args:
    - model (PODModel): trained/loaded POD model
    - X (xr.DataArray): input 3D BL DataArray
    Returns:
    - xr.DataArray: 3D DataArray of predicted precipitation
    '''
    ypredflat = model.forward(X)
    da = xr.DataArray(ypredflat.reshape(X.shape),dims=X.dims,coords=X.coords,name='predpr')
    da.attrs = dict(long_name='POD-predicted precipitation',units='mm/day')
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
    filename = f'pod_{runname}_{splitname}_pr.nc'
    filepath = os.path.join(resultsdir,filename)
    logger.info(f'   Attempting to save {filename}...')
    try:
        ypred.to_netcdf(filepath,engine='h5netcdf')
        with xr.open_dataset(filepath,engine='h5netcdf') as _:
            pass
        logger.info('      File write successful')
        return True
    except Exception:
        logger.exception('      Failed to save or verify')
        return False

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate POD models on a chosen split.')
    parser.add_argument('--split',required=True,choices=['valid','test'],help="Which split to evaluate: 'valid' or 'test'.")
    args = parser.parse_args()
    try:
        logger.info(f'Loading {args.split} data split...')
        X,y = load(args.split)
        logger.info('Evaluating POD models...')
        for config in RUNCONFIGS:
            runname     = config['run_name']
            description = config['description']
            logger.info(f'   Evaluating {description}')
            model = fetch(runname)
            ypred = predict(model,X)
            save(ypred,runname,args.split)
            del model,ypred
        logger.info('Script execution completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')