#!/usr/bin/env python

import os
import json
import logging
import warnings
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

def load(inputvar=INPUTVAR,targetvar=TARGETVAR,filedir=FILEDIR):
    '''
    Purpose: Load in training and validation data splits, which we combine for training.
    Args:
    - inputvar (str): input variable name (defaults to INPUTVAR)
    - targetvar (str): target variable name (defaults to TARGETVAR)
    - filedir (str): directory containing split files (defaults to FILEDIR)
    Returns:
    - tuple[xr.DataArray,xr.DataArray]: 3D BL/precipitation DataArrays for training
    '''
    dslist = []
    for splitname in ('train','valid'):
        filename = f'{splitname}.h5'
        filepath = os.path.join(filedir,filename)
        ds = xr.open_dataset(filepath,engine='h5netcdf')[[inputvar,targetvar]]
        dslist.append(ds)
    trainds = xr.concat(dslist,dim='time')
    X,y = trainds[inputvar].load(),trainds[targetvar].load()
    return X,y

def fit(model,Xtrain,ytrain):
    '''
    Purpose: Train a POD model by computing average precipitation in each BL bin.
    Args:
    - model (PODModel): initialized model instance
    - Xtrain (xr.DataArray): 3D BL DataArray for training
    - ytrain (xr.DataArray): 3D precipitation DataArray for training
    Returns:
    - None: the same model instance with 'binmeans' and 'nparams' populated
    '''
    Xflat = Xtrain.values.ravel()
    yflat = ytrain.values.ravel()
    idx  = np.digitize(Xflat,model.binedges)-1
    mask = (idx>=0)&(idx<model.nbins)
    counts = np.bincount(idx[mask],minlength=model.nbins).astype(np.int32)
    sums   = np.bincount(idx[mask],weights=yflat[mask],minlength=model.nbins).astype(np.float32)
    with np.errstate(divide='ignore',invalid='ignore'):
        means = sums/counts
    means[counts<model.samplethresh] = np.nan
    model.binmeans = means.astype(np.float32)
    model.nparams  = int(np.isfinite(model.binmeans).sum())
    return None

def save(model,runname,modeldir=MODELDIR):
    '''
    Purpose: Save trained model parameters and configuration to a .npz file in the specified directory, then verify the write by reopening.
    Args:
    - model (PODModel): trained model instance
    - runname (str): model run name
    - modeldir (str): output directory (defaults to MODELDIR)
    Returns:
    - bool: True if write and verification succeed, otherwise False
    '''
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

if __name__=='__main__':
    try:
        logger.info('Loading training + validation data splits combined...')
        Xtrain,ytrain = load()
        logger.info('Training and saving POD models...')
        for config in RUNCONFIGS:
            runname     = config['run_name']
            binwidth    = config['bin_width']
            description = config['description']
            logger.info(f'   Training {description}')
            model = PODModel(binwidth)
            fit(model,Xtrain,ytrain)
            save(model,runname)
            del model
        logger.info('Script execution completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')