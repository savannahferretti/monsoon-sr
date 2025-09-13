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

def load_data(splitname,filedir=FILEDIR):
    '''
    Purpose: Load in the chosen data split for evaluation.
    Args:
    - splitname (str): 'valid' | 'test'
    - filedir (str): directory containing split files (defaults to FILEDIR)
    Returns:
    - tuple[xr.DataArray,xr.DataArray]:  3D BL/precipitation DataArrays for evaluation
    '''
    if splitname not in ('valid','test'):
        raise ValueError("Split must be 'valid' or 'test'.")
    filename = f'{splitname}.h5'
    filepath = os.path.join(filedir,filename)
    evalds   = xr.open_dataset(filepath,engine='h5netcdf')[['bl','pr']]
    X,y = evalds.bl.load(),evalds.pr.load()
    return X,y

def load_model(runname,modeldir=MODELDIR):
    '''
    Purpose: Load a trained POD model from NPZ and reconstruct a PODModel instance.
    Args:
    - runname (str): model run name used in filename
    - modeldir (str): directory with saved models (defaults to MODELDIR)
    Returns:
    - PODModel: model with loaded parameters
    '''
    filename = f'pod_{runname}.npz'
    filepath = os.path.join(modeldir,filename)
    data = np.load(filepath)
    binwidth     = float(data['binwidth'])
    binmin       = float(data['binmin'])
    binmax       = float(data['binmax'])
    samplethresh = int(data['samplethresh'])
    model        = PODModel(binwidth,binmin=binmin,binmax=binmax,samplethresh=samplethresh)
    model.binedges   = data['binedges'].astype(np.float32)
    model.bincenters = data['bincenters'].astype(np.float32)
    model.binmeans   = data['binmeans'].astype(np.float32)
    model.nparams    = int(data['nparams'])
    return model

def save(ypred,runname,splitname,resultsdir=RESULTSDIR):
    '''
    Purpose: Save an xr.Dataset of predicted precipitation to a NetCDF file, then verify the write by reopening.
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
    parser = argparse.ArgumentParser(description='Evaluate POD models on a chosen split.')
    parser.add_argument('--split',required=True,choices=['valid','test'],help="Which split to evaluate: 'valid' or 'test'.")
    args = parser.parse_args()
    try:
        logger.info(f'Loading {args.split} data split...')
        X,y = load_data(args.split)
        logger.info('Evaluating POD models...')
        for config in RUNCONFIGS:
            runname     = config['run_name']
            description = config['description']
            logger.info(f'   Evaluating {description}')
            model = load_model(runname)
            ypred = model.predict(X)
            save(ypred,runname,args.split)
            del model,ypred
        logger.info('Script execution completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')