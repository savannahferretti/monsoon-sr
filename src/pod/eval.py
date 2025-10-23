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
LANDVAR     = 'lf'

def load(splitname,inputvar=INPUTVAR,landvar=LANDVAR,filedir=FILEDIR):
    if splitname not in ('valid','test'):
        raise ValueError('Splitname must be `valid` or `test`.')
    filename = f'{splitname}.h5'
    filepath = os.path.join(filedir,filename)
    ds = xr.open_dataset(filepath,engine='h5netcdf')[[inputvar,targetvar,landvar]]
    x  = ds[inputvar].load()
    lf = ds[landvar].load()
    return x,lf

def fetch(runname,modeldir=MODELDIR):
    filename = f'pod_{runname}.npz'
    filepath = os.path.join(modeldir,filename)
    with np.load(filepath) as data:
        mode  = str(data['mode'][0]) if 'mode' in data.files else 'pooled'
        model = PODModel(mode=mode,
                         landthresh=float(data['landthresh']) if 'landthresh' in data.files else 0.5,
                         binmin=float(data['binmin']),
                         binmax=float(data['binmax']),
                         binwidth=float(data['binwidth']),
                         samplethresh=int(data['samplethresh']))
        model.binedges   = data['binedges'].astype(np.float32)
        model.bincenters = data['bincenters'].astype(np.float32)
        model.nbins      = int(model.bincenters.size)
        if mode=='pooled':
            model.alphapooled  = float(data['alphapooled'])
            model.blcritpooled = float(data['blcritpooled'])
            model.nparams      = int(data['nparams'])
        elif method=='regional':
            mode.landthresh   = float(data['landthresh'])
            model.alphaland   = float(data['alphaland'])
            model.blcritland  = float(data['blcritland'])
            model.alphaocean  = float(data['alphaocean'])
            model.blcritocean = float(data['blcritocean'])
            model.nparams     = int(data['nparams'])
    return model

def predict(model, X, landfrac=None):
    ypredflat = model.forward(X, landfrac=landfrac if model.mode == 'regional' else None)
    da = xr.DataArray(ypredflat.reshape(X.shape), dims=X.dims, coords=X.coords, name='predpr')
    da.attrs = dict(long_name='POD ramp-predicted precipitation', units='mm/day')
    return da

def save(ypred, runname, splitname, resultsdir=RESULTSDIR):
    os.makedirs(resultsdir, exist_ok=True)
    filename = f'pod_{runname}_{splitname}_pr_rainy.nc'
    filepath = os.path.join(resultsdir, filename)
    logger.info(f'   Attempting to save {filename}...')
    try:
        ypred.to_netcdf(filepath, engine='h5netcdf')
        with xr.open_dataset(filepath, engine='h5netcdf') as _:
            pass
        logger.info('      File write successful')
        return True
    except Exception:
        logger.exception('      Failed to save or verify')
        return False

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate POD ramp models on a chosen split.')
    parser.add_argument('--split',required=True,choices=['valid','test'],help='Which split to evaluate: `valid` or `test`.')
    args = parser.parse_args()
    try:
        logger.info(f'Loading {args.split} data split...')
        xeval,yeval,lfeval = load(args.split)
        logger.info('Evaluating POD models...')
        for config in RUNCONFIGS:
            runname     = config['run_name']
            description = config['description']
            logger.info(f'   Evaluating {description}')
            model = fetch(runname)
            ypred = predict(model,xeval,landfrac=lf)
            save(ypred,runname,args.split)
            del model,ypred
        logger.info('Script execution completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')