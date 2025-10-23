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
    CONFIGS  = json.load(f)
FILEDIR      = CONFIGS['paths']['filedir']
MODELDIR     = CONFIGS['paths']['modeldir']
RESULTSDIR   = CONFIGS['paths']['resultsdir']
RUNCONFIGS   = CONFIGS['runs']
INPUTVAR     = 'bl'
TARGETVAR    = 'pr'
LANDVAR      = 'lf'
BINMIN       = -0.6
BINMAX       = 0.1
BINWIDTH     = 0.001
SAMPLETHRESH = 50
PRMIN        = 1.0
PRMAX        = 5.0

def load(inputvar=INPUTVAR,targetvar=TARGETVAR,landvar=LANDVAR,filedir=FILEDIR):
    dslist = []
    for splitname in ('train','valid'):
        filename = f'{splitname}.h5'
        filepath = os.path.join(filedir,filename)
        ds = xr.open_dataset(filepath,engine='h5netcdf')[[inputvar,targetvar,landvar]]
        dslist.append(ds)
    trainds = xr.concat(dslist,dim='time')
    x  = trainds[inputvar].load()
    y  = trainds[targetvar].load()
    lf = trainds[landvar].load()
    return x,y,lf

def fit(model,xtrain,ytrain,mask=None,binmin=BINMIN,binmax=BINMAX,binwidth=BINWIDTH,samplethresh=SAMPLETHRESH,prmin=PRMIN,prmax=PRMAX):  
    xflat = xtrain.values.ravel()
    yflat = ytrain.values.ravel()
    valid = np.isfinite(xflat)&np.isfinite(yflat)
    if mask is not None:
        valid &= mask.ravel()
    xflat = xflat[valid]
    yflat = yflat[valid]
    xedges = np.arange(binmin,binmax+binwidth,binwidth,dtype=np.float64)
    xidxs  = np.digitize(xflat,xedges)-1
    nxbins = xedges.size-1
    inrange = (xidxs>=0)&(xidxs<nxbins)
    ycounts  = np.bincount(xidxs[inrange],minlength=nxbins).astype(np.int64)
    ysums    = np.bincount(xidxs[inrange],weights=yflat[inrange],minlength=nxbins).astype(np.float64)
    with np.errstate(divide='ignore',invalid='ignore'):
        ymeans = ysums/ycounts
    ymeans[ycounts<samplethresh] = np.nan
    xbins = (xedges[:-1]+xedges[1:])*0.5
    keep  = np.isfinite(ymeans)&(ymeans>=prmin)&(ymeans<=prmax)
    slope,intercept = np.polyfit(xbins[keep],ymeans[keep],deg=1)
    alpha  = float(slope)
    blcrit = float(-intercept/slope)
    return alpha,blcrit

def save(model,runname,modeldir=MODELDIR):
    os.makedirs(modeldir,exist_ok=True)
    filename = f'pod_{runname}.npz'
    filepath = os.path.join(modeldir,filename)
    logger.info(f'   Attempting to save {filename}...')
    try:
        if model.mode=='pooled':
            np.savez(
                filepath,
                mode=np.array(['pooled']),
                alphapooled=np.float32(model.alphapooled),
                blcritpooled=np.float32(model.blcritpooled),
                nparams=np.int32(model.nparams))
        elif model.mode=='regional':
            np.savez(
                filepath,
                mode=np.array(['regional']),
                alphaland=np.float32(model.alphaland),
                blcritland=np.float32(model.blcritland),
                alphaocean=np.float32(model.alphaocean),
                blcritocean=np.float32(model.blcritocean),
                landthresh=np.float32(model.landthresh),
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
        xtrain,ytrain,lftrain = load()
        logger.info('Training and saving ramp-fit POD models...')
        for config in RUNCONFIGS:
            runname     = config['run_name']
            mode        = config['mode']
            description = config['description']
            logger.info(f'   Training {description}')
            model = PODModel(mode=mode)
            if mode=='pooled':
                pooledparams = fit(model,xtrain,ytrain,mask=None)
                model.alphapooled  = pooledparams[0]
                model.blcritpooled = pooledparams[1]
                model.nparams      = 2
            elif mode=='regional':
                landmask    = (lftrain.values>=model.landthresh)
                oceanmask   = ~landmask
                landparams  = fit(model,xtrain,ytrain,mask=landmask)
                oceanparams = fit(model,xtrain,ytrain,mask=oceanmask)
                model.alphaland   = landparams[0]
                model.blcritland  = landparams[1]
                model.alphaocean  = oceanparams[0]
                model.blcritocean = oceanparams[1]
                model.nparams     = 4
            else:
                raise ValueError('The `mode` in `configs.json` must be `pooled` or `regional`.')
            save(model,runname)
            del model
        logger.info('Script execution completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')