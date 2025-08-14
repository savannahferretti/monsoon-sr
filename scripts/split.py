#!/usr/bin/env python

import os
import h5py
import json
import logging
import warnings
import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

FILEDIR     = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/interim'
SAVEDIR     = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/processed'
INPUTVARS   = ['bl','cape','subsat','capeprofile','subsatprofile','t','q']
TARGETVAR   = 'pr'
NYEARS      = 2
TESTSIZE    = 0.2
VALIDSIZE   = 0.25
RANDOMSTATE = 42

def get_random_time_idxs(refname,nyears=NYEARS,filedir=FILEDIR):
    filename = f'{refname}.nc'
    filepath = os.path.join(filedir,filename)
    da = xr.open_dataarray(filepath).load()
    totaltimesteps   = len(da.time)
    timestepsperyear = totaltimesteps/len(np.unique(da.time.dt.year.values))
    targettimesteps  = int(nyears*timestepsperyear)
    timeidxs = np.sort(np.random.choice(totaltimesteps,size=targettimesteps,replace=False))
    logger.info(f'   Selected {(len(timeidxs)/totaltimesteps)*100:.2f}% timesteps to use for data split')
    return timeidxs

def load(varnames,timeidxs=None,filedir=FILEDIR):
    dalist = {}
    for varname in varnames:
        logger.info(f'Loading {varname}...')
        filename = f'{varname}.nc'
        filepath = os.path.join(filedir,filename)
        da = xr.open_dataarray(filepath).load()
        if timeidxs is not None:
            da = da.isel(time=timeidxs)
        dalist[varname] = da
    return dalist

def process_input(variable,currentcol):
    varname = variable.name
    if 'lev' not in variable.dims:
        inputdata  = variable.values.astype(np.float32).flatten().reshape(-1,1)
        inputnames = [varname]
        varinfo = {
            'columns':[currentcol],
            'n_inputs':1,
            'input_names':[varname],
            'has_levels':False}
        logger.info(f'  Added {varname} as 1 feature')
        return inputdata,inputnames,varinfo,currentcol+1
    else:
        levels     = variable.lev.values
        nlevels    = len(levels)
        nsamples   = variable.values.flatten().shape[0]//nlevels
        inputdata  = variable.values.astype(np.float32).reshape(nsamples,nlevels)
        inputnames = [f'{varname}_{int(lev)}' for lev in levels]
        info = {
            'columns':list(range(currentcol,currentcol+nlevels)),
            'n_inputs':nlevels,
            'input_names':inputnames,
            'has_levels':True,
            'levels':levels.tolist()}
        logger.info(f'  Added {varname} as {nlevels} features')
        return inputdata,inputnames,info,currentcol+nlevels

def split(dalist,inputvars,targetvar,testsize=TESTSIZE,validsize=VALIDSIZE,randomstate=RANDOMSTATE):
    logger.info('Creating data splits...')
    targetdata = dalist[targetvar].values.astype(np.float32).flatten()
    nsamples   = len(targetdata)
    inputdatalist = []
    inputnamelist = []
    inputmapping  = {}
    currentcol    = 0
    for varname in inputvars:
        logger.info(f'Processing {varname}...')
        inputdata,inputnames,info,currentcol = process_input(dalist[varname],currentcol)
        inputdatalist.append(inputdata)
        inputnamelist.extend(inputnames)
        inputmapping[varname] = info
    combinedinputs = np.concatenate(inputdatalist,axis=1)
    sampleidxs = np.arange(nsamples)
    trainvalidx,testidx = train_test_split(sampleidxs,test_size=testsize,shuffle=True,random_state=randomstate)
    trainidx,valididx   = train_test_split(trainvalidx,test_size=validsize,shuffle=True,random_state=randomstate)
    splits = {
        'inputs_train':combinedinputs[trainidx],
        'inputs_valid':combinedinputs[valididx],
        'inputs_test':combinedinputs[testidx],
        'target_train':targetdata[trainidx],
        'target_valid':targetdata[valididx],
        'target_test':targetdata[testidx]}
    logger.info(f'Split Sizes - Training: {len(trainidx)}, Validation: {len(valididx)}, Testing: {len(testidx)}')
    return splits,inputnamelist,inputmapping

def save_data_splits(datasplits,filename='datasplits.h5',savedir=SAVEDIR):
    filepath = os.path.join(savedir,filename)
    try:
        with h5py.File(filepath,'w') as f:
            for key,array in datasplits.items():
                f.create_dataset(key,data=array,compression='gzip',compression_opts=9)
        logger.info(f'Successfully saved {filename}')
        return True
    except Exception as e:
        logger.error(f'Failed to save {filename}: {e}')
        return False

def save_metadata(inputmapping,inputnames,filename='metadata.json',savedir=SAVEDIR):
    filepath = os.path.join(savedir,filename)
    metadata = {
        'description':'experiment_input_variable_mapping',
        'data_type':'float32',
        'n_available_variables':list(inputmapping.keys()),
        'n_inputs':len(inputnames),
        'input_names':inputnames,
        'input_mapping':inputmapping}
    try:
        with open(filepath,'w') as f:
            json.dump(metadata,f,indent=2)
        logger.info(f'Successfully saved metadata {filename}')
        return True
    except Exception as e:
        logger.error(f'Failed to save metadata {filename}: {e}')
        return False

if __name__=='__main__':
    try:
        logger.info('Generating random time indices...')
        timeidxs = get_random_time_idxs('pr')
        logger.info('Loading data...')
        varlist = INPUTVARS+[TARGETVAR]
        dalist  = load(varlist,timeidxs)
        logger.info('Preparing data splits with variable mapping...')
        splits,inputnamelist,inputmapping = split(dalist,INPUTVARS,TARGETVAR)
        logger.info('Saving data split and metadata...')
        save_data_splits(splits)
        save_metadata(inputmapping,inputnamelist)
        logger.info('Script completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')