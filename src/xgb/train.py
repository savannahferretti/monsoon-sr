#!/usr/bin/env python

import os
import json
import logging
import warnings
import numpy as np
import xarray as xr
from model import XGBModel

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

with open('configs.json','r',encoding='utf8') as f:
    CONFIGS = json.load(f)
FILEDIR     = CONFIGS['paths']['filedir']
MODELDIR    = CONFIGS['paths']['modeldir']
TARGETVAR   = CONFIGS['dataparams']['targetvar']
LANDVAR     = CONFIGS['dataparams']['landvar']
EXPERIMENTS = CONFIGS['experiments']
RUNS        = CONFIGS['runs']
TRAINPARAMS = CONFIGS['trainparams']

def reshape(da):
    '''
    Purpose: Convert an xr.DataArray into a 2D NumPy array suitable for model I/O.
    Args:
    - da (xr.DataArray): 3D or 4D DataArray
    Returns:
    - np.ndarray: shape (nsamples, nfeatures); for 3D, nfeatures=1, for 4D, nfeatures equals the size of the 'lev' dimension
    '''
    if 'lev' in da.dims:
        arr = da.transpose('time','lat','lon','lev').values.reshape(-1,da.lev.size)
    else:
        arr = da.transpose('time','lat','lon').values.reshape(-1,1)
    return arr

def load(splitname,inputvars,uself,landvar=LANDVAR,targetvar=TARGETVAR,filedir=FILEDIR):
    '''
    Purpose: Load in a normalized training or validation split and build a 2D feature matrix.
    Args:
    - splitname (str): 'norm_train' | 'norm_valid'
    - inputvars (list[str]): list of input variables
    - uself (bool): whether to include land fraction as an input feature
    - landvar (str): land fraction variable name (defaults to LANDVAR)
    - targetvar (str): target variable name (defaults to TARGETVAR)
    - filedir (str): directory containing split files (defaults to FILEDIR)
    Returns:
    - tuple[np.ndarray,np.ndarray]: 2D input/target arrays
    '''
    if splitname not in ('norm_train','norm_valid'):
        raise ValueError('Splitname must be `norm_train` or `norm_valid`.')
    filename = f'{splitname}.h5'
    filepath = os.path.join(filedir,filename)
    varlist  = list(inputvars)+[targetvar]
    if uself:
        varlist.append(landvar)
    ds    = xr.open_dataset(filepath,engine='h5netcdf')[varlist]
    Xlist = [reshape(ds[inputvar]) for inputvar in inputvars]
    if uself:
        Xlist.append(reshape(ds[landvar]))
    X = np.concatenate(Xlist,axis=1) if len(Xlist)>1 else Xlist[0]
    y = reshape(ds[targetvar])
    return X,y

def fit(model,runname,Xtrain,Xvalid,ytrain,yvalid,early_stopping_rounds=None):
    '''
    Purpose: Train an XGBoost model with early stopping.
    Args:
    - model (XGBModel): initialized model instance
    - runname (str): model run name
    - Xtrain (np.ndarray): training input(s)
    - Xvalid (np.ndarray): validation input(s)
    - ytrain (np.ndarray): training target
    - yvalid (np.ndarray): validation target
    - early_stopping_rounds (int): number of rounds to wait without validation improvement before early stopping
    Returns:
    - None: trains in-place and saves the best model checkpoint
    '''
    logger.info(f'      Training {runname}...')
    model.fit(Xtrain,ytrain,Xvalid,yvalid,early_stopping_rounds=early_stopping_rounds,verbose=False)

    # Log training results
    if model.evals_result:
        train_metric = list(model.evals_result['validation_0'].keys())[0]
        val_metric = list(model.evals_result['validation_1'].keys())[0]
        final_train_loss = model.evals_result['validation_0'][train_metric][-1]
        final_val_loss = model.evals_result['validation_1'][val_metric][-1]
        best_iteration = model.model.best_iteration
        logger.info(f'         Best iteration: {best_iteration}')
        logger.info(f'         Final training loss: {final_train_loss:.6f}')
        logger.info(f'         Final validation loss: {final_val_loss:.6f}')

    save(model,runname)

def save(model,runname,modeldir=MODELDIR):
    '''
    Purpose: Save trained XGBoost model to a JSON file in the specified directory, then verify the write by reopening.
    Args:
    - model (XGBModel): trained model instance
    - runname (str): model run name
    - modeldir (str): output directory (defaults to MODELDIR)
    Returns:
    - bool: True if write and verification succeed, otherwise False
    '''
    os.makedirs(modeldir,exist_ok=True)
    filename = f'xgb_{runname}.json'
    filepath = os.path.join(modeldir,filename)
    logger.info(f'      Attempting to save {filename}...')
    try:
        model.save(filepath)
        # Verify by loading
        test_model = XGBModel(**model.params)
        test_model.load(filepath)
        logger.info('         File write successful')
        return True
    except Exception:
        logger.exception('         Failed to save or verify')
        return False

if __name__=='__main__':
    try:
        explookup = {experiment['exp_name']:experiment for experiment in EXPERIMENTS}
        logger.info('Training and saving XGBoost models...')
        for run in RUNS:
            runname   = run['run_name']
            expname   = run['exp_name']
            uself     = run['use_lf']
            objective = run['objective']
            exp         = explookup[expname]
            inputvars   = exp['input_vars']
            description = exp['description']
            lfstr       = 'with' if uself else 'without'
            logger.info(f'   Training {description} {lfstr} land fraction using {objective} objective')
            Xtrain,ytrain = load('norm_train',inputvars,uself)
            Xvalid,yvalid = load('norm_valid',inputvars,uself)

            # Create model with parameters from config and run-specific objective
            model_params = {k:v for k,v in TRAINPARAMS.items() if k not in ['early_stopping_rounds']}
            model_params['objective'] = objective
            model = XGBModel(**model_params)

            early_stopping = TRAINPARAMS.get('early_stopping_rounds',None)
            fit(model,runname,Xtrain,Xvalid,ytrain,yvalid,early_stopping_rounds=early_stopping)
            del model,Xtrain,Xvalid,ytrain,yvalid
        logger.info('Script completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')
