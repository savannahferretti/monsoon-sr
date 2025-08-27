#!/usr/bin/env python

import os
import h5py
import logging
import warnings
import numpy as np
import xarray as xr

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

FILEDIR     = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/interim'
SAVEDIR     = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/processed'
INPUTVARS   = ['bl','cape','subsat','capeprofile','subsatprofile','t','q']
TARGETVAR   = 'pr'
NYEARS      = 2
TRAINFRAC   = 0.7
RANDOMSTATE = 42

def get_random_time_idxs(refname,nyears=NYEARS,filedir=FILEDIR):
    '''
    Purpose: Generate random time indices for data sampling based on a reference file.
    Args:
    - refname (str): name of the reference NetCDF file (without .nc extension)
    - nyears (int): number of years worth of data to sample (defaults to NYEARS)
    - filedir (str): directory containing the reference file (defaults to FILEDIR)
    Returns:
    - numpy.ndarray: sorted array of randomly selected time indices
    '''
    filename = f'{refname}.nc'
    filepath = os.path.join(filedir,filename)
    da = xr.open_dataarray(filepath).load()
    totaltimesteps = len(da.time)
    uniqueyears    = len(np.unique(da.time.dt.year.values))
    timestepsperyear = totaltimesteps/uniqueyears
    targettimesteps  = int(nyears*timestepsperyear)
    timeidxs = np.sort(np.random.choice(totaltimesteps,size=targettimesteps,replace=False))
    logger.info(f'   Selected {(len(timeidxs)/totaltimesteps)*100:.2f}% timesteps to use for data split')
    return timeidxs

def load(varnames,timeidxs=None,filedir=FILEDIR):
    '''
    Purpose: Load multiple NetCDF variables as xarray.DataArrays with optional time indexing.
    Args:
    - varnames (list): list of variable names to load
    - timeidxs (numpy.ndarray, optional): time indices to select from each variable
    - filedir (str): directory containing the NetCDF files (defaults to FILEDIR)
    Returns:
    - dict: dictionary mapping variable names to their corresponding xarray.DataArrays
    '''
    dalist = {}
    for varname in varnames:
        filename = f'{varname}.nc'
        filepath = os.path.join(filedir,filename)
        da = xr.open_dataarray(filepath).load()
        if timeidxs is not None:
            da = da.isel(time=timeidxs)
        dalist[varname] = da
        logger.info(f'   Loaded {varname}')
    return dalist

def create_temporal_splits(nsamples,trainfrac=TRAINFRAC,randomstate=RANDOMSTATE):
    '''
    Purpose: Create random temporal data splits for training, validation, and testing.
    Args:
    - nsamples (int): total number of samples to split (time × lat × lon)
    - trainfrac (float): fraction of data to use for training (defaults to TRAINFRAC)
    - randomstate (int): random seed for reproducible splits (defaults to RANDOMSTATE)
    Returns:
    - (numpy.ndarray, numpy.ndarray, numpy.ndarray): training, validation, and test set indices
    '''
    trainsize = int(trainfrac*nsamples)
    validsize = int(((1-trainfrac)/2)*nsamples)
    testsize  = nsamples-trainsize-validsize
    idxs = np.arange(nsamples)
    np.random.shuffle(idxs)
    trainidxs = idxs[:trainsize]
    valididxs = idxs[trainsize:trainsize+validsize]
    testidxs  = idxs[trainsize+validsize:]
    return trainidxs,valididxs,testidxs

def reshape(da,shape):
    '''
    Purpose: Convert an xarray.DataArray to a reshaped NumPy array.
    Args:
    - da (xarray.DataArray): input 3D (time, lat, lon) or 4D (time, lat, lon, lev) DataArray
    - shape (str): output array shape, '1D' for flattened arrays or '2D' for (nsamples, nfeatures) arrays
    Returns:
    - numpy.ndarray: reshaped array (1D for POD models, 2D for ML-based models)
    '''
    if shape=='1D':
        array = da.values.flatten()
    elif shape=='2D':
        if 'lev' in da.dims:
            nsamples = da.time.size*da.lat.size*da.lon.size
            nlevels  = da.lev.size
            array = da.values.reshape(nsamples,nlevels)
        else:
            array = da.values.flatten().reshape(-1,1)
    else:
        logger.error('Invalid shape parameter, must be "1D" or "2D"')
        return None
    return array

def normalize(array,mean=None,std=None):
    '''
    Purpose: Normalize an array to zero mean and unit variance using provided or calculated parameters.
    Args:
    - array (numpy.ndarray): input array to normalize
    - mean (float, optional): mean for normalization (if None, calculated from array)
    - std (float, optional): standard deviation for normalization (if None, calculated from array)
    Returns:
    - (numpy.ndarray, float, float): the normalized array, and the mean/standard deviation used for normalization
    '''
    if mean is None:
        mean = np.nanmean(array)
    if std is None:
        std  = np.nanstd(array)
    normarray = (array-mean)/std
    return normarray,mean,std

def prepare_pod(dalist,trainidxs,valididxs,testidxs,shape='1D',inputvar='bl',targetvar=TARGETVAR):
    '''
    Purpose: Prepare data for use with the buoyancy-based POD models by reshaping and splitting the data
             (where training and validation splits are combined).
    Args:
    - dalist (dict): dictionary of loaded xarray.DataArrays
    - trainidxs (numpy.ndarray): indices for the training set
    - valididxs (numpy.ndarray): indices for the validation set 
    - testidxs (numpy.ndarray): indices for the test set
    - shape (str): output array shape, '1D' for flattened arrays or '2D' for (nsamples, nfeatures) arrays (defaults to '1D')
    - inputvar (str): input variable name (defaults to 'bl')
    - targetvar (str): target variable name (defaults to TARGETVAR)
    Returns:
    - dict: dictionary containing reshaped input/target data organized by data split
    '''
    datadict = {}
    X = reshape(dalist[inputvar],shape)
    y = reshape(dalist[targetvar],shape) 
    datadict[f'{inputvar}_train']  = X[np.concatenate([trainidxs,valididxs])]
    datadict[f'{inputvar}_test']   = X[testidxs]
    datadict[f'{targetvar}_train'] = y[np.concatenate([trainidxs,valididxs])]
    datadict[f'{targetvar}_test']  = y[testidxs]
    return datadict

def prepare_ml(dalist,trainidxs,valididxs,testidxs,shape='2D',inputvars=INPUTVARS,targetvar=TARGETVAR):
    '''
    Purpose: Prepare data for use with the ML models by reshaping, splitting, and normalizing the data.
    Args:
    - dalist (dict): dictionary of loaded xarray.DataArrays
    - trainidxs (numpy.ndarray): indices for the training set
    - valididxs (numpy.ndarray): indices for the validation set 
    - testidxs (numpy.ndarray): indices for the test set
    - shape (str): output array shape, '1D' for flattened arrays or '2D' for (nsamples, nfeatures) arrays (defaults to '2D')
    - inputvars (list): input variable names (defaults to INPUTVARS)
    - targetvar (str): target variable name (defaults to TARGETVAR)
    Returns:
    - (dict, dict): dictionary containing reshaped input/target data organized by data split, and normalization parameters
    '''
    datadict   = {}
    normparams = {}
    for inputvar in inputvars:
        X = reshape(dalist[inputvar],shape)
        Xtrain,mean,std = normalize(X[trainidxs])
        Xvalid,_,_      = normalize(X[valididxs],mean,std)
        Xtest,_,_       = normalize(X[testidxs],mean,std)
        datadict[f'{inputvar}_train']  = Xtrain
        datadict[f'{inputvar}_valid']  = Xvalid
        datadict[f'{inputvar}_test']   = Xtest
        normparams[f'{inputvar}_mean'] = mean
        normparams[f'{inputvar}_std']  = std
    y = reshape(dalist[targetvar],shape)
    datadict[f'{targetvar}_train'] = y[trainidxs]
    datadict[f'{targetvar}_valid'] = y[valididxs]
    datadict[f'{targetvar}_test']  = y[testidxs]
    return datadict,normparams

def save(datadict,normparams,filename,savedir=SAVEDIR):
    '''
    Purpose: Save a data dictionary (and normalization parameters, if applicable) to an HDF5 file in the specified directory.
             Verify the file was saved successfully by attempting to reopen it.      
    Args:
    - datadict (dict): flat dictionary containing arrays to save
    - normparams (dict): normalization statistics (means/standard deviations) or None
    - filename (str): name of output file (should include .h5 extension)
    - savedir (str): directory to save file (defaults to SAVEDIR)
    Returns:
    - bool: True if save successful, False otherwise
    '''    
    filepath = os.path.join(savedir,filename)
    logger.info(f'Attempting to save {filename}...')
    try:
        with h5py.File(filepath,'w') as f:
            for key,array in datadict.items():
                f.create_dataset(key,data=array,chunks=True)
            if normparams is not None:
                normgroup = f.create_group('normalization')
                for key,value in normparams.items():
                    normgroup.attrs[key] = value
        logger.info(f'   File writing successful')
        with h5py.File(filepath,'r') as f:
            pass
        logger.info(f'   File verification successful')
        return True
    except Exception as e:
        logger.error(f'   Failed to save or verify: {e}')
        return False
        
if __name__=='__main__':
    try:
        np.random.seed(RANDOMSTATE)
        logger.info('Generating random time indices...')
        timeidxs = get_random_time_idxs('pr')
        logger.info('Importing data...')
        varlist = INPUTVARS+[TARGETVAR]
        dalist  = load(varlist,timeidxs)
        logger.info('Creating temporal data splits...')
        nsamples = len(dalist['pr'].values.flatten())
        trainidxs,valididxs,testidxs = create_temporal_splits(nsamples)
        logger.info(f'Split Sizes: {len(trainidxs):,} training, {len(valididxs):,} validation, {len(testidxs):,} testing')
        logger.info('Preparing POD data...')
        poddata = prepare_pod(dalist,trainidxs,valididxs,testidxs)
        save(poddata,None,'pod_data_subset.h5')
        del poddata
        logger.info('Preparing ML model data...')
        mldata,normparams = prepare_ml(dalist,trainidxs,valididxs,testidxs)
        save(mldata,normparams,'ml_data_subset.h5')
        del mldata
        logger.info('Script completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')