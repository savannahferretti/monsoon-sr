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
PSFILEPATH  = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/raw/ERA5_surface_pressure.nc'
INPUTVARS   = ['bl','cape','subsat','capeprofile','subsatprofile','t','q']
TARGETVAR   = 'pr'
NYEARS      = 2
TRAINFRAC   = 0.7
RANDOMSTATE = 42

def get_random_time_idxs(refname,nyears=NYEARS,filedir=FILEDIR,randomstate=RANDOMSTATE):
    '''
    Purpose: Generate random time indices for data sampling based on a reference file.
    Args:
    - refname (str): name of the reference NetCDF file (without .nc extension)
    - nyears (int): number of years of data to sample (defaults to NYEARS)
    - filedir (str): directory containing the reference file (defaults to FILEDIR)
    - randomstate (int): Seed for reproducible sampling (defaults to RANDOMSTATE)
    Returns:
    - numpy.ndarray: sorted array of randomly selected time indices
    '''
    ####################################
    filename = f'{refname}_unfiltered.nc'
    ####################################
    filepath = os.path.join(filedir,filename)
    da = xr.open_dataarray(filepath,engine='h5netcdf')
    totaltimesteps = da.time.size
    totalyears     = np.unique(da.time.dt.year.values).size
    timestepsperyear = totaltimesteps/totalyears
    targettimesteps  = int(nyears*timestepsperyear)
    rng = np.random.default_rng(randomstate)
    timeidxs = np.sort(rng.choice(totaltimesteps,size=targettimesteps,replace=False))
    logger.info(f'   Selected {(len(timeidxs)/totaltimesteps)*100:.2f}% timesteps to use for data split')
    return timeidxs
    
def load(varnames,timeidxs=None,filedir=FILEDIR):
    '''
    Purpose: Load multiple NetCDF variables as xarray.DataArrays with optional time indexing.
    Args:
    - varnames (list[str]): list of variable names to load
    - timeidxs (numpy.ndarray, optional): time indices to select from each variable
    - filedir (str): directory containing the NetCDF files (defaults to FILEDIR)
    Returns:
    - dict[str,xarray.DataArray]: mapping from variable name to loaded DataArray
    '''
    dalist = {}
    for varname in varnames:
        ####################################
        filename = f'{varname}_unfiltered.nc'
        ####################################
        filepath = os.path.join(filedir,filename)
        da = xr.open_dataarray(filepath,engine='h5netcdf')
        if timeidxs is not None:
            da = da.isel(time=timeidxs)
        dalist[varname] = da.load()
        logger.info(f'   Loaded {varname}')
    return dalist

def create_splits(nsamples,trainfrac=TRAINFRAC,randomstate=RANDOMSTATE):
    '''
    Purpose: Create random, disjoint temporal data splits for training, validation, and testing.
    Args:
    - nsamples (int): total number of samples to split (time × lat × lon)
    - trainfrac (float): fraction of samples to use for training (defaults to TRAINFRAC)
    - randomstate (int): random seed for reproducible splits (defaults to RANDOMSTATE)
    Returns:
    - tuple[numpy.ndarray,numpy.ndarray,numpy.ndarray]: training, validation, and test set indices
    '''
    rng = np.random.default_rng(randomstate)
    trainsize = int(trainfrac*nsamples)
    validsize = int(((1-trainfrac)/2)*nsamples)
    idxs = np.arange(nsamples)
    rng.shuffle(idxs)
    trainidxs = idxs[:trainsize]
    valididxs = idxs[trainsize:trainsize+validsize]
    testidxs  = idxs[trainsize+validsize:]
    return trainidxs,valididxs,testidxs

def reshape(da):
    '''
    Purpose: Convert an xarray.DataArray to a reshaped NumPy array with shape (nsamples, nlevels), where 
    surface variables have shape (nsamples, 1).
    Args:
    - da (xarray.DataArray): 3D (time, lat, lon) or 4D (time, lat, lon, lev) DataArray
    Returns:
    - numpy.ndarray: reshaped array
    '''
    if 'lev' in da.dims:
        nsamples = da.time.size*da.lat.size*da.lon.size
        nlevels  = da.lev.size
        return da.values.reshape(nsamples,nlevels)
    else:
        return da.values.flatten().reshape(-1,1)

# def split(dalist,inputvars=INPUTVARS,targetvar=TARGETVAR):
#     '''
#     Purpose: Prepare model-ready arrays by reshaping and splitting the data.
#     Args:
#     - dalist (dict[str,xarray.DataArray]): dictionary of loaded DataArrays
#     - inputvars (list[str]): input variable names (defaults to INPUTVARS)
#     - targetvar (str): target variable name (defaults to TARGETVAR)
#     Returns:
#     - dict[str,numpy.ndarray]: reshaped input/target arrays organized by data split
#     '''
#     y = reshape(dalist[targetvar])        
#     trainidxs,valididxs,testidxs = create_splits(y.shape[0])
#     datadict = {}
#     for inputvar in inputvars:
#         X = reshape(dalist[inputvar])          
#         datadict[f'{inputvar}_train'] = X[trainidxs]
#         datadict[f'{inputvar}_valid'] = X[valididxs]
#         datadict[f'{inputvar}_test']  = X[testidxs]
#     datadict[f'{targetvar}_train'] = y[trainidxs]
#     datadict[f'{targetvar}_valid'] = y[valididxs]
#     datadict[f'{targetvar}_test']  = y[testidxs]
#     return datadict

def split(dalist,inputvars=INPUTVARS,targetvar=TARGETVAR):
    '''
    Purpose: Prepare model-ready arrays by reshaping and splitting the data.
    Args:
    - dalist (dict[str,xarray.DataArray]): dictionary of loaded DataArrays
    - inputvars (list[str]): input variable names (defaults to INPUTVARS)
    - targetvar (str): target variable name (defaults to TARGETVAR)
    Returns:
    - dict[str,numpy.ndarray]: reshaped input/target arrays organized by data split
    '''
    psall = xr.open_dataarray(PSFILEPATH,engine='h5netcdf')
    y  = reshape(dalist[targetvar])
    trainidxs,valididxs,testidxs = create_splits(y.shape[0])
    datadict = {}
    for inputvar in inputvars:
        da = dalist[inputvar]            
        X  = reshape(da)                       
        if 'lev' in da.dims:
            ps   = psall.sel(time=da.time,lat=da.lat,lon=da.lon)
            mask = xr.where(da.lev<=ps,1,0).transpose('time','lat','lon','lev')
            mask = mask.values.reshape(X.shape).astype('uint8')
        else:
            mask = np.ones_like(X,dtype='uint8')
        datadict[f'{inputvar}_train'] = X[trainidxs]
        datadict[f'{inputvar}_valid'] = X[valididxs]
        datadict[f'{inputvar}_test']  = X[testidxs]
        datadict[f'mask_{inputvar}_train'] = mask[trainidxs]
        datadict[f'mask_{inputvar}_valid'] = mask[valididxs]
        datadict[f'mask_{inputvar}_test']  = mask[testidxs]
    datadict[f'{targetvar}_train'] = y[trainidxs]
    datadict[f'{targetvar}_valid'] = y[valididxs]
    datadict[f'{targetvar}_test']  = y[testidxs]
    return datadict

def save(datadict,metadata=None,filename='data.h5',savedir=SAVEDIR):
    '''
    Purpose: Save a data dictionary (with metadata) to an HDF5 file, then verify the write by reopening.
    Args:
    - datadict (dict[str,numpy.ndarray]): data dictionary to save
    - metadata (dict[str,str], optional): metadata to save as file attributes
    - filename (str): output file name (defaults to 'data.h5')
    - savedir (str): output directory (defaults to SAVEDIR)
    Returns:
    - bool: True if writing and verification succeed, otherwise False
    '''    
    filepath = os.path.join(savedir,filename)
    logger.info(f'Attempting to save {filename}...')
    try:
        with h5py.File(filepath,'w') as f:
            for key,array in datadict.items():
                f.create_dataset(key,data=array,chunks=True)
            if metadata is not None:
                for key,value in metadata.items():
                    f.attrs[key] = value
        with h5py.File(filepath,'r') as _:
            pass
        logger.info('   File write successful')
        return True
    except Exception:
        logger.exception('   Failed to save or verify')
        return False

if __name__=='__main__':
    try:
        logger.info('Generating random time indices...')
        timeidxs = get_random_time_idxs('pr')
        logger.info('Loading data...')
        varlist = INPUTVARS+[TARGETVAR]
        dalist  = load(varlist,timeidxs)
        logger.info('Preparing data splits...')
        datadict = split(dalist)
        metadata = {
            'inputvars':','.join(INPUTVARS),
            'targetvar':TARGETVAR,
            'note':'Inputs/targets are shaped (nsamples, nlevels), with surface variables shaped (nsamples, 1).'
                   'Per-input masks saved as mask_<varname>_*, where 1 = above-surface, 0 = below-surface.'}
        # metadata = {
        #     'inputvars':','.join(INPUTVARS),
        #     'targetvar':TARGETVAR,
        #     'note':'Inputs/targets are shaped (nsamples, nlevels), with surface variables shaped (nsamples, 1).'}
        save(datadict,metadata=metadata)
        logger.info('Script completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')