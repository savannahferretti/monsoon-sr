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
TRAINRANGE  = ('2000','2014')
VALIDRANGE  = ('2015','2017')
TESTRANGE   = ('2018','2020')

def retrieve(varname,filedir=FILEDIR):
    '''
    Purpose: Lazily import a variable as an xr.DataArray and standardize the dimension order.
    Args:
    - varname (str): variable short name
    - filedir (str): directory containing the NetCDF files (defaults to FILEDIR)
    Returns:
    - xr.DataArray: DataArray with standardized dimensions
    '''
    filename = f'{varname}.nc'
    filepath = os.path.join(filedir,filename)
    da = xr.open_dataarray(filepath,engine='h5netcdf')
    if 'lev' in da.dims:
        return da.transpose('time','lat','lon','lev')
    return da.transpose('time','lat','lon')

def make_mask(refda,splitrange,psfilepath=PSFILEPATH):
    '''
    Purpose: Create a below-surface mask for a given split (applies to all variables with a 'lev' dimension).
    Args:
    - refda (xr.DataArray): any variable for this split that has a 'lev' dimension
    - splitrange (tuple[str,str]): inclusive start/end years for the split
    - psfilepath (str): path to ERA5 surface pressure NetCDF file (defaults to PSFILEPATH)
    Returns:
    - xr.DataArray: uint8 mask that is 1 where the levels exist (lev ≤ ps), and 0 otherwise
    '''
    pssplit    = xr.open_dataarray(psfilepath,engine='h5netcdf').sel(time=slice(*splitrange))
    refdasplit = refda.sel(time=slice(*splitrange))
    mask = (refdasplit.lev<=pssplit).transpose('time','lat','lon','lev').astype('uint8')
    mask.name = 'mask'
    return mask


def split(splitname, splitrange, inputvars=INPUTVARS, targetvar=TARGETVAR):
    '''
    Purpose: Assemble the inputs/target data and one shared mask for a given split into a single xr.Dataset,
             and prepare an encoding dict for HDF5 (compression and chunking).
    Args:
    - splitname (str): 'train' | 'valid' | 'test'
    - splitrange (tuple[str,str]): inclusive start/end years for the split
    - inputvars (list[str]): input variable names (defaults to INPUTVARS)
    - targetvar (str): target variable name (defaults to TARGETVAR)
    Returns:
    - dict[str,object]: dictionary mapping of items needed to save the data split to disk
    '''
    datavars = {}
    y = retrieve(targetvar).sel(time=slice(*splitrange))
    datavars[targetvar] = y
    levels = None
    for inputvar in inputvars:
        da = retrieve(inputvar).sel(time=slice(*splitrange))
        datavars[inputvar] = da
        if ('lev' in da.dims) and levels is None:
            levels = da.lev.values
            datavars['mask'] = make_mask(da,splitrange)
    ds = xr.Dataset(datavars)
    encoding = {}
    for varname,da in ds.data_vars.items():
        chunks = (24,da.lat.size,da.lon.size,da.lev.size) if 'lev' in da.dims else (24,da.lat.size,da.lon.size)
        encoding[varname] = {'compression':'lzf','shuffle':True,'chunksizes':chunks}
    plan = {
        'split':splitname,
        'years':splitrange,
        'ds':ds,
        'encoding':encoding}
    logger.info(f'   Compiled {splitname} split')
    return plan

def save(splitname,plan,savedir=SAVEDIR):
    '''
    Purpose: Save a compiled xr.Dataset to a NetCDF file, then verify the write by reopening.
    Args:
    - splitname (str): 'train' | 'valid' | 'test'
    - plan (dict): output of split()
    - savedir (str): output directory (defaults to SAVEDIR)
    Returns:
    - bool: True if writing and verification succeed, otherwise False
    '''
    os.makedirs(savedir,exist_ok=True)
    filename = f'{splitname}.h5'
    filepath = os.path.join(savedir,filename)
    logger.info(f'Attempting to save {filename}...')
    try:
        plan['ds'].to_netcdf(filepath,engine='h5netcdf',encoding=plan['encoding'])
        with h5py.File(filepath,'r') as _:
            pass
        logger.info('   File write successful')
        return True
    except Exception:
        logger.exception('   Failed to save or verify')
        return False

if __name__=='__main__':
    try:
        splitdict = [
            ('train',TRAINRANGE),
            ('valid',VALIDRANGE),
            ('test',TESTRANGE)]
        for splitname,years in splitdict:
            plan = split(splitname,years)
            save(splitname,plan)
            del plan
        logger.info('All split files written successfully!')
    except Exception as e:
        logger.exception(f'Unexpected error: {e}')