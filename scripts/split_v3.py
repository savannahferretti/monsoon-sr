import os
import pickle
import logging
import warnings
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

FILEDIR  = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/interim'
SAVEDIR  = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/processed'
TESTSIZE = 0.2

def load(varname,filedir=FILEDIR):
    '''
    Purpose: Load a variable from a NetCDF file as an xarray.DataArray.
    Args:
    - varname (str): name of the variable to load
    - filedir (str): directory containing the file (defaults to FILEDIR)
    Returns:
    - xarray.DataArray: loaded variable DataArray 
    '''
    filename = f'{varname}.nc'
    filepath = os.path.join(filedir,filename)
    ds = xr.open_dataset(filepath)
    da = da.sel(time=(da.time.dt.year.isin([2000,2010))
    da = ds[varname].load()
    return da

def prepare(X,y,testsize=TESTSIZE):
    '''
    Purpose: Prepare features and target data for machine learning by merging, sorting, and splitting into training and test sets.
             Handles both 3D and 4D variables by converting 4D variables into multiple 3D feature columns.
    Args:
    - X (list or xarray.DataArray): input feature(s); can be either 3D or 4D
    - y (xarray.DataArray): target variable (3D)
    - testsize (float): fraction of data to use for testing (defaults to TESTSIZE)
    Returns:
    - tuple (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame): containing DataFrames for feature/target training/testing data splits
    '''    
    X   = [X] if not isinstance(X,list) else X
    X3D = [x for x in X if 'lev' not in x.dims]
    X4D = [x for x in X if 'lev' in x.dims]
    datatomerge = [y]
    datatomerge.extend(X3D)
    for x in X4D:
        for lev in x.lev.values:
            xatlev = x.sel(lev=lev).drop_vars('lev')
            xatlev.name = f'{x.name}_{int(lev)}'
            datatomerge.append(xatlev)
    df = xr.merge(datatomerge).to_dataframe().reset_index()
    df = df.sort_values('time')
    Xcols = [var.name for var in datatomerge[1:]] 
    ycol  = y.name
    Xtrain,Xtest,ytrain,ytest = train_test_split(df[Xcols],df[ycol],test_size=testsize,shuffle=False)
    return Xtrain,Xtest,ytrain,ytest

def save(data,filename,savedir=SAVEDIR):
    '''
    Purpose: Save a Series or DataFrame to a single parquet file in the specified directory.
    Args:
    - data (pd.Series or pd.DataFrame): data to save
    - filename (str): name of the parquet file
    - savedir (str): directory where the file should be saved (defaults to SAVEDIR)
    Returns:
    - bool: True if the save operation was successful, False otherwise
    '''    
    filepath = os.path.join(savedir,filename)
    try:
        if isinstance(data,pd.DataFrame):
            data.to_parquet(filepath)
        elif isinstance(data,pd.Series):
            data.to_frame().to_parquet(filepath)
        else:
            pd.DataFrame(data).to_parquet(filepath)
        logger.info(f'Successfully saved {filename}')
        return True
    except Exception as e:
        logger.error(f'Failed to save {filename}: {e}')
        return False
        
if __name__=='__main__':
    try:
        logger.info('Loading data...')
        t = load('t')
        q = load('q')
        pr  = load('pr')
        bl  = load('bl')
        cape   = load('cape')
        subsat = load('subsat')
        capeprofile   = load('capeprofile')
        subsatprofile = load('subsatprofile')
        logger.info('Preparing train/test splits...')
        Xtrain,Xtest,ytrain,ytest = prepare([bl,cape,subsat,capeprofile,subsatprofile,t,q],pr)
        logger.info('Saving training data...')
        save(Xtrain,'Xtrain_v3.parquet')
        save(ytrain,'ytrain_v3.parquet') 
        logger.info('Saving testing data...')
        save(Xtest,'Xtest_v3.parquet')
        save(ytest,'ytest_v3.parquet')
        logger.info('All data saved successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')