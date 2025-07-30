import os
import pickle
import logging
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

FILEDIR  = '/global/cfs/cdirs/m4334/sferrett/monsoon-pod/data/processed'
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
    filename = 'LR_ERA5_IMERG_pr_bl_terms.nc'
    filepath = os.path.join(filedir,filename)
    ds = xr.open_dataset(filepath)
    da = ds[varname].load()
    return da

def prepare(X,y,testsize=TESTSIZE):
    '''
    Purpose: Prepare features and target data for machine learning by merging, sorting, and splitting into training and test sets.
    Args:
    - X (list or xarray.DataArray): input feature(s) 
    - y (xarray.DataArray): target variable
    - testsize (float): fraction of data to use for testing (defaults to TESTSIZE)
    Returns:
    - tuple (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame): containing DataFrames for feature/target training/testing data splits
    '''    
    X = [X] if not isinstance(X,list) else X
    df = xr.merge([*X,y]).to_dataframe().reset_index()
    df = df.sort_values('time')
    Xcols = [x.name for x in X]
    ycol  = y.name
    Xtrain,Xtest,ytrain,ytest = train_test_split(df[Xcols],df[ycol],test_size=testsize,shuffle=False)
    return Xtrain,Xtest,ytrain,ytest

def save(datadict,filename,savedir=SAVEDIR):
    '''
    Purpose: Save a data dictionary to a pickle file in the specified directory. Verify the file was saved successfully by attempting to reopen it.
    Args:
    - datadict (dict): data dictionary to save
    - filename (str): name of the pickle file
    - savedir (str): directory where the file should be saved (defaults to SAVEDIR)
    Returns:
    - bool: True if the save operation was successful, False otherwise
    '''  
    filepath = os.path.join(savedir,filename)
    logger.info(f'Attempting to save {filename}...')   
    try:
        with open(filepath,'wb') as file:
            pickle.dump(datadict,file)
            logger.info(f'File written successfully: {filename}')
            with open(filepath,'rb') as file:
                test = pickle.load(file)
            logger.info(f'File verification successful: {filename}')
            return True
    except Exception as e:
        logger.error(f'Failed to save or verify {filename}: {e}')
        return False
        
if __name__=='__main__':
    try:
        logger.info('Loading data...')
        bl = load('bl')
        pr = load('pr')
        logger.info('Preparing train/test splits...')
        Xtrain,Xtest,ytrain,ytest = prepare([bl],pr)
        ytrainlog = np.log(ytrain+1)
        ytestlog  = np.log(ytest+1)
        logger.info('Saving training data...')
        traindata = {
            'Xtrain':Xtrain,
            'ytrain':ytrain,
            'ytrainlog':ytrainlog}
        save(traindata,'traindata.pkl')
        logger.info('Saving testing data...')
        testdata = {
            'Xtest':Xtest,
            'ytest':ytest,
            'ytestlog':ytestlog}
        save(testdata,'testdata.pkl')
        logger.info('All data saved successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')