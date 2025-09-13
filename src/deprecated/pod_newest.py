#!/usr/bin/env python

import os
import h5py
import pickle
import logging
import warnings
import numpy as np

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

FILEPATH = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/processed/data.h5'
SAVEDIR  = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/results'
CONFIGS  = [
    {'name':'bw_0.1','binwidth':0.1,'description':'Binwidth = 0.1 m/s²'},
    {'name':'bw_0.01','binwidth':0.01,'description':'Binwidth = 0.01 m/s²'},
    {'name':'bw_0.001','binwidth':0.001,'description':'Binwidth = 0.001 m/s²'}]
        
class PODMODEL:
    
    def __init__(self,binwidth,binmin=-0.6,binmax=0.1,samplethresh=50):
        '''
        Purpose: Initialize a buoyancy-based POD model for precipitation prediction.
        Args:
        - binwidth (float): width of each bin for discretizing BL input values (m/s²)
        - binmin (float): minimum boundary for the binning range (defaults to -0.6 m/s²)
        - binmax (float): maximum boundary for the binning range (defaults to 0.1 m/s²)
        - samplethresh (int): minimum number of samples required per bin to compute the bin average (defaults to 50)
        '''
        self.binwidth     = float(binwidth)
        self.binmin       = float(binmin)
        self.binmax       = float(binmax)
        self.binedges     = np.arange(self.binmin,self.binmax+self.binwidth,self.binwidth,dtype=np.float32)
        self.bincenters   = (self.binedges[:-1]+self.binedges[1:])/2
        self.nbins        = len(self.bincenters)
        self.samplethresh = int(samplethresh)
        self.binmeans     = None

    def fit(self,Xtrain,ytrain):
        '''
        Purpose: Train a POD model by computing average precipitation in each BL bin.
        Args:
        - Xtrain (numpy.ndarray): training BL values
        - ytrain (numpy.ndarray): training precipitation values
        Returns:
        - PODMODEL: fitted model (self)
        '''  
        idxs   = np.digitize(Xtrain,self.binedges)-1
        counts = np.zeros(self.nbins)
        sums   = np.zeros(self.nbins)
        for i in range(ytrain.size):
            idx    = idxs[i]
            yvalue = ytrain[i]
            if 0<=idx<self.nbins and np.isfinite(yvalue):
                counts[idx] += 1
                sums[idx]   += yvalue
        with np.errstate(divide='ignore',invalid='ignore'):
            means = sums/counts
        means[counts<self.samplethresh] = np.nan
        self.binmeans = means
        return self
        
    def predict(self,X):
        '''
        Purpose: Generate precipitation predictions from BL values using learned bin means.
        Args:
        - X (numpy.ndarray): BL values for prediction
        Returns:
        - numpy.ndarray: predicted precipitation values (non-negative constraint enforced)
        '''
        binidxs = np.clip(np.digitize(X,self.binedges)-1,0,self.nbins-1)
        ypred   = self.binmeans[binidxs]
        ypred   = np.maximum(ypred,0)
        return ypred

def load(filepath=FILEPATH):
    '''
    Purpose: Load data splits from an HDF5 file, combining training and validation sets for training.
    Args:
    - filepath (str): path to HDF5 file produced by split.py (defaults to FILEPATH)
    Returns:
    - tuple[numpy.ndarray,numpy.ndarray,numpy.ndarray,numpy.ndarray]: 1D BL and precipitation training/test arrays
    '''
    with h5py.File(filepath,'r') as f:
        Xtrainarray = np.concatenate([f['bl_train'][:],f['bl_valid'][:]],axis=0)
        Xtestarray  = f['bl_test'][:]
        ytrainarray = np.concatenate([f['pr_train'][:],f['pr_valid'][:]],axis=0)
        ytestarray  = f['pr_test'][:]
        Xtrain = Xtrainarray.squeeze().astype(np.float32)
        Xtest  = Xtestarray.squeeze().astype(np.float32)
        ytrain = ytrainarray.squeeze().astype(np.float32)
        ytest  = ytestarray.squeeze().astype(np.float32)
    return Xtrain,Xtest,ytrain,ytest
        
def process(configs=CONFIGS,filepath=FILEPATH):
    '''
    Purpose: Train and evaluate POD models with multiple bin width configurations.
    Args:
    - configs (list[dict[str,object]]): model configurations specifying bin widths and descriptions (defaults to CONFIGS)
    - filepath (str): path to HDF5 file produced by split.py
    Returns:
    - dict[str,dict[str,object]]: mapping from configuration name to POD results
    '''
    Xtrain,Xtest,ytrain,ytest = load(filepath)
    results = {}
    for config in configs:
        name        = config['name']
        binwidth    = config['binwidth']
        description = config['description']
        logger.info(f'   Running {description}')
        model = PODMODEL(binwidth)
        model.fit(Xtrain,ytrain)
        results[name] = {
            'description':description,
            'bin_width':model.binwidth,
            'bin_centers':model.bincenters,
            'bin_means':model.binmeans,
            'n_params':np.sum(~np.isnan(model.binmeans)),
            'y_pred':model.predict(Xtest)}
    return results

def save(results,filename='pod_results.pkl',savedir=SAVEDIR):
    '''
    Purpose: Save POD model results to a pickle file in the specified directory, then verify the write by reopening.
    Args:
    - results (dict[str,dict[str,object]]): POD model results to save
    - filename (str): output file name (defaults to 'pod_results.pkl')
    - savedir (str): output directory (defaults to SAVEDIR)
    Returns:
    - bool: True if write and verification succeed, otherwise False
    '''
    try:
        os.makedirs(savedir,exist_ok=True)
        filepath = os.path.join(savedir,filename)
        logger.info(f'Attempting to save results to {filepath}...')
        with open(filepath,'wb') as f:
            pickle.dump(results,f,protocol=pickle.HIGHEST_PROTOCOL)
        with open(filepath,'rb') as f:
            _ = pickle.load(f)
        logger.info('   File write successful')
        return True
    except Exception:
        logger.exception('   Failed to save or verify')
        return False

if __name__=='__main__':
    try:
        logger.info('Training POD models...')
        results = process()
        logger.info('Saving results...')
        save(results)
        del results
        logger.info('Script execution completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')