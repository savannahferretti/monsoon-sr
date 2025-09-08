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

FILEDIR  = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/processed'
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
        self.binwidth     = binwidth
        self.binmin       = binmin
        self.binmax       = binmax
        self.binedges     = np.arange(self.binmin,self.binmax+self.binwidth,self.binwidth)
        self.bincenters   = (self.binedges[:-1]+self.binedges[1:])/2
        self.nbins        = len(self.bincenters)
        self.samplethresh = samplethresh
        self.binmeans     = None

    def fit(self,Xtrain,ytrain):
        '''
        Purpose: Train a POD model by computing average precipitation in each BL bin.
        Args:
        - Xtrain (numpy.ndarray): training input BL values
        - ytrain (numpy.ndarray): training target precipitation values
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
            self.binmeans = np.where(counts>=self.samplethresh,sums/counts,np.nan)

    def predict(self,X):
        '''
        Purpose: Generate precipitation predictions using the bin averages.
        Args:
        - X (numpy.ndarray): input values for prediction
        Returns:
        - numpy.ndarray: predicted precipitation values, with non-negative constraint enforced
        '''
        binidxs = np.clip(np.digitize(X,self.binedges)-1,0,self.nbins-1)
        ypred   = np.maximum((self.binmeans[binidxs]),0)
        return ypred

def load(filename,filedir=FILEDIR):
    '''
    Purpose: Load POD data splits from an HDF5 file.
    Args:
    - filename (str): name of the HDF5 file
    - filedir (str): directory containing the HDF5 file (defaults to FILEDIR)
    Returns:
    - (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray): BL and precipitation arrays organized by data split
    '''
    filepath = os.path.join(filedir,filename)
    with h5py.File(filepath,'r') as f:
        Xtrain = f['bl_train'][:]
        Xtest  = f['bl_test'][:]
        ytrain = f['pr_train'][:]
        ytest  = f['pr_test'][:]
    return Xtrain,Xtest,ytrain,ytest

def process(filename,configs=CONFIGS):
    '''
    Purpose: Train and evaluate POD models with multiple bin width configurations.
    Args:
    - filename (str): name of the HDF5 file
    - configs (list): model configurations specifying bin widths and descriptions (defaults to CONFIGS)
    Returns:
    - dict: dictionary containing POD model results
    '''
    Xtrain,Xtest,ytrain,ytest = load(filename)
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

if __name__=='__main__':
    try:
        logger.info('Training POD models...')
        results = process('pod_data_subset.h5')
        logger.info('Saving results...')
        with open(f'{SAVEDIR}/pod_subset_results.pkl','wb') as f:
            pickle.dump(results,f)
        del results
        logger.info('Script execution completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')