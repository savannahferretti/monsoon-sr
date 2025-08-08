import os
import pickle
import logging
import warnings
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

FILEDIR  = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/processed'
SAVEDIR  = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/results'
MODELDIR = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/models/mlp'
CONFIGS  = [
    {'name':'bw_0.1','binwidth':0.1,'description':'Binwidth = 0.1 m/s$^2$'},
    {'name':'bw_0.01','binwidth':0.01,'description':'Binwidth = 0.01 m/s$^2$'},
    {'name':'bw_0.001','binwidth':0.001,'description':'Binwidth = 0.001 m/s$^2$'}]

class BASELINE:
    
    def __init__(self,binwidth,binmin,binmax,samplethresh):
        '''
        Purpose: Initialize an analytical baseline model that predicts target values based on input feature bin averages.
        Args:
        - binwidth (float): width of each bin for discretizing input values
        - binmin (float): minimum value for binning range
        - binmax (float): maximum value for binning range
        - samplethresh (int): minimum number of samples required per bin to compute the bin average
        '''
        self.binwidth     = binwidth
        self.binmin       = binmin
        self.binmax       = binmax
        self.binedges     = np.arange(self.binmin,self.binmax+self.binwidth,self.binwidth)
        self.bincenters   = (self.binedges[:-1]+self.binedges[1:])/2
        self.nbins        = len(self.bincenters)
        self.samplethresh = samplethresh
        self.model        = None

    def fit(self,x,y):
        '''
        Purpose: Train the baseline model by computing average target values for each feature bin.
        Args:
        - x (pd.Series): single input feature to be binned
        - y (pd.Series): target values corresponding to inputs
        Returns:
        - tuple (np.ndarray, np.ndarray): containing arrays of bin centers and their corresponding average target values
        '''
        xvalues = x.values
        yvalues = y.values
        idxs  = np.digitize(xvalues,self.binedges)-1
        Q0,Q1 = np.zeros(self.nbins),np.zeros(self.nbins)
        for i in range(yvalues.size):
            idx    = idxs[i]
            yvalue = yvalues[i]
            if 0<=idx<self.nbins and np.isfinite(yvalue):
                Q0[idx] += 1
                Q1[idx] += yvalue
        with np.errstate(divide='ignore',invalid='ignore'):
            self.model = np.where(Q0>=self.samplethresh,Q1/Q0,np.nan)
        return (self.bincenters,self.model)

    def predict(self,x):
        '''
        Purpose: Make predictions using the trained baseline model by mapping the input feature to its corresponding bin averages.
        Args:
        - x (pd.Series): single input feature for which to make predictions
        Returns:
        - np.ndarray: predicted target values based on bin averages, with non-negative constraint enforced
        '''
        xvalues = x.values
        binidxs = np.clip(np.digitize(xvalues,self.binedges)-1,0,self.nbins-1)
        ypred   = np.maximum((self.model[binidxs]),0)
        return ypred

def load(filename,filedir=FILEDIR):
    '''
    Purpose: Load the prepared data from a parquet file.
    Args:
    - filename (str): name of the file to open
    - filedir (str): directory containing the file (defaults to FILEDIR)
    Returns:
    - pd.DataFrame: loaded data
    '''
    filepath = os.path.join(filedir,filename)
    data     = pd.read_parquet(filepath)
    return data
    
def process(xtrain,ytrain,xtest,configs=CONFIGS):
    '''
    Purpose: Train analytical baseline models with different bin widths and evaluate on test data.
    Args:
    - xtrain (pd.Series): training input feature
    - ytrain (pd.Series): training target values
    - xtest (pd.Series): test input feature
    - configs (list): list of "models" to train/test and their attributes (defaults to CONFIGS)
    Returns:
    - dict: dictionary containing baseline model results
    '''
    results = {}
    for config in configs:
        name        = config['name']
        binwidth    = config['binwidth']
        description = config['description']
        logger.info(f'Training {description}')
        baseline = BASELINE(binwidth=binwidth,binmin=-0.6,binmax=0.1,samplethresh=50)
        model   = baseline.fit(xtrain,ytrain)
        ypred   = baseline.predict(xtest)
        nparams = np.sum(~np.isnan(baseline.model))
        results[name] = {
            'description':description,
            'binwidth':binwidth,
            'bincenters':baseline.bincenters,
            'binmeans':baseline.model,
            'nparams':nparams,
            'ypred':ypred}
    return results

if __name__=='__main__':
    logger.info('Load in training/testing data...')
    Xtrain  = load('Xtrain.parquet')
    ytrain  = load('ytrain.parquet')
    Xtest   = load('Xtest.parquet')
    logger.info('Training baseline models...')
    results = process(Xtrain['bl'],ytrain['pr'],Xtest['bl'])
    logger.info('Saving results...')
    with open(f'{SAVEDIR}/baseline_results.pkl','wb') as file:
        pickle.dump(results,file)
    del results
    logger.info('Training and saving complete!')