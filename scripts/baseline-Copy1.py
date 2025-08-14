import os
import h5py
import json
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

FILEDIR  = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/processed'
SAVEDIR  = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/results'
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
        - x (np.ndarray): single input feature to be binned
        - y (np.ndarray): target values corresponding to inputs
        Returns:
        - tuple (np.ndarray, np.ndarray): containing arrays of bin centers and their corresponding average target values
        '''
        xvalues = x
        yvalues = y
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
        - x (np.ndarray): single input feature for which to make predictions
        Returns:
        - np.ndarray: predicted target values based on bin averages, with non-negative constraint enforced
        '''
        xvalues = x
        binidxs = np.clip(np.digitize(xvalues,self.binedges)-1,0,self.nbins-1)
        ypred   = np.maximum((self.model[binidxs]),0)
        return ypred

def load_hdf5_data():
    '''
    Purpose: Load the data splits from HDF5 file and metadata, combining train and validation for baseline training.
    Returns:
    - tuple: inputs_trainval, targets_trainval, inputs_test, targets_test, metadata
    '''
    datapath = os.path.join(FILEDIR, 'datasplits.h5')
    metadatapath = os.path.join(FILEDIR, 'metadata.json')
    
    with open(metadatapath, 'r') as f:
        metadata = json.load(f)
    
    with h5py.File(datapath, 'r') as f:
        inputs_train = torch.tensor(f['X_train'][:], dtype=torch.float32)
        targets_train = torch.tensor(f['y_train'][:], dtype=torch.float32)
        inputs_valid = torch.tensor(f['X_valid'][:], dtype=torch.float32)
        targets_valid = torch.tensor(f['y_valid'][:], dtype=torch.float32)
        inputs_test = torch.tensor(f['X_test'][:], dtype=torch.float32)
        targets_test = torch.tensor(f['y_test'][:], dtype=torch.float32)

    # Remove rows with NaN values for each split
    train_mask = ~torch.isnan(inputs_train).any(dim=1) & ~torch.isnan(targets_train)
    valid_mask = ~torch.isnan(inputs_valid).any(dim=1) & ~torch.isnan(targets_valid)
    test_mask = ~torch.isnan(inputs_test).any(dim=1) & ~torch.isnan(targets_test)
    
    inputs_train_clean = inputs_train[train_mask]
    targets_train_clean = targets_train[train_mask]
    inputs_valid_clean = inputs_valid[valid_mask]
    targets_valid_clean = targets_valid[valid_mask]
    inputs_test_clean = inputs_test[test_mask]
    targets_test_clean = targets_test[test_mask]
    
    # Combine training and validation data
    inputs_trainval = torch.cat([inputs_train_clean, inputs_valid_clean], dim=0)
    targets_trainval = torch.cat([targets_train_clean, targets_valid_clean], dim=0)
    
    logger.info(f"Training samples: {inputs_train_clean.shape[0]:,} (removed {(~train_mask).sum():,} NaN rows)")
    logger.info(f"Validation samples: {inputs_valid_clean.shape[0]:,} (removed {(~valid_mask).sum():,} NaN rows)")
    logger.info(f"Combined train+val samples: {inputs_trainval.shape[0]:,}")
    logger.info(f"Test samples: {inputs_test_clean.shape[0]:,} (removed {(~test_mask).sum():,} NaN rows)")
    
    return inputs_trainval.numpy(), targets_trainval.numpy(), inputs_test_clean.numpy(), targets_test_clean.numpy(), metadata

def extract_bl_feature(inputs, metadata):
    '''
    Purpose: Extract the BL (boundary layer) feature from the input array.
    Args:
    - inputs (np.ndarray): full input array with all features
    - metadata (dict): metadata containing input mapping
    Returns:
    - np.ndarray: BL feature values
    '''
    input_mapping = metadata['feature_mapping']
    bl_info = input_mapping['bl']
    bl_columns = bl_info['columns']
    
    # BL should be a single feature, so take the first (and only) column
    if len(bl_columns) != 1:
        raise ValueError(f"Expected BL to have 1 column, got {len(bl_columns)}")
    
    bl_column = bl_columns[0]
    return inputs[:, bl_column]
    
def process(inputs_train, targets_train, inputs_test, targets_test, metadata, configs=CONFIGS):
    '''
    Purpose: Train analytical baseline models with different bin widths and evaluate on test data.
    Args:
    - inputs_train (np.ndarray): training input features
    - targets_train (np.ndarray): training target values
    - inputs_test (np.ndarray): test input features
    - targets_test (np.ndarray): test target values (for ytrue in results)
    - metadata (dict): metadata containing input mapping
    - configs (list): list of "models" to train/test and their attributes (defaults to CONFIGS)
    Returns:
    - dict: dictionary containing baseline model results
    '''
    # Extract BL feature for baseline models
    bl_train = extract_bl_feature(inputs_train, metadata)
    bl_test = extract_bl_feature(inputs_test, metadata)
    
    results = {}
    for config in configs:
        name        = config['name']
        binwidth    = config['binwidth']
        description = config['description']
        logger.info(f'Training {description}')
        baseline = BASELINE(binwidth=binwidth,binmin=-0.6,binmax=0.1,samplethresh=50)
        model   = baseline.fit(bl_train, targets_train)
        ypred   = baseline.predict(bl_test)
        nparams = np.sum(~np.isnan(baseline.model))
        results[name] = {
            'description':description,
            'binwidth':binwidth,
            'bincenters':baseline.bincenters,
            'binmeans':baseline.model,
            'nparams':nparams,
            'ypred':ypred,
            'ytrue':targets_test  # Add true values for compatibility with plotting code
        }
    return results

if __name__=='__main__':
    logger.info('Loading data from HDF5 splits...')
    inputs_train, targets_train, inputs_test, targets_test, metadata = load_hdf5_data()
    
    logger.info('Training baseline models...')
    results = process(inputs_train, targets_train, inputs_test, targets_test, metadata)
    
    logger.info('Saving results...')
    os.makedirs(SAVEDIR, exist_ok=True)
    with open(f'{SAVEDIR}/v4_baseline_results.pkl','wb') as file:
        pickle.dump(results,file)
    
    # Print summary
    logger.info('Baseline model summary:')
    for name, result in results.items():
        logger.info(f"  {name}: {result['description']} - {result['nparams']} parameters")
    
    del results
    logger.info('Training and saving complete!')