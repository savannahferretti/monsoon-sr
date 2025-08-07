import os
import torch
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from models import BASELINE,MLP

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

FILEDIR    = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/processed'
SAVEDIR    = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/results'
MODELDIR   = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/models/mlp'

BINCONFIGS = [
    {'binwidth':0.1,'description':'Binwidth = 0.1'},
    {'binwidth':0.01,'description':'Binwidth = 0.01'},
    {'binwidth':0.001,'description':'Binwidth = 0.001'}]
LINCONFIGS = [
    {'name':'linear_mse','activation':torch.nn.Identity(),'criterion':torch.nn.MSELoss(),'logtransform':False,'description':'Linear MSE'},
    {'name':'linear_mae','activation':torch.nn.Identity(),'criterion':torch.nn.L1Loss(),'logtransform':False,'description':'Linear MAE'},
    {'name':'linear_mse_log','activation':torch.nn.Identity(),'criterion':torch.nn.MSELoss(),'logtransform':True,'description':'Log-Normalized Linear MSE'},
    {'name':'linear_mae_log','activation':torch.nn.Identity(),'criterion':torch.nn.L1Loss(),'logtransform':True,'description':'Log-Normalized Linear MAE'}]
NONLINCONFIGS = [
    {'name':'relu_mse','activation':torch.nn.ReLU(),'criterion':torch.nn.MSELoss(),'logtransform': False,'description':'Nonlinear MSE'},
    {'name':'relu_mae','activation':torch.nn.ReLU(),'criterion':torch.nn.L1Loss(),'logtransform':False,'description':'Nonlinear MAE'},
    {'name':'relu_mse_log','activation':torch.nn.ReLU(),'criterion':torch.nn.MSELoss(),'logtransform':True,'description':'Log-Normalized Nonlinear MSE'},
    {'name':'relu_mae_log','activation':torch.nn.ReLU(),'criterion':torch.nn.L1Loss(),'logtransform':True,'description':'Log-Normalized Nonlinear MAE'}]
ARCHCONFIGS = [
    {'nhiddenlayers':1,'hiddenlayersize':64,'suffix':'','description':'1H-64N'},
    {'nhiddenlayers':1,'hiddenlayersize':128,'suffix':'_1x128','description':'1H-128N'},
    {'nhiddenlayers':2,'hiddenlayersize':64,'suffix':'_2x64','description':'2H-64N'},
    {'nhiddenlayers':2,'hiddenlayersize':128,'suffix':'_2x128','description':'2H-128N'},
    {'nhiddenlayers':3,'hiddenlayersize':64,'suffix':'_3x64','description':'3H-64N'},
    {'nhiddenlayers':3,'hiddenlayersize':128,'suffix':'_3x128','description':'3H-128N'}]

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
    
def train_baseline_models(xtrain,ytrain,xtest,binconfigs):
    '''
    Purpose: Train analytical baseline models with different bin widths and evaluate on test data.
    Args:
    - xtrain (pd.Series): training input feature
    - ytrain (pd.Series): training target values
    - xtest (pd.Series): test input feature
    - binconfigs (list): list of binwidth configuration dictionaries
    Returns:
    - dict: dictionary containing baseline model results
    '''
    results = {}
    for config in binconfigs:
        logger.info(f'Training {config["description"]}...')
        modelname = f'binwidth_{config["binwidth"]}'
        baseline  = BASELINE(
            binwidth=config['binwidth'],
            binmin=-0.6,
            binmax=0.1,
            samplethresh=50)
        model   = baseline.fit(xtrain,ytrain)
        ypred   = baseline.predict(xtest)
        nparams = np.sum(~np.isnan(baseline.model))
        results[modelname] = {
            'description':config['description'],
            'binwidth':config['binwidth'],
            'bincenters':baseline.bincenters,
            'binmeans':baseline.model,
            'nparams':nparams,
            'ypred':ypred}
    return results

def train_mlp_models(Xtrain,ytrain,ytrainlog,Xtest,ytest,ytestlog,mlpconfigs,archconfigs,modeldir=MODELDIR):
    '''
    Purpose: Train MLP models with various configurations and save best models.
    Args:
    - Xtrain (pd.Series or pd.DataFrame): training input feature(s)
    - ytrain (pd.Series or pd.DataFrame): training target values (original scale)
    - ytrainlog (pd.Series or pd.DataFrame): training target values (log-transformed)
    - Xtest (pd.Series or pd.DataFrame): test input feature(s)
    - ytest (pd.Series or pd.DataFrame): test target values (original scale)
    - ytestlog (pd.Series or pd.DataFrame):test target values (log-transformed)
    - mlpconfigs (list): list of MLP configuration dictionaries
    - archconfigs (list): list of architecture configuration dictionaries
    - modeldir (str): directory where the models should be saved (defaults to MODELDIR)
    Returns:
    - dict: dictionary containing MLP model results
    '''
    results = {}
    for archconfig in archconfigs:
        logger.info(f'Training models with {archconfig["description"]} architecture...')
        for config in mlpconfigs:
            modelname   = f'{config["name"]}{archconfig["suffix"]}'
            description = f'{config["description"]} ({archconfig["description"]})'
            logger.info(f'Training {description}')
            if hasattr(Xtrain,'shape'):
                if len(Xtrain.shape)==1:
                    inputlayersize = 1
                else:
                    inputlayersize = Xtrain.shape[1] 
            else:
                inputlayersize = 1
            mlp = MLP(
                inputlayersize=inputlayersize,
                hiddenlayersize=archconfig['hiddenlayersize'],
                nhiddenlayers=archconfig['nhiddenlayers'],
                activation=config['activation'],
                criterion=config['criterion'],
                learningrate=0.001,
                nepochs=30,
                batchsize=500,
                validsize=0.25,
                patience=2,
                randomstate=42,
                logtransform=config['logtransform'])
            nparams = sum(p.numel() for p in mlp.model.parameters())
            if config['logtransform']:
                losses   = mlp.fit(Xtrain,ytrainlog)
                ypred    = mlp.predict(Xtest)
                testloss = mlp.evaluate(Xtest,ytestlog)
            else:
                losses   = mlp.fit(Xtrain,ytrain)
                ypred    = mlp.predict(Xtest)
                testloss = mlp.evaluate(Xtest,ytest)
            filename = f'best_model_{modelname}.pth'
            filepath = os.path.join(modeldir,filename)
            torch.save(mlp.model.state_dict(),filepath)
            results[modelname] = {
                'description':description,
                'trainlosses':losses[0],
                'validlosses':losses[1],
                'testloss':testloss,
                'nparams':nparams,
                'ypred':ypred}
    return results

if __name__=='__main__':
    logger.info('Load in training/testing data...')
    Xtrain    = load('Xtrain.parquet')
    ytrain    = load('ytrain.parquet')
    ytrainlog = load('ytrainlog.parquet')
    Xtest     = load('Xtest.parquet')
    ytest     = load('ytest.parquet')
    ytestlog  = load('ytestlog.parquet')
    logger.info('Training baseline models...')
    baseresults   = train_baseline_models(Xtrain['bl'],ytrain['pr'],Xtest['bl'],BINCONFIGS)
    with open(f'{SAVEDIR}/baseline_results.pkl','wb') as file:
        pickle.dump(baseresults,file)
    del baseresults
    logger.info('Training linear MLP models...')
    linresults    = train_mlp_models(Xtrain['bl'],ytrain['pr'],ytrainlog['pr'],Xtest['bl'],ytest['pr'],ytestlog['pr'],LINCONFIGS,[ARCHCONFIGS[0]])
    with open(f'{SAVEDIR}/linear_results.pkl','wb') as file:
        pickle.dump(linresults,file)
    del linresults
    logger.info('Training nonlinear MLP models...')
    nonlinresults = train_mlp_models(Xtrain['bl'],ytrain['pr'],ytrainlog['pr'],Xtest['bl'],ytest['pr'],ytestlog['pr'],NONLINCONFIGS,ARCHCONFIGS)
    with open(f'{SAVEDIR}/nonlinear_results.pkl','wb') as file:
        pickle.dump(nonlinresults,file)    
    del nonlinresults
    print('\nTraining and saving complete!')
    print('Model files saved to models/ directory')