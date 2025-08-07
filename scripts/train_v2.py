import os
import torch
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from models import BASELINE,MLPV2

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

FILEDIR    = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/processed'
SAVEDIR    = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/results'
MODELDIR   = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/models/mlp'

BINCONFIGS = [
    {'name':'bw_0.1','binwidth':0.1,'description':'Binwidth = 0.1 m/s$^2$'},
    {'name':'bw_0.01','binwidth':0.01,'description':'Binwidth = 0.01 m/s$^2$'},
    {'name':'bw_0.001','binwidth':0.001,'description':'Binwidth = 0.001 m/s$^2$'}]
MLPCONFIGS = [
    {'name':'mse_shallow','criterion':torch.nn.MSELoss(),'depth':'shallow','description':'MSE Shallow'},
    {'name':'mse_medium','criterion':torch.nn.MSELoss(),'depth':'medium','description':'MSE Medium'},
    {'name':'mse_deep','criterion':torch.nn.MSELoss(),'depth':'deep','description':'MSE Deep'},
    {'name':'mae_shallow','criterion':torch.nn.L1Loss(),'depth':'shallow','description':'MAE Shallow'},
    {'name':'mae_medium','criterion':torch.nn.L1Loss(),'depth':'medium','description':'MAE Medium'},
    {'name':'mae_deep','criterion':torch.nn.L1Loss(),'depth':'deep','description':'MAE Deep'}]
    
def load(filename,filedir=FILEDIR):
    filepath = os.path.join(filedir,filename)
    data     = pd.read_parquet(filepath)
    return data
    
def train_baseline_models(xtrain,ytrain,xtest,binconfigs):
    results = {}
    for config in binconfigs:
        logger.info(f'Training {config["description"]}')
        baseline  = BASELINE(
            binwidth=config['binwidth'],
            binmin=-0.6,
            binmax=0.1,
            samplethresh=50)
        model   = baseline.fit(xtrain,ytrain)
        ypred   = baseline.predict(xtest)
        nparams = np.sum(~np.isnan(baseline.model))
        results[config['name']] = {
            'description':config['description'],
            'binwidth':config['binwidth'],
            'bincenters':baseline.bincenters,
            'binmeans':baseline.model,
            'nparams':nparams,
            'ypred':ypred}
    return results

def train_mlp_models(Xtrain,ytrain,Xtest,ytest,mlpconfigs,modeldir=MODELDIR):
    results = {}
    if hasattr(Xtrain,'shape'):
        if len(Xtrain.shape)==1:
            inputlayersize = 1
        else:
            inputlayersize = Xtrain.shape[1] 
    else:
        inputlayersize = 1
        
    for config in mlpconfigs:
        logger.info(f'Training {config["description"]}')
        mlp = MLPV2(
            inputlayersize=inputlayersize,
            depth=config['depth'],
            criterion=config['criterion'])
        nparams  = sum(p.numel() for p in mlp.model.parameters())
        losses   = mlp.fit(Xtrain,ytrain)
        ypred    = mlp.predict(Xtest)
        testloss = mlp.evaluate(Xtest,ytest)
        filename = f'best_model_{config["name"]}.pth'
        filepath = os.path.join(modeldir,filename)
        torch.save(mlp.model.state_dict(),filepath)
        results[config['name']] = {
            'description':config['description'],
            'depth':config['depth'],
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
    Xtest     = load('Xtest.parquet')
    ytest     = load('ytest.parquet')
    
    baseresults = train_baseline_models(Xtrain['bl'],ytrain['pr'],Xtest['bl'],BINCONFIGS)
    with open(f'{SAVEDIR}/baseline_results.pkl','wb') as file:
        pickle.dump(baseresults,file)
    del baseresults
    
    mlpresults  = train_mlp_models(Xtrain['bl'],ytrain['pr'],Xtest['bl'],ytest['pr'],MLPCONFIGS)
    with open(f'{SAVEDIR}/v2_results.pkl','wb') as file:
        pickle.dump(mlpresults,file)
    del mlpresults
    
    print('\nTraining and saving complete!')
    print('Model files saved to models/ directory')