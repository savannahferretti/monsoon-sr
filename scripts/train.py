import os
import torch
import pickle
import warnings
from models import BASELINE,MLP

warnings.filterwarnings('ignore')

FILEDIR  = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/processed'
MODELDIR = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/models'

def load(filename,filedir=FILEDIR):
    '''
    Purpose: Load the prepared data split from a pickle file.
    Args:
    - filename (str): name of the file to open
    - filedir (str): directory containing the file (defaults to FILEDIR)
    Returns:
    - dict: dictionary containing X and y data for the particualr data split
    '''
    filepath = os.path.join(filedir,filename)
    with open(filepath,'rb') as file:
        datadict = pickle.load(file)
    return datadict
    
def train_baseline_models(xtrain,ytrain,xtest):
    '''
    Purpose: Train analytical baseline models with different bin widths and evaluate on test data.
    Args:
    - Xtrain (pd.DataFrame): training input features
    - ytrain (pd.Series): training target values
    - Xtest (pd.DataFrame): test input features
    Returns:
    - dict: dictionary containing baseline model results with keys for each binwidth configuration (description, binwidth, bincenters, binmeans, and testoutputs)
    '''
    print('Training Baseline Models')
    print('='*50)
    results = {}
    configs = [
        {'binwidth':0.1,'description':'Binwidth = 0.1'},
        {'binwidth':0.01,'description':'Binwidth = 0.01'},
        {'binwidth':0.001,'description':'Binwidth = 0.001'}]
    for config in configs:
        print(f'Training {config["description"]}...')
        baseline = BASELINE(
            binwidth=config['binwidth'],
            binmin=-0.6,
            binmax=0.1,
            samplethresh=50)
        model = baseline.fit(xtrain['bl'],ytrain)
        ypred = baseline.predict(xtest)
        results[f'binwidth_{config["binwidth"]}'] = {
            'description':config['description'],
            'binwidth':config['binwidth'],
            'bincenters':baseline.bincenters,
            'binmeans':baseline.model,
            'ypred':ypred}
    return results

def train_mlp_models(xtrain,ytrain,ytrainlog,xtest,ytest,ytestlog):
    '''
    Purpose: Train MLP models with various configurations (linear/nonlinear, MSE/MAE, log-transformed/regular) and save best models.
    Args:
    - xtrain (pd.DataFrame): training input features
    - ytrain (pd.DataFrame): training target values (original scale)
    - ytrainlog (pd.DataFrame): training target values (log-transformed)
    - xtest (pd.DataFrame): test input features
    - ytest (pd.DataFrame): test target values (original scale)
    - ytestlog (pd.DataFrame): test target values (log-transformed)
    Returns:
    - dict: dictionary containing MLP model results with keys for each model configuration,
            including trainlosses, validlosses, testoutputs, testloss, modelfile, description, 
            logtransform flag, activation type, and criterion type
    '''
    print('\nTraining MLP Models')
    print('='*50)
    configs = [
        {'name':'linear_mse','activation':torch.nn.Identity(),'criterion':torch.nn.MSELoss(),'logtransform':False,'description':'Linear MSE'},
        {'name':'linear_mae','activation':torch.nn.Identity(),'criterion':torch.nn.L1Loss(),'logtransform':False,'description':'Linear MAE'},
        {'name':'relu_mse','activation':torch.nn.ReLU(),'criterion':torch.nn.MSELoss(),'logtransform': False,'description':'Nonlinear MSE'},
        {'name':'relu_mae','activation':torch.nn.ReLU(),'criterion':torch.nn.L1Loss(),'logtransform':False,'description':'Nonlinear MAE'},
        {'name':'linear_mse_log','activation':torch.nn.Identity(),'criterion':torch.nn.MSELoss(),'logtransform':True,'description':'Log-Normalized Linear MSE'},
        {'name':'linear_mae_log','activation':torch.nn.Identity(),'criterion':torch.nn.L1Loss(),'logtransform':True,'description':'Log-Normalized Linear MAE'},
        {'name':'relu_mse_log','activation':torch.nn.ReLU(),'criterion':torch.nn.MSELoss(),'logtransform':True,'description':'Log-Normalized Nonlinear MSE'},
        {'name':'relu_mae_log','activation':torch.nn.ReLU(),'criterion':torch.nn.L1Loss(),'logtransform':True,'description':'Log-Normalized Nonlinear MAE'}]
    results = {}
    for config in configs:
        print(f"\nTraining {config['description']}")
        print(f"{'='*60}")
        mlp = MLP(
            inputsize=xtrain.shape[1],
            hiddensize=64,
            outputsize=ytrain.shape[1],
            nhiddenlayers=1,
            activation=config['activation'],
            criterion=config['criterion'],
            learningrate=0.001,
            nepochs=50,
            batchsize=500,
            validsplit=0.25,
            patience=2,
            randomstate=42,
            logtransform=config['logtransform'])
        if config['logtransform']:
            trainlosses,validlosses = mlp.fit(xtrain['bl'].values,ytrainlog['pr'].values)
            testoutputs             = mlp.predict(xtest['bl'].values)
            testloss                = mlp.evaluate(xtest['bl'].values,ytestlog['pr'].values)
        else:
            trainlosses,validlosses = mlp.fit(xtrain['bl'].values,ytrain['pr'].values)
            testoutputs             = mlp.predict(xtest['bl'].values)
            testloss                = mlp.evaluate(xtest['bl'].values,ytest['pr'].values)
        modelfilename = f'models/best_model_{config["name"]}.pth'
        torch.save(mlp.model.state_dict(),modelfilename)
        print(f'Model saved as {modelfilename}')
        results[config['name']] = {
            'trainlosses':trainlosses,
            'validlosses':validlosses,
            'testoutputs':testoutputs,
            'testloss':testloss,
            'modelfile':modelfilename,
            'description':config['description'],
            'logtransform':config['logtransform'],
            'activation':config['activation'].__class__.__name__,
            'criterion':config['criterion'].__class__.__name__}
    return results

if __name__=='__main__':
    logger.info('Load in training/testing data...')
    traindata = load('traindata.pkl')
    testdata  = load('testdata.pkl')
    logger.info('Extracting data...')
    Xtrain    = traindata['Xtrain']
    ytrain    = traindata['ytrain']
    ytrainlog = traindata['ytrainlog']
    Xtest    = testdata['Xtest']
    ytest    = testdata['ytest']
    ytestlog = testdata['ytestlog']
    
    # baseresults = train_baseline_models(xtrain['bl'],ytrain,xtest)


    
    mlpresults  = train_mlp_models(xtrain,ytrain,ytrainlog,xtest,ytest,ytestlog)
    with open(f'{SAVEDIR}/baseline_results.pkl','wb') as file:
        pickle.dump(baselineresults,file)
    with open(f'{SAVEDIR}/mlp_results.pkl','wb') as file:
        pickle.dump(mlpresults,file)
    print('\nTraining complete!')
    print('Baseline results saved to baseline_results.pkl')
    print('MLP results saved to mlp_results.pkl')
    print('Model files saved to models/ directory')