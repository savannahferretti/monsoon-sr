#!/usr/bin/env python

import os
import time
import h5py
import torch
import wandb
import pickle
import logging
import warnings
import numpy as np

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

FILEPATH = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/processed/data.h5'
SAVEDIR  = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/results'
MODELDIR = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/models/nn'
CONFIGS  = [
    {'name':'exp_1','inputvars':['bl'],'description':'Experiment 1'},
    {'name':'exp_2','inputvars':['cape','subsat'],'description':'Experiment 2'},
    {'name':'exp_3','inputvars':['capeprofile'],'description':'Experiment 3'},
    {'name':'exp_4','inputvars':['subsatprofile'],'description':'Experiment 4'},
    {'name':'exp_5','inputvars':['capeprofile','subsatprofile'],'description':'Experiment 5'},
    {'name':'exp_6','inputvars':['t','q'],'description':'Experiment 6'}]

class NNMODEL:
    
    def __init__(self,inputsize,batchsize=64000,epochs=30,criterion=nn.MSELoss(),learningrate=1e-4,patience=3):
        '''
        Purpose: Initialize a NN model for precipitation prediction.
        Args:
        - inputsize (int): number of input features
        - batchsize (int): the batch size for training (defaults to 64,000)
        - epochs (int): maximum number of training epochs (defaults to 30)
        - criterion (torch.nn.Module): the loss function (defaults to MSE)
        - learningrate (float): learning rate (defaults to 0.0001)
        - patience (int): early stopping patience (defaults to 3)
        '''
        self.inputsize    = int(inputsize)
        self.batchsize    = int(batchsize)
        self.epochs       = int(epochs)
        self.criterion    = criterion
        self.learningrate = float(learningrate)
        self.patience     = int(patience)
        self.device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model        = MLP(self.inputsize).to(self.device)
        self.bestloss     = float('inf')
        self.bestepoch    = 0
        self.beststate    = None    

    def save(self,name,modeldir=MODELDIR):
        '''
        Purpose: Save the best model (lowest validation loss) to a PyTorch checkpoint file.
        Args:
        - name (str): model name prefix
        - modeldir (str): output directory (defaults to MODELDIR)
        '''
        os.makedirs(modeldir,exist_ok=True)
        filename = f'{name}_best.pth'
        filepath = os.path.join(modeldir,filename)
        torch.save(self.beststate,path)
        self.bestpath = filepath

    def load(self,name,modeldir=MODELDIR):
        '''
        Purpose: Load the best model (lowest validation loss) weights from disk and set model to eval().
        Args:
        - name (str): model name prefix
        - modeldir (str): directory where model weights are saved (defaults to MODELDIR)
        '''
        filename = f'{name}_best.pth'
        filepath = os.path.join(modeldir,filename)
        state = torch.load(filepath,map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.bestpath = filepath

        
    def fit(self,Xtrain,Xvalid,ytrain,yvalid):
        '''
        Purpose: Train a NN model with early stopping and learning rate scheduling.
        Args:
        - Xtrain (torch.Tensor): training input features
        - Xvalid (torch.Tensor): validation input features
        - ytrain (torch.Tensor): training precipitation values
        - yvalid (torch.Tensor): validation precipitation values
        '''
        traindataset = torch.utils.data.TensorDataset(Xtrain,ytrain)
        validdataset = torch.utils.data.TensorDataset(Xvalid,yvalid)
        trainloader = torch.utils.data.DataLoader(traindataset,batch_size=self.batchsize,shuffle=True,num_workers=8,pin_memory=True)
        validloader = torch.utils.data.DataLoader(validdataset,batch_size=self.batchsize,shuffle=False,num_workers=8,pin_memory=True)
        optimizer   = torch.optim.Adam(self.model.parameters(),lr=self.learningrate)
        scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=self.patience)
        counter     = 0
        starttime   = time.time()
        for epoch in range(self.epochs):
            trainloss = 0.0
            self.model.train()
            for X,y in trainloader:
                X,y = X.to(self.device),y.to(self.device)
                optimizer.zero_grad()
                ypred = self.model(X)
                loss  = self.criterion(ypred.squeeze(),y.squeeze())
                loss.backward()
                optimizer.step()
                trainloss += loss.item()
                if not np.isfinite(trainloss):
                    logger.warning('   Training loss is non-finite. Stopping...')
                    break
            validloss = 0.0
            self.model.eval()
            with torch.no_grad():
                for X,y in validloader:
                    X,y = X.to(self.device),y.to(self.device)
                    Xbatch = Xbatch.to(self.device)
                    ybatch = ybatch.to(self.device)
                    ybatchpred = self.model(Xbatch)
                    meanbatchloss = self.criterion(ybatchpred.squeeze(),ybatch.squeeze())
                    sumbatchloss += float(meanbatchloss)*Xbatch.size(0)
                    samplesseen  += Xbatch.size(0)
            validloss = sumbatchloss/samplesseen
            if np.isfinite(validloss):
                scheduler.step(validloss)
            else:
                logger.warning('   Validation loss is non-finite. Stopping...')
                break
           wandb.log({
                'Epoch':epoch+1,
                'Training loss':trainloss,
                'Validation loss':validloss,
                'Learning rate':optimizer.param_groups[0]['lr']})
        if validloss<self.bestloss:
            counter = 0
            self.bestloss  = validloss
            self.bestepoch = epoch+1
            self.beststate = {key:value.detach().cpu().clone() for key,value in self.model.state_dict().items()}
            self.save(name)
        else:
            counter += 1
            if counter>self.patience:
                logger.info(f' Early stopping at epoch {epoch+1}!')
                break
        if self.beststate is not None:
            self.model.load_state_dict(self.beststate)
            self.model.to(self.device).eval()
        trainingtime = time.time()-starttime
        wandb.run.summary.update({
            'Best model at epoch':self.bestepoch,
            'Best validation loss':self.bestloss,
            'Total training epochs':epoch+1,
            'Training duration (s)':trainingtime,
            'Stopped early':counter>self.patience})
        wandb.define_metric('Epoch', summary='none')
        for metric in ['Training loss','Validation loss','Learning rate']:
            wandb.define_metric(metric,step_metric='Epoch',summary='none')

    def predict(self,X):
        '''
        Purpose: Generate precipitation predictions using the trained model.
        Args:
        - X (torch.Tensor): input features for prediction
        Returns:
        - numpy.ndarray: predicted precipitation values
        '''
        self.model.eval()
        with torch.no_grad():
            X     = X.to(self.device)
            ypred = self.model(X)
            ypred = ypred.squeeze().cpu().numpy()
        return ypred

    def save(self,name,modeldir=MODELDIR):
        '''
        Purpose: Save the best model (lowest validation loss) to a PyTorch checkpoint, including optimizer and metadata.
        Args:
        - name (str): model name prefix
        - modeldir (str): output directory (defaults to MODELDIR)
        '''
        os.makedirs(modeldir,exist_ok=True)
        filename = f'{name}_best.pth'
        filepath = os.path.join(modeldir,filename)
        torch.save(self.bestcheckpoint,filepath)


    def save_best_model(self, run_name, modeldir=MODELDIR):
        '''
        Purpose: Save weights-only checkpoint for the current model as the best.
        Args:
        - run_name (str): file prefix for best weights
        - modeldir (str): directory to save into
        '''
        os.makedirs(modeldir,exist_ok=True)
        filename = f'{name}_best.pth'
        filepath = os.path.join(modeldir,filename)
        modelstate = {key:value.detach().cpu().clone() for key.value in self.model.state_dict().items()}
        torch.save(mos, path)
        self.best_weights_path = path

    def load_best(self, run_name, modeldir=MODELDIR):
        '''
        Purpose: Load the best (lowest validation loss) weights from disk and set model to eval().
        Args:
        - run_name (str): file prefix used during training
        - modeldir (str): directory from which to load
        '''
        path = os.path.join(modeldir, f'{run_name}_best_weights.pth')
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()


def normalize_input(trainarray,validarray,testarray,columnwise):
    '''
    Purpose: Z-score normalize input variable splits using training statistics.
    Args:
    - trainarray (numpy.ndarray): training input array
    - validarray (numpy.ndarray): validation input array
    - testarray  (numpy.ndarray): test input array
    - columnwise (bool): if True, compute mean/standard deviation per column for profile-like variables, else use scalars
    Returns:
    - tuple[numpy.ndarray,numpy.ndarray,numpy.ndarray]: normalized training/validation/test arrays
    '''
    if columnwise and trainarray.shape[1]>1:
        mean = np.nanmean(trainarray,axis=0)
        std  = np.nanstd(trainarray, axis=0)
    else:
        mean = np.nanmean(trainarray)
        std  = np.nanstd(trainarray)
    trainnorm = (trainarray-mean)/std
    validnorm = (validarray-mean)/std
    testnorm  = (testarray-mean)/std
    return trainnorm,validnorm,testnorm

def normalize_target(trainarray,validarray,testarray):
    '''
    Purpose: Apply log1p then z-score normalize precipitation data splits using training statistics.
    Args:
    - trainarray (numpy.ndarray): training precipitation values
    - validarray (numpy.ndarray): validation precipitation values
    - testarray  (numpy.ndarray): test precipitation values
    Returns:
    - tuple[np.ndarray,np.ndarray,np.ndarray,dict[str,float]]: normalized training/validation/test arrays and 
      training mean/standard deviation
    '''
    logtrain = np.log1p(trainarray)
    logvalid = np.log1p(validarray)
    logtest  = np.log1p(testarray)
    mean = np.nanmean(logtrain)
    std  = np.nanstd(logtrain)
    trainnorm = (logtrain-mean)/std
    validnorm = (logvalid-mean)/std
    testnorm  = (logtest-mean)/std
    normparams = {
        'mean':float(mean),
        'std':float(std)}
    return trainnorm,validnorm,testnorm,normparams

def denormalize_target(ynorm,normparams):
    '''
    Purpose: Inverse the log-transformation and z-score normalization applied to the precipitation data.
    Args:
    - ynorm (numpy.ndarray): normalized precipitation values
    - normparams (dict[str,float]): mean/standard deviation from the training data
    Returns:
    - numpy.ndarray: precipitation values on the original scale (mm/day)
    '''
    y = ynorm*normparams['std']+normparams['mean']
    y = np.expm1(y)
    return y

# def load(inputvars,filepath,columnwise=True):
#     '''
#     Purpose: Load data splits from an HDF5 file, normalize inputs/targets using training statistics,
#     and return PyTorch tensors.
#     Args:
#     - inputvars (list[str]): input variable names to horizontally concatenate
#     - filepath (str): path to HDF5 file produced by split.py
#     - columnwise (bool): if True, per-column input normalization, else scalar
#     Returns:
#     - tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,dict[str,float]]: input and 
#       precipitation training/validation/test tensors, alongside training statistics for target denormalization
#     '''
#     with h5py.File(filepath,'r') as f:
#         Xtrainlist = []
#         Xvalidlist = []
#         Xtestlist  = []
#         for inputvar in inputvars:
#             Xtrainarray = f[f'{inputvar}_train'][:]
#             Xvalidarray = f[f'{inputvar}_valid'][:]
#             Xtestarray  = f[f'{inputvar}_test'][:]
#             Xtrainnorm,Xvalidnorm,Xtestnorm = normalize_input(Xtrainarray,Xvalidarray,Xtestarray,columnwise)
#             Xtrainlist.append(Xtrainnorm)
#             Xvalidlist.append(Xvalidnorm)
#             Xtestlist.append(Xtestnorm)
#         Xtrain = np.concatenate(Xtrainlist,axis=1)
#         Xvalid = np.concatenate(Xvalidlist,axis=1)
#         Xtest  = np.concatenate(Xtestlist,axis=1)
#         ytrainarray = f['pr_train'][:]
#         yvalidarray = f['pr_valid'][:]
#         ytestarray  = f['pr_test'][:]
#         ytrainnorm,yvalidnorm,ytestnorm,normparams = normalize_target(ytrainarray,yvalidarray,ytestarray)
#     Xtrain = torch.tensor(Xtrain,dtype=torch.float32)
#     Xvalid = torch.tensor(Xvalid,dtype=torch.float32)
#     Xtest  = torch.tensor(Xtest,dtype=torch.float32)
#     ytrain = torch.tensor(ytrainnorm,dtype=torch.float32)
#     yvalid = torch.tensor(yvalidnorm,dtype=torch.float32)
#     ytest  = torch.tensor(ytestnorm,dtype=torch.float32)
#     return Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest,normparams

def load(inputvars,filepath,columnwise=True,usemasks=False):
    '''
    Purpose: Load data splits from an HDF5 file, normalize inputs/targets using training statistics,
    and return PyTorch tensors.
    Args:
    - inputvars (list[str]): input variable names to horizontally concatenate
    - filepath (str): path to HDF5 file produced by split.py
    - columnwise (bool): if True, per-column input normalization, else scalar
    Returns:
    - tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,dict[str,float]]: input and 
      precipitation training/validation/test tensors, alongside training statistics for target denormalization
    '''
    with h5py.File(filepath,'r') as f:
        Xtrainlist = []
        Xvalidlist = []
        Xtestlist  = []
        for inputvar in inputvars:
            Xtrainarray = f[f'{inputvar}_train'][:]
            Xvalidarray = f[f'{inputvar}_valid'][:]
            Xtestarray  = f[f'{inputvar}_test'][:]
            if usemasks:
                def _read_mask(key,fallbackshape):
                    ds = f.get(key)
                    if ds is None:
                        return np.ones(fallbackshape,dtype=np.uint8)
                    arr = ds[:]
                    return arr if arr.shape==fallbackshape else np.ones(fallbackshape,dtype=np.uint8)
                trainmask = _read_mask(f'mask_{inputvar}_train',Xtrainarray.shape)
                validmask = _read_mask(f'mask_{inputvar}_valid',Xvalidarray.shape)
                testmask  = _read_mask(f'mask_{inputvar}_test',Xtestarray.shape)
                Xtrainarray = np.where(trainmask==1,Xtrainarray,np.nan)
                Xvalidarray = np.where(validmask==1,Xvalidarray,np.nan)
                Xtestarray  = np.where(testmask==1,Xtestarray,np.nan)
            Xtrainnorm,Xvalidnorm,Xtestnorm = normalize_input(Xtrainarray,Xvalidarray,Xtestarray,columnwise)
            Xtrainnorm = np.nan_to_num(Xtrainnorm,nan=0.0,posinf=0.0,neginf=0.0)
            Xvalidnorm = np.nan_to_num(Xvalidnorm,nan=0.0,posinf=0.0,neginf=0.0)
            Xtestnorm  = np.nan_to_num(Xtestnorm,nan=0.0,posinf=0.0,neginf=0.0)
            Xtrainlist.append(Xtrainnorm)
            Xvalidlist.append(Xvalidnorm)
            Xtestlist.append(Xtestnorm)
        Xtrain = np.concatenate(Xtrainlist,axis=1)
        Xvalid = np.concatenate(Xvalidlist,axis=1)
        Xtest  = np.concatenate(Xtestlist,axis=1)
        ytrainarray = f['pr_train'][:]
        yvalidarray = f['pr_valid'][:]
        ytestarray  = f['pr_test'][:]
        ytrainnorm,yvalidnorm,ytestnorm,normparams = normalize_target(ytrainarray,yvalidarray,ytestarray)
    Xtrain = torch.tensor(Xtrain,dtype=torch.float32)
    Xvalid = torch.tensor(Xvalid,dtype=torch.float32)
    Xtest  = torch.tensor(Xtest,dtype=torch.float32)
    ytrain = torch.tensor(ytrainnorm,dtype=torch.float32)
    yvalid = torch.tensor(yvalidnorm,dtype=torch.float32)
    ytest  = torch.tensor(ytestnorm,dtype=torch.float32)
    return Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest,normparams

# def process(columnwise=False,configs=CONFIGS,filepath=FILEPATH):
def process(columnwise=False,usemasks=False,configs=CONFIGS,filepath=FILEPATH):
    '''
    Purpose: Train and evaluate NN models with multiple input variable configurations.
    Args:
    - columnwise (bool): if True, per-column input normalization for profile-like variables
    - configs (list[dict[str,object]]): model configurations specifying input variables and descriptions (defaults to CONFIGS)
    - filepath (str): path to HDF5 file produced by split.py
    Returns:
    - dict[str,dict[str,object]]: mapping from configuration name to NN results
    '''
    results = {}
    for config in configs:
        name        = config['name']
        inputvars   = config['inputvars']
        description = config['description']
        logger.info(f'   Running {description}')
        # Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest,normparams = load(inputvars,filepath,columnwise)
        Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest,normparams = load(inputvars,filepath,columnwise,usemasks)
        model = NNMODEL(Xtrain.shape[1])
        wandb.init(
            project='Unfiltered NNs',
            name=description,
            config={
                'Input variables':', '.join(f'"{inputvar}"' for inputvar in inputvars),
                ####################################
                'Mask below-surface values':usemasks,
                ####################################
                'Model parameter count':sum(p.numel() for p in model.model.parameters()),
                'Training batch size':model.batchsize,
                'Maximum training epochs':model.epochs,
                'Network architecture':'256→128→64→32→1 (GELU + BatchNorm)',
                'Optimizer':'Adam',
                'Loss function':'Mean squared error',
                'Initial learning rate':model.learningrate,
                'Learning rate scheduler':'ReduceLROnPlateau (factor=0.5)',
                'Early stopping patience':model.patience})
        model.fit(Xtrain,Xvalid,ytrain,yvalid)
        model.save(name)
        results[name] = {
            'description':description,
            'n_params':sum(p.numel() for p in model.model.parameters()),
            'y_pred':denormalize_target(model.predict(Xtest),normparams)}
        wandb.finish()
        del model,Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest,normparams
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results

def save(results,filename='nn_results.pkl',savedir=SAVEDIR):
    '''
    Purpose: Save NN model results to a pickle file in the specified directory, then verify the write by reopening.
    Args:
    - results (dict[str,dict[str,object]]): NN model results to save
    - filename (str): output file name (defaults to 'nn_results.pkl')
    - savedir (str): output directory (defaults to SAVEDIR)
    Returns:
    - bool: True if write and verification succeed, otherwise False
    '''
    try:
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
        logger.info('Training NN models...')
        results = process(columnwise=False)
        logger.info('Saving results...')
        save(results)
        del results
        logger.info('Script execution completed successfully!')
    except Exception as e:
        logger.error(f'An unexpected error occurred: {str(e)}')