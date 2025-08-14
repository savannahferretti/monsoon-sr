import os
import torch
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset,DataLoader

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

FILEDIR  = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/processed'
SAVEDIR  = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/data/results'
MODELDIR = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/models/mlp/v1'
CONFIGS  = [
    {'name':'lin_mse','nonlinear':False,'criterion':torch.nn.MSELoss(),'depth':'shallow','logtransform':False,'description':'Linear MSE Shallow'},
    {'name':'lin_mse_log','nonlinear':False,'criterion':torch.nn.MSELoss(),'depth':'shallow','logtransform':True,'description':'Log Linear MSE Shallow'},
    {'name':'lin_mae','nonlinear':False,'criterion':torch.nn.L1Loss(),'depth':'shallow','logtransform':False,'description':'Linear MAE Shallow'},
    {'name':'lin_mae_log','nonlinear':False,'criterion':torch.nn.L1Loss(),'depth':'shallow','logtransform':True,'description':'Log Linear MAE Shallow'},
    {'name':'mse_shallow','nonlinear':True,'criterion':torch.nn.MSELoss(),'depth':'shallow','logtransform':False,'description':'MSE Shallow'},
    {'name':'mse_medium','nonlinear':True,'criterion':torch.nn.MSELoss(),'depth':'medium','logtransform':False,'description':'MSE Medium'},
    {'name':'mse_deep','nonlinear':True,'criterion':torch.nn.MSELoss(),'depth':'deep','logtransform':False,'description':'MSE Deep'},
    {'name':'mse_shallow_log','nonlinear':True,'criterion':torch.nn.MSELoss(),'depth':'shallow','logtransform':True,'description':'Log MSE Shallow'},
    {'name':'mse_medium_log','nonlinear':True,'criterion':torch.nn.MSELoss(),'depth':'medium','logtransform':True,'description':'Log MSE Medium'},
    {'name':'mse_deep_log','nonlinear':True,'criterion':torch.nn.MSELoss(),'depth':'deep','logtransform':True,'description':'Log MSE Deep'},
    {'name':'mae_shallow','nonlinear':True,'criterion':torch.nn.L1Loss(),'depth':'shallow','logtransform':False,'description':'MAE Shallow'},
    {'name':'mae_medium','nonlinear':True,'criterion':torch.nn.L1Loss(),'depth':'medium','logtransform':False,'description':'MAE Medium'},
    {'name':'mae_deep','nonlinear':True,'criterion':torch.nn.L1Loss(),'depth':'deep','logtransform':False,'description':'MAE Deep'},
    {'name':'mae_shallow_log','nonlinear':True,'criterion':torch.nn.L1Loss(),'depth':'shallow','logtransform':True,'description':'Log MAE Shallow'},
    {'name':'mae_medium_log','nonlinear':True,'criterion':torch.nn.L1Loss(),'depth':'medium','logtransform':True,'description':'Log MAE Medium'},
    {'name':'mae_deep_log','nonlinear':True,'criterion':torch.nn.L1Loss(),'depth':'deep','logtransform':True,'description':'Log MAE Deep'}]

class MLPMODEL(torch.nn.Module):
    
    def __init__(self,inputsize,depth,nonlinear=False,logtransform=False):
        '''
        Purpose: Initialize a multi-layer perceptron model with specified architecture.
        Args:
        - inputsize (int): number of input features
        - depth (str): either 'shallow', 'medium', or 'deep' architecture
        - nonlinear (bool): whether the model is nonlinear and requires activation functions (defaults to False)
        - logtransform (bool): whether to target data has been log-transformed (defaults to False)
        '''
        super(MLPMODEL,self).__init__()
        self.nonlinear     = nonlinear
        self.activation    = torch.nn.ReLU() if self.nonlinear is True else torch.nn.Identity()
        self.logtransform  = logtransform
        if depth=='shallow':
            layers = torch.nn.Sequential(
                torch.nn.Linear(inputsize,64),
                self.activation,
                torch.nn.Linear(64,32),
                self.activation,
                torch.nn.Linear(32,1))
        elif depth=='medium':
            layers = torch.nn.Sequential(
                torch.nn.Linear(inputsize,128),
                self.activation,
                torch.nn.Linear(128,64),
                self.activation,
                torch.nn.Linear(64,32),
                self.activation,
                torch.nn.Linear(32,1))
        elif depth=='deep':
            layers = torch.nn.Sequential(
                torch.nn.Linear(inputsize,256),
                self.activation,
                torch.nn.Linear(256,128),
                self.activation,
                torch.nn.Linear(128,64),
                self.activation,
                torch.nn.Linear(64,32),
                self.activation,
                torch.nn.Linear(32,1))
        else:
            raise ValueError(f"Invalid depth '{depth}'; choose from 'shallow', 'medium', 'deep'")
        self.layers = torch.nn.Sequential(*layers)
        
    def forward(self,Xtensor):
        '''
        Purpose: Define the forward pass through the MLP.
        Args:
        - Xtensor (torch.Tensor): tensor of input feature(s)
        Returns:
        - torch.Tensor: output tensor after passing through all layers
        '''
        return self.layers(Xtensor)

class MLP:
    def __init__(self,inputsize,depth,criterion,nonlinear=False,logtransform=False):
        '''
        Purpose: Initialize an MLP trainer with specified hyperparameters and training configuration.
        Args:
        - inputsize (int): number of input features
        - depth (str): either 'shallow', 'medium', or 'deep' architecture
        - criterion (torch.nn.Module): loss function for training
        - nonlinear (bool): whether the model is nonlinear and requires activation functions (defaults to False)
        - logtransform (bool): whether to apply log transformation to targets (defaults to False)
        '''
        self.model        = MLPMODEL(inputsize,depth,nonlinear,logtransform)
        self.criterion    = criterion
        self.nonlinear    = nonlinear
        self.logtransform = logtransform
        self.patience     = 3
        self.nepochs      = 30
        self.batchsize    = 500
        self.validsize    = 0.2
        self.randomstate  = 42
        self.learningrate = 0.001
        self.optimizer    = torch.optim.Adam(self.model.parameters(),lr=self.learningrate)
        self.device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _tensor(self,data):
        '''
        Purpose: Convert various data types to PyTorch tensors with appropriate reshaping.
        Args:
        - data (pd.Series or pd.DataFrame): input feature data to convert
        Returns:
        - torch.FloatTensor: converted tensor with appropriate shape
        '''
        if isinstance(data,pd.Series):
            return torch.FloatTensor(data.values.reshape(-1,1))
        elif isinstance(data,pd.DataFrame):
            return torch.FloatTensor(data.values)
        else:
            raise ValueError('Unsupported data type. Please provide a pd.Series or pd.DataFrame.')

    def fit(self,X,y):
        '''
        Purpose: Train the MLP using the provided feature(s) and target data with early stopping and validation.
        Args:
        - X (pd.Series or pd.DataFrame): input feature(s) for training
        - y (pd.Series): target values for training
        Returns:
        - tuple (list, list): containing training and validation losses over epochs (if logtransform=True, these are back-transformed in original units)
        '''
        Xtensor = self._tensor(X)
        ytensor = self._tensor(y)
        Xtrain,Xvalid,ytrain,yvalid = train_test_split(Xtensor,ytensor,test_size=self.validsize,random_state=self.randomstate)
        traindataset = TensorDataset(Xtrain,ytrain)
        validdataset = TensorDataset(Xvalid,yvalid)
        trainloader  = DataLoader(traindataset,batch_size=self.batchsize,shuffle=True)
        validloader  = DataLoader(validdataset,batch_size=self.batchsize,shuffle=False)
        trainlosses  = []
        validlosses  = []
        bestvalidloss   = float('inf')
        patiencecounter = 0
        for epoch in range(self.nepochs):
            self.model.train()
            trainloss = 0.0
            for batchX,batchy in trainloader:
                batchX,batchy = batchX.to(self.device),batchy.to(self.device)
                self.optimizer.zero_grad()
                batchypred = self.model(batchX)
                # if not self.logtransform:
                #     batchypred = torch.clamp(batchypred,min=0)
                batchloss  = self.criterion(batchypred,batchy)
                batchloss.backward()
                self.optimizer.step()
                trainloss += batchloss.item()*batchX.size(0)
            trainloss /= len(trainloader.dataset)
            trainlosses.append(trainloss)
            self.model.eval()
            validloss = 0.0
            with torch.no_grad():
                for batchX,batchy in validloader:
                    batchX,batchy = batchX.to(self.device),batchy.to(self.device)
                    batchypred    = self.model(batchX)
                    # if not self.logtransform:
                    #     batchypred = torch.clamp(batchypred,min=0)
                    batchloss  = self.criterion(batchypred,batchy)
                    validloss += batchloss.item()*batchX.size(0)
            validloss /= len(validloader.dataset)
            validlosses.append(validloss)
            logger.info(f'Epoch {epoch+1}/{self.nepochs} - Training Loss: {trainloss:.4f} - Validation Loss: {validloss:.4f}')
            if validloss<bestvalidloss:
                bestvalidloss   = validloss
                patiencecounter = 0
                torch.save(self.model.state_dict(),f'{MODELDIR}/temp_best_model.pth')
            else:
                patiencecounter += 1
                if patiencecounter>=self.patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    self.model.load_state_dict(torch.load(f'{MODELDIR}/temp_best_model.pth'))
                    break
        return (trainlosses,validlosses)

    def predict(self,X):
        '''
        Purpose: Generate predictions using the trained MLP model.
        Args:
        - X (pd.Series or pd.DataFrame): input feature(s) for which to make predictions
        Returns:
        - np.ndarray: predicted target values (in original units)
        '''
        Xtensor = self._tensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            ypred = self.model(Xtensor)
            if self.logtransform:
                ypred = torch.exp(ypred)-1
            elif not self.logtransform:
                ypred = torch.clamp(ypred,min=0)
        return ypred.cpu().numpy()
        
    def evaluate(self,X,y):
        '''
        Purpose: Evaluate the trained model on provided feature(s) and target data using the specified loss criterion.
        Args:
        - X (array-like): input feature(s) for evaluation
        - y (array-like): target values for evaluation
        Returns:
        - float: loss value computed using the model's criterion
        '''
        Xtensor = self._tensor(X).to(self.device)
        ytensor = self._tensor(y).to(self.device)
        self.model.eval()
        with torch.no_grad():
            ypred = self.model(Xtensor)
            if self.logtransform:
                ypred   = torch.exp(ypred)-1
                ytensor = torch.exp(ytensor)-1
            elif not self.logtransform:
                ypred = torch.clamp(ypred,min=0)
            loss = self.criterion(ypred,ytensor).item()
        return loss

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
    
def process(Xtrain,ytrain,ytrainlog,Xtest,ytest,ytestlog,configs=CONFIGS,modeldir=MODELDIR):
    '''
    Purpose: Train MLP models with various configurations, save best models, and evaluate on test data.
    Args:
    - Xtrain (pd.Series or pd.DataFrame): training input feature(s)
    - ytrain (pd.Series or pd.DataFrame): training target values (original scale)
    - ytrainlog (pd.Series or pd.DataFrame): training target values (log-transformed)
    - Xtest (pd.Series or pd.DataFrame): test input feature(s)
    - ytest (pd.Series or pd.DataFrame): test target values (original scale)
    - ytestlog (pd.Series or pd.DataFrame):test target values (log-transformed)
    - configs (list): list of "models" to train/test and their attributes (defaults to CONFIGS)
    - modeldir (str): directory where the models should be saved (defaults to MODELDIR)
    Returns:
    - dict: dictionary containing MLP model results
    '''
    results = {}
    for config in configs:
        name         = config['name']
        depth        = config['depth']
        criterion    = config['criterion']
        nonlinear    = config['nonlinear']
        logtransform = config['logtransform']
        description  = config['description']
        inputsize    = getattr(Xtrain,'shape',[1])[-1] if hasattr(Xtrain,'shape') and len(Xtrain.shape)>1 else 1
        logger.info(f'Training {description}')
        mlp = MLP(inputsize=inputsize,depth=depth,criterion=criterion,nonlinear=nonlinear,logtransform=logtransform)
        nparams = sum(p.numel() for p in mlp.model.parameters())
        if logtransform:
            losses   = mlp.fit(Xtrain,ytrainlog)
            ypred    = mlp.predict(Xtest)
            testloss = mlp.evaluate(Xtest,ytestlog)
        else:
            losses   = mlp.fit(Xtrain,ytrain)
            ypred    = mlp.predict(Xtest)
            testloss = mlp.evaluate(Xtest,ytest)
        filename = f'best_model_{name}.pth'
        filepath = os.path.join(modeldir,filename)
        torch.save(mlp.model.state_dict(),filepath)
        results[name] = {
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
    logger.info('Training MLP models...')
    results = process(Xtrain['bl'],ytrain['pr'],ytrainlog['pr'],Xtest['bl'],ytest['pr'],ytestlog['pr'])
    logger.info('Saving results...')
    with open(f'{SAVEDIR}/v1_results.pkl','wb') as file:
        pickle.dump(results,file)
    del results
    logger.info('Training and saving complete!')