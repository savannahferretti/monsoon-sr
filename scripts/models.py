import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset,DataLoader

MODELDIR = '/global/cfs/cdirs/m4334/sferrett/monsoon-sr/models'

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

class MLPMODEL(torch.nn.Module):
    
    def __init__(self,inputlayersize,hiddenlayersize,nhiddenlayers,activation=None,logtransform=False):
        '''
        Purpose: Initialize a multi-layer perceptron model with specified architecture.
        Args:
        - inputlayersize (int): number of input features
        - hiddenlayersize (int): number of neurons in each hidden layer
        - nhiddenlayers (int): number of hidden layers
        - activation (torch.nn.Module): activation function to use (defaults to Identity if None)
        - logtransform (bool): whether to apply log transformation to targets (defaults to False)
        '''
        super(MLPMODEL,self).__init__()
        self.inputlayersize  = inputlayersize
        self.hiddenlayersize = hiddenlayersize
        self.nhiddenlayers   = nhiddenlayers
        self.activation      = activation if activation is not None else torch.nn.Identity()
        self.logtransform    = logtransform
        layers = []
        layers.append(torch.nn.Linear(inputlayersize,hiddenlayersize))
        if not isinstance(self.activation,torch.nn.Identity):
            layers.append(self.activation)        
        for _ in range(nhiddenlayers-1):
            layers.append(torch.nn.Linear(hiddenlayersize,hiddenlayersize))
            if not isinstance(self.activation,torch.nn.Identity):
                layers.append(self.activation)    
        layers.append(torch.nn.Linear(hiddenlayersize,1))
        if not self.logtransform and not isinstance(self.activation,torch.nn.Identity):
            layers.append(self.activation)
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
    def __init__(self,inputlayersize,hiddenlayersize,nhiddenlayers,activation,criterion, 
                 learningrate,nepochs,batchsize,validsize,patience,randomstate,logtransform=False):
        '''
        Purpose: Initialize an MLP trainer with specified hyperparameters and training configuration.
        Args:
        - inputlayersize (int): number of input features
        - hiddenlayersize (int): number of neurons in each hidden layer
        - nhiddenlayers (int): number of hidden layers
        - activation (torch.nn.Module): activation function to use
        - criterion (torch.nn.Module): loss function for training
        - learningrate (float): learning rate for optimizer
        - nepochs (int): maximum number of training epochs
        - batchsize (int): batch size for training
        - validsize (float): fraction of training data to use for validation
        - patience (int): number of epochs to wait before early stopping
        - randomstate (int): random seed for reproducibility
        - logtransform (bool): whether to apply log transformation to targets (defaults to False)
        '''
        self.model        = MLPMODEL(inputlayersize,hiddenlayersize,nhiddenlayers,activation,logtransform)
        self.activation   = activation
        self.criterion    = criterion
        self.optimizer    = torch.optim.Adam(self.model.parameters(),lr=learningrate)
        self.nepochs      = nepochs
        self.batchsize    = batchsize
        self.validsize    = validsize
        self.patience     = patience
        self.randomstate  = randomstate
        self.logtransform = logtransform
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
                batchypred    = self.model(batchX)
                if self.logtransform:
                    batchypred = torch.clamp((torch.exp(batchypred)-1),min=0)
                    batchy     = torch.exp(batchy)-1
                    batchloss  = self.criterion(batchypred,batchy)
                else:
                    if isinstance(self.activation,torch.nn.Identity):
                        batchypred = torch.clamp(batchypred,min=0)
                    batchloss = self.criterion(batchypred,batchy)
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
                    if self.logtransform:
                        batchypred = torch.clamp((torch.exp(batchypred)-1),min=0)
                        batchy     = torch.exp(batchy)-1
                        batchloss  = self.criterion(batchypred,batchy)
                    else:
                        if isinstance(self.activation,torch.nn.Identity):
                            batchypred = torch.clamp(batchypred,min=0)
                        batchloss = self.criterion(batchypred,batchy)
                    validloss += batchloss.item()*batchX.size(0)
            validloss /= len(validloader.dataset)
            validlosses.append(validloss)
            print(f'Epoch {epoch+1}/{self.nepochs} - Training Loss: {trainloss:.4f} - Validation Loss: {validloss:.4f}')
            if validloss<bestvalidloss:
                bestvalidloss   = validloss
                patiencecounter = 0
                torch.save(self.model.state_dict(),f'{MODELDIR}/mlp/temp_best_model.pth')
            else:
                patiencecounter += 1
                if patiencecounter>=self.patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    self.model.load_state_dict(torch.load(f'{MODELDIR}/mlp/temp_best_model.pth'))
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
                ypred = torch.clamp((torch.exp(ypred)-1),min=0)
            else:
                if isinstance(self.activation,torch.nn.Identity):
                    ypred = torch.clamp(ypred,min=0)
                else:
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
                ypred   = torch.clamp((torch.exp(ypred)-1),min=0)
                ytensor = torch.exp(ytensor)-1
                loss    = self.criterion(ypred,ytensor).item()
            else:
                if isinstance(self.activation,torch.nn.Identity):
                    ypred = torch.clamp(ypred,min=0)
                loss = self.criterion(ypred,ytensor).item()
        return loss




class MLPMODELV2(torch.nn.Module):
    
    def __init__(self,inputlayersize,depth):
        super(MLPMODELV2,self).__init__()
        if depth=='shallow':
            # 2 hidden layers: input → 128 → 64 → 1
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(inputlayersize,128),
                torch.nn.BatchNorm1d(128),
                torch.nn.GELU(),
                torch.nn.Linear(128,64),
                torch.nn.BatchNorm1d(64),
                torch.nn.GELU(),
                torch.nn.Linear(64,1),
                torch.nn.ReLU())
        elif depth=='medium':
            # 3 hidden layers: input → 256 → 128 → 64 → 1
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(inputlayersize,256),
                torch.nn.BatchNorm1d(256),
                torch.nn.GELU(),
                torch.nn.Linear(256,128),
                torch.nn.BatchNorm1d(128),
                torch.nn.GELU(),
                torch.nn.Linear(128,64),
                torch.nn.BatchNorm1d(64),
                torch.nn.GELU(),
                torch.nn.Linear(64,1),
                torch.nn.ReLU())
        elif depth=='deep':
            # 4 hidden layers: input → 256 → 128 → 64 → 64 → 1
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(inputlayersize,256),
                torch.nn.BatchNorm1d(256),
                torch.nn.GELU(),
                torch.nn.Linear(256,128),
                torch.nn.BatchNorm1d(128),
                torch.nn.GELU(),
                torch.nn.Linear(128,64),
                torch.nn.BatchNorm1d(64),
                torch.nn.GELU(),
                torch.nn.Linear(64,64),
                torch.nn.BatchNorm1d(64),
                torch.nn.GELU(),
                torch.nn.Linear(64,1),
                torch.nn.ReLU())
        else:
            raise ValueError(f"Invalid depth '{depth}'. Choose from 'shallow', 'medium', 'deep'")

    def forward(self,Xtensor):
        return self.layers(Xtensor)

class MLPV2:
    def __init__(self,inputlayersize,depth,criterion):
        self.model = MLPMODELV2(inputlayersize,depth)
        self.criterion    = criterion
        self.optimizer    = torch.optim.Adam(self.model.parameters(),lr=.001)
        self.nepochs      = 30
        self.batchsize    = 500
        self.validsize    = 0.2
        self.patience     = 5
        self.randomstate  = 42
        self.device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _tensor(self,data):
        if isinstance(data,pd.Series):
            return torch.FloatTensor(data.values.reshape(-1,1))
        elif isinstance(data,pd.DataFrame):
            return torch.FloatTensor(data.values)
        else:
            raise ValueError('Unsupported data type. Please provide a pd.Series or pd.DataFrame.')

    def fit(self,X,y):
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
                    batchloss     = self.criterion(batchypred,batchy)
                    validloss    += batchloss.item()*batchX.size(0)
            validloss /= len(validloader.dataset)
            validlosses.append(validloss)
            print(f'Epoch {epoch+1}/{self.nepochs} - Training Loss: {trainloss:.3f} - Validation Loss: {validloss:.3f}')
            if validloss<bestvalidloss:
                bestvalidloss   = validloss
                patiencecounter = 0
                torch.save(self.model.state_dict(),f'{MODELDIR}/mlp/temp_best_model.pth')
            else:
                patiencecounter += 1
                if patiencecounter>=self.patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    self.model.load_state_dict(torch.load(f'{MODELDIR}/mlp/temp_best_model.pth'))
                    break
        return (trainlosses,validlosses)

    def predict(self,X):
        Xtensor = self._tensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            ypred = self.model(Xtensor)
        return ypred.cpu().numpy()
        
    def evaluate(self,X,y):
        Xtensor = self._tensor(X).to(self.device)
        ytensor = self._tensor(y).to(self.device)
        self.model.eval()
        with torch.no_grad():
            ypred = self.model(Xtensor)
            loss  = self.criterion(ypred,ytensor).item()
        return loss