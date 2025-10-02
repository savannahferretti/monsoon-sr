#!/usr/bin/env python

import os
import json
import time
import torch
import logging
import warnings
import numpy as np
import xarray as xr
from model import NNModel
from torch.utils.data import TensorDataset,DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

with open('configs.json','r',encoding='utf8') as f:
    CONFIGS   = json.load(f)
FILEDIR      = CONFIGS['paths']['filedir']
RESULTSDIR   = CONFIGS['paths']['resultsdir']
MODELDIR     = CONFIGS['paths'].get('modeldir', os.path.join(RESULTSDIR,'models'))
INPUTVAR     = 'bl'
TARGETVAR    = 'pr'
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCHS       = 20
BATCHSIZE    = 235224
LEARNINGRATE = 5e-4
PATIENCE     = 3
SEED         = 1337
CRITERION    = torch.nn.MSELoss()

np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == 'cuda':
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def reshape(da):
    if 'lev' in da.dims:
        arr = da.transpose('time','lat','lon','lev').values.reshape(-1,da.lev.size)
    else:
        arr = da.transpose('time','lat','lon').values.reshape(-1,1)
    return arr

def load(splitname,inputvar=INPUTVAR,targetvar=TARGETVAR,filedir=FILEDIR):
    if splitname not in ('norm_train','norm_valid'):
        raise ValueError("splitname must be 'norm_train' or 'norm_valid'")
    filepath = os.path.join(filedir,f'{splitname}.h5')
    ds = xr.open_dataset(filepath,engine='h5netcdf')[[inputvar,targetvar]]
    X  = torch.tensor(reshape(ds[inputvar]),dtype=torch.float32)
    y  = torch.tensor(reshape(ds[targetvar]),dtype=torch.float32)
    ytemplate = ds[targetvar]
    return X,y,ytemplate

def load_target_stats(filedir=FILEDIR,targetvar=TARGETVAR):
    with open(os.path.join(filedir,'stats.json'),'r',encoding='utf-8') as f:
        stats = json.load(f)
    mean = float(stats[f'{targetvar}_mean'])
    std  = float(stats[f'{targetvar}_std'])
    return mean,std

def denormalize(ynorm,mean,std):
    def _to_torch(x):
        if isinstance(x,torch.Tensor):
            return x
        return torch.as_tensor(x,dtype=torch.float32)
    y = _to_torch(ynorm)*std+mean
    y = torch.expm1(y)
    y = torch.clamp_min(y,0.0)  
    return y

def fit_mean_model(ytrain):
    ytrain = ytrain.squeeze(-1)
    ytrainmean = ytrain.mean().item()
    return float(ytrainmean)

def predict_mean_model(ytrainmean,nsamples,mean,std):
    ypred = np.full((nsamples,),ytrainmean,dtype=np.float32)
    ypred = denormalize(ypred,mean,std)
    return ypred.cpu().numpy()
    
def fit_linear_regression_model(Xtrain,ytrain,chunksize=BATCHSIZE):
    assert Xtrain.ndim==2 and Xtrain.shape[1]==1
    assert ytrain.shape==Xtrain.shape
    nsamples = Xtrain.shape[0]
    xsum = 0.0
    ysum = 0.0
    for start in range(0,nsamples,chunksize):
        stop   = start+chunksize
        xchunk = Xtrain[start:stop,0].cpu().numpy()
        ychunk = ytrain[start:stop,0].cpu().numpy()
        xsum += xchunk.sum(dtype=np.float64)
        ysum += ychunk.sum(dtype=np.float64)
    xmean = xsum/nsamples
    ymean = ysum/nsamples
    ssx = 0.0
    sxy = 0.0
    for start in range(0,nsamples,chunksize):
        stop   = start+chunksize
        xchunk = Xtrain[start:stop,0].cpu().numpy().astype(np.float64,copy=False)
        ychunk = ytrain[start:stop,0].cpu().numpy().astype(np.float64,copy=False)
        dx = xchunk-xmean
        dy = ychunk-ymean
        ssx += np.dot(dx,dx)
        sxy += np.dot(dx,dy)
    if ssx<=0.0:
        return 0.0,float(ymean)
    slope     = float(sxy/ssx)
    intercept = float(ymean-slope*xmean)
    return slope,intercept

def predict_linear_regression_model(slope,intercept,X,mean,std):
    x = X[:,0].cpu().numpy().astype(np.float64,copy=False)
    ypred = (slope*x+intercept).astype(np.float32)
    ypred = denormalize(ypred,mean,std)
    return ypred.cpu().numpy()

def save(modelstate,runname,modeldir=MODELDIR):
    os.makedirs(modeldir,exist_ok=True)
    filepath = os.path.join(modeldir,f'nn_{runname}.pth')
    logger.info(f'   Attempting to save nn_{runname}.pth...')
    try:
        torch.save(modelstate,filepath)
        _ = torch.load(filepath,map_location='cpu')
        logger.info('      File write successful')
        return True
    except Exception:
        logger.exception('      Failed to save or verify')
        return False

def fit_nn_model(model,Xtrain,ytrain,Xvalid,yvalid,mean,std,
                 batchsize=BATCHSIZE,device=DEVICE,learningrate=LEARNINGRATE,patience=PATIENCE,criterion=CRITERION,epochs=EPOCHS):
    trainloader = DataLoader(TensorDataset(Xtrain,ytrain),batch_size=batchsize,shuffle=True,num_workers=8,persistent_workers=True,pin_memory=True)
    validloader = DataLoader(TensorDataset(Xvalid,yvalid),batch_size=batchsize,shuffle=False,num_workers=8,persistent_workers=True,pin_memory=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=learningrate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=patience)
    bestloss  = float('inf')
    noimprove = 0
    for epoch in range(1,epochs+1):
        model.train()
        runningloss = 0.0
        for Xbatch,ybatch in trainloader:
            Xbatch,ybatch = Xbatch.to(device,non_blocking=True),ybatch.to(device,non_blocking=True)
            optimizer.zero_grad()
            ybatchpred = model(Xbatch)
            ybatchpredphys = denormalize(ybatchpred.squeeze(-1),mean,std)
            ybatchphys     = denormalize(ybatch.squeeze(-1),mean,std)
            loss = criterion(ybatchpredphys,ybatchphys)
            loss.backward()
            optimizer.step()
            runningloss += loss.item()*Xbatch.size(0)
        trainloss = runningloss/len(trainloader.dataset)
        model.eval()
        runningloss = 0.0
        with torch.no_grad():
            for Xbatch,ybatch in validloader:
                Xbatch,ybatch = Xbatch.to(device,non_blocking=True),ybatch.to(device,non_blocking=True)
                ybatchpred = model(Xbatch)
                ybatchpredphys = denormalize(ybatchpred.squeeze(-1),mean,std)
                ybatchphys     = denormalize(ybatch.squeeze(-1),mean,std)
                loss = criterion(ybatchpredphys,ybatchphys)
                runningloss += loss.item()*Xbatch.size(0)
        validloss = runningloss/len(validloader.dataset)
        scheduler.step(validloss)
        improved = validloss<bestloss
        if improved:
            bestloss  = validloss
            noimprove = 0
            save(model.state_dict(),'debug_mse')
        else:
            noimprove += 1
        if noimprove>=patience:
            break
    return model

def predict_nn_model(model,X,mean,std,batchsize=BATCHSIZE,device=DEVICE):
    evaldataset = TensorDataset(X)
    evalloader  = DataLoader(evaldataset,batch_size=batchsize,shuffle=False,num_workers=8,persistent_workers=True,pin_memory=True)
    ypredlist   = []
    model.eval()
    with torch.no_grad():
        for (Xbatch,) in evalloader:
            Xbatch = Xbatch.to(device,non_blocking=True)
            ybatchpred = model(Xbatch)
            ybatchpredphys = denormalize(ybatchpred.squeeze(-1),mean,std)
            ypredlist.append(ybatchpredphys.cpu().numpy())
    ypred = np.concatenate(ypredlist,axis=0)
    return ypred

if __name__=='__main__':
    try:
        logger.info('Loading normalized inputs + normalized targets...')
        Xtrain,ytrain,_         = load('norm_train')
        Xvalid,yvalid,ytemplate = load('norm_valid')
        mean,std = load_target_stats()
        logger.info('Running mean model...')
        ytrainmean = fit_mean_model(ytrain)
        ypredmean  = predict_mean_model(ytrainmean,yvalid.numel(),mean,std)
        logger.info('Running linear regression model...')
        slope,intercept = fit_linear_regression_model(Xtrain,ytrain)
        ypredlinreg     = predict_linear_regression_model(slope,intercept,Xvalid,mean,std) 
        logger.info('Running untrained NN...')
        nnuntrained    = NNModel(inputsize=Xtrain.shape[1]).to(DEVICE)
        ypreduntrained = predict_nn_model(nnuntrained,Xvalid,mean,std)
        logger.info('Running trained NN...')
        nntrained    = fit_nn_model(nnuntrained,Xtrain,ytrain,Xvalid,yvalid,mean,std)
        ypredtrained = predict_nn_model(nntrained,Xvalid,mean,std)
        logger.info('Saving validation predictions NetCDF...')
        os.makedirs(RESULTSDIR,exist_ok=True)
        outpath = os.path.join(RESULTSDIR,'debug_mse_valid_pr.nc')
        def reshape_to_template(arr,template,name):
            return xr.DataArray(arr.reshape(template.shape),dims=template.dims,coords=template.coords,name=name)
        dsout = xr.Dataset({
            'predpr_mean':reshape_to_template(ypredmean,ytemplate,'predpr_mean'),
            'predpr_linear':reshape_to_template(ypredlinreg,ytemplate,'predpr_linear'),
            'predpr_nn_untrained':reshape_to_template(ypreduntrained,ytemplate,'predpr_nn_untrained'),
            'predpr_nn_trained':reshape_to_template(ypredtrained,ytemplate,'predpr_nn_trained')})
        dsout.to_netcdf(outpath,engine='h5netcdf')
        with xr.open_dataset(outpath,engine='h5netcdf'): 
            pass
        logger.info(f'Wrote {outpath}')
    except Exception as e:
        logger.exception(f'An unexpected error occurred: {e}')