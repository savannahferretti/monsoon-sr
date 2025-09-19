#!/usr/bin/env python

import os
import json
import time
import torch
import logging
import warnings
import numpy as np
import xarray as xr

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("debug_exp1")
warnings.filterwarnings('ignore')

with open('configs.json','r',encoding='utf8') as f:
    CONFIGS = json.load(f)
FILEDIR     = CONFIGS['paths']['filedir']
RESULTSDIR  = CONFIGS['paths']['resultsdir']
INPUTVAR    = 'bl' 
TARGETVAR   = 'pr'
DESCRIPTION = 'Experiment 1'         
DEVICE      =  'cuda' if torch.cuda.is_available() else 'cpu'

if DEVICE=='cuda':
    SEED = 1337
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def reshape(da):
    if 'lev' in da.dims:
        arr = da.transpose('time','lat','lon','lev').values.reshape(-1,da.lev.size)
    else:
        arr = da.transpose('time','lat','lon').values.reshape(-1,1)
    return arr

def r2_score(ytrue,ypred):
    ytrue = ytrue.ravel()
    ypred = ypred.ravel()
    rss = np.sum((ytrue-ypred)**2)
    tss = np.sum((ytrue-np.mean(ytrue))**2)
    return (1.0-(rss/tss)) if tss>0 else np.nan

def denormalize(ynormflat,targetvar=TARGETVAR,filedir=FILEDIR):
    with open(os.path.join(filedir,'stats.json'),'r',encoding='utf-8') as f:
        stats = json.load(f)
    mean  = float(stats[f'{targetvar}_mean'])
    std   = float(stats[f'{targetvar}_std'])
    yflat = np.expm1((ynormflat*std+mean)) 
    return yflat

def load(splitname,inputvar=INPUTVAR,targetvar=TARGETVAR,filedir=FILEDIR):
    if splitname not in ('norm_train','norm_valid'):
        raise ValueError('Split name must be norm_train or norm_valid')
    filename = f'{splitname}.h5'
    filepath = os.path.join(filedir,filename)
    varlist  = [inputvar,targetvar]
    ds = xr.open_dataset(filepath,engine='h5netcdf')[varlist]
    if splitname=='norm_train':
        ds = ds.sel(time=slice('2011-06-01','2014-08-31'))
    else:
        ds = ds.sel(time=slice('2015-06-01','2015-08-31'))
    X = torch.tensor(reshape(ds[inputvar]),dtype=torch.float32)
    y = torch.tensor(reshape(ds[targetvar]),dtype=torch.float32)
    ytemplate = ds[targetvar]
    return X,y,ytemplate

def to_xarray(yflat,template,name):
    da = xr.DataArray(yflat.reshape(template.shape),dims=template.dims,coords=template.coords,name=name)
    da.attrs = dict(long_name=f'{name} precipitation',units='mm/day')
    return da

def fit_mean_model(ytrainflat):
    mu = float(np.mean(ytrainflat))
    return mu
    
def predict_mean_model(mu,n):
    return np.full((n,),mu,dtype=np.float32)

def fit_linear_regression_model(Xtrain,ytrain) :
    Xb = np.hstack([Xtrain,np.ones((Xtrain.shape[0],1),dtype=Xtrain.dtype)])
    beta,*_ = np.linalg.lstsq(Xb,ytrain,rcond=None)
    return beta

def predict_linear_regression_model(beta,X):
    Xb = np.hstack([X,np.ones((X.shape[0],1),dtype=X.dtype)])
    return Xb@ beta

class NNModel(torch.nn.Module):
    def __init__(self,inputsize):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(inputsize,64),
            torch.nn.GELU(),
            torch.nn.Linear(64,32),
            torch.nn.GELU(),
            torch.nn.Linear(32,1))
    def forward(self,X):
        return self.layers(X)

def fit_nn_model(Xtrain,ytrain,epochs=10,batchsize=92,learningrate=1e-3):
    model = NNModel(Xtrain.shape[1]).to(DEVICE)
    traindataset = torch.utils.data.TensorDataset(Xtrain,ytrain)
    trainloader  = torch.utils.data.DataLoader(traindataset,batch_size=batchsize,shuffle=True,num_workers=2,pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(),lr=learningrate)
    criterion = torch.nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xbatch,ybatch in trainloader:
            xbatch = xbatch.to(DEVICE,non_blocking=True)
            ybatch = ybatch.to(DEVICE,non_blocking=True)
            optimizer.zero_grad()
            ybatchpred = model(xbatch)
            loss = criterion(ybatchpred.squeeze(-1),ybatch.squeeze(-1))
            loss.backward()
            optimizer.step()
    return model

def predict_nn_model(model,X,batchsize=92):
    evaldataset = torch.utils.data.TensorDataset(X)
    evalloader  = torch.utils.data.DataLoader(evaldataset,batch_size=batchsize,shuffle=False,num_workers=2,pin_memory=True)
    ypredlist   = []
    model.eval()
    with torch.no_grad():
        for (xbatch,) in evalloader:
            xbatch     = xbatch.to(DEVICE,non_blocking=True)
            ybatchpred = model(xbatch).squeeze(-1).detach().cpu().numpy()
            ypredlist.append(ybatchpred)
    return np.concatenate(ypredlist,axis=0)

if __name__=='__main__':
    try:
        start = time.time()
        logger.info(f'Device: {DEVICE}')
        
        logger.info('Loading training/validation splits for Experiment 1...')
        Xtrainnorm,ytrainnorm,ytraintemplate = load('norm_train')
        Xvalidnorm,yvalidnorm,yvalidtemplate = load('norm_valid')
        ytrainnormflat = ytrainnorm.squeeze(-1).numpy().ravel().astype(np.float32)
        yvalidnormflat = yvalidnorm.squeeze(-1).numpy().ravel().astype(np.float32)
        ytrainflat = denormalize(ytrainnormflat)
        yvalidflat = denormalize(yvalidnormflat)

        logger.info('Running mean model...')
        munorm = fit_mean_model(ytrainnormflat)
        ytrainnormflat1 = predict_mean_model(munorm,n=ytrainnormflat.size)
        yvalidnormflat1 = predict_mean_model(munorm,n=yvalidnormflat.size)
        ytrainflat1 = denormalize(ytrainnormflat1)
        yvalidflat1 = denormalize(yvalidnormflat1)
        r2train1 = r2_score(ytrainflat,ytrainflat1)
        r2valid1 = r2_score(yvalidflat,yvalidflat1)
        logger.info(f'   R² (train): {r2train1:.4f} | R² (valid): {r2valid1:.4f}')

        logger.info('Running linear regression model...')
        beta = fit_linear_regression_model(Xtrainnorm.numpy(),ytrainnormflat[:,None])  
        beta = beta.squeeze(-1)
        ytrainnormflat2 = predict_linear_regression_model(beta,Xtrainnorm.numpy())
        yvalidnormflat2 = predict_linear_regression_model(beta,Xvalidnorm.numpy())
        ytrainflat2 = denormalize(ytrainnormflat2)
        yvalidflat2 = denormalize(yvalidnormflat2)
        r2train2 = r2_score(ytrainflat,ytrainflat2)
        r2valid2 = r2_score(yvalidflat,yvalidflat2)
        logger.info(f'   R² (train): {r2train2:.4f} | R² (valid): {r2valid2:.4f}')

        logger.info('Running untrained NN...')
        nnuntrained = NNModel(inputsize=Xtrainnorm.shape[1]).to(DEVICE)
        ytrainnormflat3 = predict_nn_model(nnuntrained,Xtrainnorm)
        yvalidnormflat3 = predict_nn_model(nnuntrained,Xvalidnorm)
        ytrainflat3 = denormalize(ytrainnormflat3)
        yvalidflat3 = denormalize(yvalidnormflat3)
        r2train3 = r2_score(ytrainflat,ytrainflat3)
        r2valid3 = r2_score(yvalidflat,yvalidflat3)
        logger.info(f'   R² (train): {r2train3:.4f} | R² (valid): {r2valid3:.4f}')
    
        logger.info('Running trained NN...')
        nntrained = fit_nn_model(Xtrainnorm,ytrainnorm)
        ytrainnormflat4 = predict_nn_model(nntrained,Xtrainnorm)
        yvalidnormflat4 = predict_nn_model(nntrained,Xvalidnorm)
        ytrainflat4 = denormalize(ytrainnormflat4)
        yvalidflat4 = denormalize(yvalidnormflat4)
        r2train4 = r2_score(ytrainflat,ytrainflat4)
        r2valid4 = r2_score(yvalidflat,yvalidflat4)
        logger.info(f'   R² (train): {r2train4:.4f} | R² (valid): {r2valid4:.4f}')
    
        logger.info('Saving validation set predictions to a single NetCDF...')
        os.makedirs(RESULTSDIR,exist_ok=True)
        outpath = os.path.join(RESULTSDIR,'debug_exp1_norm_valid_pr.nc')
        da1 = to_xarray(yvalidflat1,yvalidtemplate,name='predpr_mean')
        da2 = to_xarray(yvalidflat2,yvalidtemplate,name='predpr_linear')
        da3 = to_xarray(yvalidflat3,yvalidtemplate,name='predpr_nn_untrained')
        da4 = to_xarray(yvalidflat4,yvalidtemplate,name='predpr_nn_trained')
        dsout = xr.Dataset({da1.name:da1,da2.name:da2,da3.name:da3,da4.name:da4})
        dsout.to_netcdf(outpath,engine='h5netcdf')
        with xr.open_dataset(outpath,engine='h5netcdf') as _:
            pass
        logger.info(f'Wrote {outpath}')
        logger.info(f'Done in {time.time()-start:.1f}s!')
    except Exception as e:
        logger.exception(f'An unexpected error occurred: {e}')
        raise