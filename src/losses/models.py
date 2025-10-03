#!/usr/bin/env python

import torch
import numpy as np

class PODModel:
    def __init__(self,binwidth,binmin=-0.6,binmax=0.1,samplethresh=50):
        self.binwidth     = float(binwidth)
        self.binmin       = float(binmin)
        self.binmax       = float(binmax)
        self.binedges     = np.arange(self.binmin,self.binmax+self.binwidth,self.binwidth,dtype=np.float32)
        self.bincenters   = ((self.binedges[:-1]+self.binedges[1:])*0.5).astype(np.float32)
        self.nbins        = int(self.bincenters.size)
        self.samplethresh = int(samplethresh)
        self.binmeans     = None 
        self.nparams      = 0   
    def forward(self,X):
        if self.binmeans is None:
            raise RuntimeError('Parameters not set; train or load a POD model first.')
        Xflat   = X.values.ravel()
        binidxs = np.digitize(Xflat,self.binedges)-1
        binidxs = np.clip(binidxs,0,self.nbins-1)
        ypred   = self.binmeans[binidxs].astype(np.float32,copy=False)
        nonfinite = ~np.isfinite(Xflat)
        if np.any(nonfinite):
            ypred[nonfinite] = np.nan
        return ypred

class NNModel(torch.nn.Module):
    def __init__(self,inputsize):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(inputsize,256), torch.nn.GELU(),
            torch.nn.Linear(256,128),       torch.nn.GELU(),
            torch.nn.Linear(128,64),        torch.nn.GELU(),
            torch.nn.Linear(64,32),         torch.nn.GELU(),
            torch.nn.Linear(32,1),          torch.nn.Softplus())
    def forward(self,X):
        return self.layers(X)

class TweedieDevianceLoss(torch.nn.Module):
    def __init__(self,p=1.5,epsilon=1e-6):
        super().__init__()
        if not (1.0<float(p)<2.0):
            raise ValueError('TweedieDevianceLoss requires 1 < p < 2.')
        self.p       = float(p)
        self.epsilon = float(epsilon)
    def forward(self,ypred,ytrue):
        ypred = torch.clamp(ypred,min=self.epsilon)
        ytrue = ytrue.to(ypred.dtype)
        terma = torch.clamp(ytrue,min=0.0).pow(2.0-self.p)/((1.0-self.p)*(2.0-self.p))
        termb = (ytrue*ypred.pow(1.0-self.p))/(1.0-self.p)
        termc = ypred.pow(2.0-self.p)/(2.0-self.p)
        deviance = 2.0*(terma-termb+termc)
        return deviance.mean()