#!/usr/bin/env python

import numpy as np

class PODModel:
    
    def __init__(self,binwidth,binmin=-0.6,binmax=0.1,samplethresh=50):
        '''
        Purpose: Initialize a buoyancy-based POD model for precipitation prediction.
        Args:
        - binwidth (float): width of each BL bin (m/s²)
        - binmin (float): minimum boundary for the binning range (defaults to -0.6 m/s²)
        - binmax (float): maximum boundary for the binning range (defaults to 0.1 m/s²)
        - samplethresh (int): number of samples in a bin required to compute the bin average, otherwise NaN (defaults to 50)
        Returns:
        - None
        '''
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
        '''
        Purpose: Forward pass through the POD model.
        Args:
        - X (xr.DataArray): input 3D BL DataArray
        Returns:
        - np.ndarray: predicted prediction array of shape (X.size,)
        '''
        if self.binmeans is None:
            raise RuntimeError('Parameters not set; train or load a POD model first.')
        Xflat   = X.values.ravel()
        binidxs = np.digitize(Xflat,self.binedges)-1
        ypred   = np.full(Xflat.shape,np.nan,dtype=np.float32)
        valid   = (binidxs>=0)&(binidxs<self.nbins)&np.isfinite(Xflat)
        ypred[valid] = self.binmeans[binidxs[valid]]
        return ypred