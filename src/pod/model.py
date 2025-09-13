#!/usr/bin/env python

import numpy as np
import xarray as xr

class PODModel:
    
    def __init__(self,binwidth,binmin=-0.6,binmax=0.1,samplethresh=50):
        '''
        Purpose: Initialize a buoyancy-based POD model for precipitation prediction.
        Args:
        - binwidth (float): width of each BL bin (m/s²)
        - binmin (float): minimum boundary for the binning range (defaults to -0.6 m/s²)
        - binmax (float): maximum boundary for the binning range (defaults to 0.1 m/s²)
        - samplethresh (int): minimum number of samples required per bin to compute the bin average (defaults to 50)
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

    def fit(self,X,y):
        '''
        Purpose: Train a POD model by computing average precipitation in each BL bin.
        Args:
        - X (xr.DataArray): 3D DataArray of BL values for training
        - y (xr.DataArray): 3D DataArray of precipitation values for training
        Returns:
        - None (updates self.binmeans)
        '''
        Xflat  = X.values.flatten()
        yflat  = y.values.flatten()
        idx    = np.digitize(Xflat,self.binedges)-1
        mask   = (idx>=0)&(idx<self.nbins)&np.isfinite(yflat)
        counts = np.bincount(idx[mask],minlength=self.nbins).astype(np.int32)
        sums   = np.bincount(idx[mask],weights=yflat[mask],minlength=self.nbins).astype(np.float32)
        with np.errstate(divide='ignore',invalid='ignore'):
            means = sums/counts
        means[counts<self.samplethresh] = np.nan
        self.binmeans = means.astype(np.float32)
        self.nparams  = int(np.isfinite(self.binmeans).sum())

    def predict(self,X):
        '''
        Purpose: Generate precipitation predictions using the bin averages.
        Args:
        - X (xr.DataArray): input 3D DataArray of BL values
        Returns:
        - xr.DataArray: 3D DataArray of predicted precipitation
        '''
        if self.binmeans is None:
            raise RuntimeError('POD model is not fit. Call fit() or load a saved model first.')
        Xflat     = X.values.flatten()
        binidxs   = np.clip(np.digitize(Xflat,self.binedges)-1,0,self.nbins-1)
        ypredflat = self.binmeans[binidxs]
        ypred = xr.DataArray(ypredflat.reshape(X.shape),dims=X.dims,coords=X.coords,name='predpr')
        ypred.attrs = dict(long_name='Predicted precipitation',units='mm/day')
        return ypred