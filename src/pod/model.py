#!/usr/bin/env python

import numpy as np

class PODModel:
    
    def __init__(self,mode,landthresh=0.5):
        '''
        Purpose: Initialize a ramp-based POD model for precipitation prediction using Eq. 8 from Ahmed F., Adames Á.F., & 
        Neelin J.D. (2020), J. Atmos. Sci.
        Args: 
        - mode (str): 'pooled' (single ramp) | 'regional' (separate land/ocean ramps)
        - landthresh (float): routing threshold for 'regional' mode, where land occurs ≥ 'landthresh' (defaults to 0.5)
        Returns:
        - None
        '''
        self.mode         = str(mode)
        self.landthresh   = float(landthresh)
        self.alphapooled  = np.nan
        self.blcritpooled = np.nan
        self.alphaland    = np.nan
        self.blcritland   = np.nan
        self.alphaocean   = np.nan
        self.blcritocean  = np.nan
        self.nparams      = 0

    def forward(self,x,lf=None):
        '''
        Purpose: Forward pass through the POD ramp. 
        Args:
        - x (xr.DataArray): input 3D BL DataArray
        - lf (xr.DataArray): land fraction for routing in 'regional' mode (same shape as 'x')
        Returns:
        - np.ndarray: predicted prediction array of shape (x.size,)
        '''
        xflat  = x.values.ravel()
        ypred  = np.full(xflat.shape,np.nan,dtype=np.float32)
        finite = np.isfinite(xflat)
        if self.mode=='pooled':
            ypred[finite] = self.alphapooled*np.maximum(0.0,xflat[finite]-self.blcritpooled).astype(np.float32)
        elif self.mode=='regional':
            if lf is None:
                raise ValueError('Regional mode requires `lf` for routing.')
            landmask      = (lf.values.ravel()[finite]>=self.landthresh)
            ypredland     = self.alphaland*np.maximum(0.0,xflat[finite]-self.blcritland)
            ypredocean    = self.alphaocean*np.maximum(0.0,xflat[finite]-self.blcritocean)
            ypred[finite] = np.where(landmask,ypredland,ypredocean).astype(np.float32)
        else:
            raise ValueError('The mode must be `pooled` or `regional`.')
        return ypred