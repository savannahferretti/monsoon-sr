#!/usr/bin/env python

import torch

class NNModel(torch.nn.Module):

    def __init__(self,inputsize):
        '''
        Purpose: Define a feedforward neural network (NN) for precipitation prediction.
        Args:
        - inputsize (int): number of input features per sample (for 3D variables it's 1 per variable; for 4D variables it's
          'nlevels' per variable; for experiments with more than one variable it's the sum across variables)
        '''
        super().__init__()
        # self.layers = torch.nn.Sequential(
        #     torch.nn.Linear(inputsize,256), torch.nn.GELU(),
        #     torch.nn.Linear(256,128),       torch.nn.GELU(),
        #     torch.nn.Linear(128,64),        torch.nn.GELU(),
        #     torch.nn.Linear(64,32),         torch.nn.GELU(),
        #     torch.nn.Linear(32,1))
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(inputsize,64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64,32),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32,1))

    def forward(self,X):
        '''
        Purpose: Forward pass through the NN.
        Args:
        - X (torch.Tensor): input features tensor of shape (nsamples, inputsize)
        Returns:
        - torch.Tensor: raw prediction tensor of shape (nsamples, 1)
        '''
        return self.layers(X)

class TweedieDevianceLoss(torch.nn.Module):

    def __init__(self,p=1.5,epsilon=1e-12):
        '''
        Purpose: Initialize parameters for computing Tweedie unit deviance loss.
        Args:
        - p (float): Tweedie power parameter where 1 < p < 2
        - epsilon (float): small clamp to keep predicted precipitation away from 0 for numerical stability
        '''
        super().__init__()
        if not (1.0<float(p)<2.0):
            raise ValueError('TweedieDevianceLoss requires 1 < p < 2.')
        self.p       = float(p)
        self.epsilon = float(epsilon)

    def forward(self,ypred,ytrue):
        '''
        Purpose: Define the Tweedie unit deviance using Eq. 10 from Hunt KMR. (2025), arXiv:2509.08369.
        Args:
        - ypred (torch.Tensor): 1D tensor of predicted mean precipitation (≥ 0 mm/hr) 
        - ytrue (torch.Tensor): 1D tensor of observed precipitation (≥ 0 mm/hr)
        Returns:
        - torch.Tensor: mean Tweedie deviance loss
        '''
        ypred = torch.clamp(ypred,min=self.epsilon)
        ytrue = ytrue.to(ypred.dtype)
        terma = torch.clamp(ytrue,min=0.0).pow(2.0-self.p)/((1.0-self.p)*(2.0-self.p))
        termb = (ytrue*ypred.pow(1.0-self.p))/(1.0-self.p)
        termc = ypred.pow(2.0-self.p)/(2.0-self.p)
        deviance = 2.0*(terma-termb+termc)
        return deviance.mean()