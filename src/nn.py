#!/usr/bin/env python

import torch
import numpy as np

class MLPModel(nn.Module):

    def __init__(self,inputsize):
        '''
        Purpose: Define a feedforward multi-layer perceptron (MLP) for precipitation prediction.
        Args:
        - inputsize (int): number of pressure levels the input data is available on (1 for surface variables)
        '''
        super(MLPModel,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputsize,256),nn.BatchNorm1d(256),nn.GELU(),
            nn.Linear(256,128),nn.BatchNorm1d(128),nn.GELU(),
            nn.Linear(128,64),nn.BatchNorm1d(64),nn.GELU(),
            nn.Linear(64,32),nn.BatchNorm1d(32),nn.GELU(),
            nn.Linear(32,1),nn.ReLU())

    def forward(self,X):
        '''
        Purpose: Forward pass through the MLP.
        Args:
        - X (torch.Tensor): input features tensor of shape (nsamples, inputsize)
        Returns:
        - torch.Tensor: raw prediction tensor of shape (nsamples, 1)
        '''
        return self.layers(x)