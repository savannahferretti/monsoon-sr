#!/usr/bin/env python

import torch

class NNModel(torch.nn.Module):

    def __init__(self,inputsize):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(inputsize,128), torch.nn.SiLU(),
            torch.nn.Linear(128,64),        torch.nn.SiLU(), 
            torch.nn.Linear(64,32),         torch.nn.SiLU(),
            torch.nn.Linear(32,1))

    def forward(self,X):
        return self.layers(X)