import os
import sys
sys.path.append('utils/')
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, sampler
import torch.optim as optim

class ConvNet(nn.Module):
    def __init__(self):
        """
        init function to create the network
        """
        super(ConvNet, self).__init__()

        #hyperparameters for 3d conv layers
        kernel_3d = (3, 4, 5)
        maxpool_3d = (1, 1, 3)
        mpstride_3d = (1, 1, 3)

        #hyperparameters for 2d conv layer
        kernel_2d = (3, 5)
        maxpool_2d = (3, 3)
        mpstride_2d =(3, 3)
        
        #conv3d layers
        self.conv3d = nn.Sequential(nn.Conv3d(1, 32, kernel_3d), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(maxpool_3d, stride=mpstride_3d),
                                   nn.Conv3d(32, 64, kernel_3d), nn.BatchNorm3d(64),   nn.ReLU(),nn.MaxPool3d(maxpool_3d, stride=mpstride_3d), 
                                   nn.Conv3d(64, 128, kernel_3d), nn.BatchNorm3d(128),   nn.ReLU(),nn.MaxPool3d(maxpool_3d, stride=mpstride_3d))
        
        #conv2d layers
        self.features = nn.Sequential(nn.Conv2d(128, 256, kernel_2d), nn.BatchNorm2d(256),  nn.ReLU(), nn.MaxPool2d(maxpool_2d, stride=mpstride_2d))

        #classifier layer
        self.classifier = nn.Sequential(nn.Linear(256*3*4, 4))
        
    def forward(self, input_data):
        # type: (Tensor, int) -> Tensor
        """
        :param input_data: input EEG tensor
        :return: network output
        """
        x = self.conv3d(x)
        x = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3], x.shape[4])    
        x = self.features(x)
        x = torch.flatten(x,1) 
        x = self.classifier(x)
        return x  
