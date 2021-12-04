#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xavier Roynard
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.nn.init import calculate_gain

#from . import basic_blocks
from .basic_blocks import Initializer, GlobalPoolingBlock, Conv3dBlock

#%%
###############################################################################    
# Class VoxNetBase
###############################################################################
class VoxNetBase(torch.nn.Module):
    def __init__(self, nb_channels=1, nb_hidden=32):
        super(VoxNetBase, self).__init__()
        
        self.model_name = "VoxNetBase" + "_" + str(nb_channels) + "_" + str(nb_hidden)
        
        self.nb_channels = nb_channels
        self.nb_hidden = nb_hidden
        
        self.conv = nn.Sequential(nn.Conv3d(self.nb_channels, 32, 5, stride=2), # 32 -> 14
                                  nn.LeakyReLU(negative_slope=0.1),
                                  nn.Conv3d(32, self.nb_hidden, 3), # 14 -> 12
                                  nn.LeakyReLU(negative_slope=0.1),
                                  nn.MaxPool3d(2) # 12 -> 6
                                  )

    def forward(self, x):        
        x = self.conv(x)
        
        return x
    
    def init(self, init_method="orthogonal"):
        initializer = Initializer(init_method)        
                    
        for m in self.conv:
            if m.__class__.__name__.find('Conv')!=-1:
                for p in m.parameters():  
                    if p.ndimension()>1:              
                        p = initializer(p, gain=calculate_gain('leaky_relu',param=0.1), a=0.1)                    
    
    def name(self):
        return self.model_name
    
###############################################################################
# Class ModVoxNetBase
###############################################################################
class ModVoxNetBase(torch.nn.Module):
    def __init__(self, nb_channels=1, nb_hidden=128):
        super(ModVoxNetBase, self).__init__()
                
        self.model_name = "ModVoxNetBase" + "_" + str(nb_channels) + "_" + str(nb_hidden)
        
        self.nb_channels = nb_channels
        self.nb_hidden = nb_hidden
        
        self.conv = nn.Sequential(Conv3dBlock(self.nb_channels, 32, 5, stride=2, bias=False), # 32 -> 14
                                  Conv3dBlock(32, 64, 3, bias=False), # 14 -> 12
                                  Conv3dBlock(64, 64, 3, stride=2, padding=1, bias=False), # 12 -> 6
                                  Conv3dBlock(64, 128, 3, bias=False), # 6 -> 4
                                  Conv3dBlock(128, self.nb_hidden, 3, stride=2, padding=1, bias=False), # 4 -> 2
                                  )

    def forward(self, x):        
        x = self.conv.forward(x)
        
        return x.squeeze(4).squeeze(3).squeeze(2)
    
    def init(self, init_method="orthogonal"):
        initializer = Initializer(init_method)
        
        for i,m in enumerate(self.conv):
            if m.__class__.__name__.find('Conv')!=-1:
                if i==len(self.conv)-1:
                    for p in m.parameters():  
                        if p.ndimension()>1:                      
                            p = initializer(p, gain=calculate_gain('sigmoid'), a=0)
                else:
                    for p in m.parameters():  
                        if p.ndimension()>1:
                            p = initializer(p, gain=calculate_gain('leaky_relu',param=0.25), a=0.25)
    
    def name(self):
        return self.model_name
    
#%%
###############################################################################
# Class VGGLikeNetBase
###############################################################################
class VGGLikeNetBase(torch.nn.Module):
    def __init__(self, nb_channels=1, nb_hidden=64):
        super(VGGLikeNetBase, self).__init__()
        
        self.model_name = "VGGLikeNetBase" + "_" + str(nb_channels) + "_" + str(nb_hidden)
        
        self.nb_channels = nb_channels
        self.nb_hidden = nb_hidden
        
        self.conv = nn.Sequential(Conv3dBlock(self.nb_channels, 32, 3, norm=False, activation=nn.ReLU), # 32 -> 30
                                  Conv3dBlock(32, 32, 3, norm=False, activation=nn.ReLU), # 30 -> 28 
                                  Conv3dBlock(32, 32, 3, stride=2, padding=1, norm=False, activation=nn.ReLU), # 28 -> 14
                                  
                                  Conv3dBlock(32, 64, 3, norm=False, activation=nn.ReLU), # 14 -> 12 
                                  Conv3dBlock(64, 64, 3, norm=False, activation=nn.ReLU), # 12 -> 10
                                  Conv3dBlock(64, self.nb_hidden, 3, stride=2, padding=1, norm=False, activation=nn.ReLU), # 10 -> 5
                                  )

    def forward(self, x):        
        x = self.conv(x)
        
        return x
    
    def init(self, init_method="orthogonal"):
        initializer = Initializer(init_method)
                    
        for m in self.conv:
            if m.__class__.__name__.find('Conv')!=-1:
                for p in m.parameters():
                    if p.ndimension()>1:
                        p = initializer(p, gain=calculate_gain('relu'), a=0)
    
    def name(self):
        return self.model_name
    
###############################################################################
# Class ModVGGLikeNetBase
###############################################################################
class ModVGGLikeNetBase(torch.nn.Module):
    def __init__(self, nb_channels=1, nb_hidden=64):
        super(ModVGGLikeNetBase, self).__init__()
        
        self.model_name = "ModVGGLikeNetBase" + "_" + str(nb_channels) + "_" + str(nb_hidden)
        
        self.nb_channels = nb_channels
        self.nb_hidden = nb_hidden
                
        self.conv = nn.Sequential(Conv3dBlock(self.nb_channels, 32, 3, bias=False, se=16), # 32 -> 30
                                  Conv3dBlock(32, 32, 3, bias=False, se=16), # 30 -> 28 
                                  Conv3dBlock(32, 32, 3, stride=2, padding=1, bias=False, se=16), # 28 -> 14
                                  
                                  Conv3dBlock(32, 64, 3, bias=False, se=16), # 14 -> 12 
                                  Conv3dBlock(64, 64, 3, bias=False, se=16), # 12 -> 10
                                  Conv3dBlock(64, self.nb_hidden, 3, stride=2, padding=1, bias=False, se=16), # 10 -> 5
                                  )

    def forward(self, x):        
        x = self.conv(x)
        
        return x
    
    def init(self, init_method="orthogonal"):
        initializer = Initializer(init_method)
                    
        for m in self.conv:
            if m.__class__.__name__.find('Conv')!=-1:
                for p in m.parameters():
                    if p.ndimension()>1:
                        p = initializer(p, gain=calculate_gain('leaky_relu',param=0.25), a=0.25)
    
    def name(self):
        return self.model_name
     
#%%
###############################################################################
# Class BasicFC
###############################################################################
class BasicFC(torch.nn.Module):
    def __init__(self, nb_scales=1, nb_hidden=256, nb_classes=1, return_intermediate=False):
        super(BasicFC, self).__init__()
        
        self.model_name = "BasicFC" + "_" + str(nb_scales) + "_" + str(nb_hidden) + "_" + str(nb_classes)
        
        self.nb_scales = nb_scales
        self.nb_hidden = nb_hidden
        self.nb_classes = nb_classes
        
        self.conv = nn.Sequential(GlobalPoolingBlock(),
                                  Conv3dBlock(self.nb_scales*self.nb_hidden,256,1,norm=False),
                                  Conv3dBlock(256,self.nb_classes,1,dp=0.5),
                                  )

    def forward(self, x):
        return self.conv(x)
    
    def init(self, init_method="orthogonal"):
                    
        for m in self.conv:
            m.init(init_method)
                
    def name(self):
        return self.model_name