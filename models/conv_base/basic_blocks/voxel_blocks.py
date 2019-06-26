#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xavier Roynard
"""

from __future__ import print_function, division

import numpy as np

import torch.nn as nn
from torch.nn.init import calculate_gain

from torch.nn.init import xavier_normal_, orthogonal_, kaiming_normal_

###############################################################################
# Class Initializer
###############################################################################
class Initializer():
    def __init__(self, init_method="kaiming"):
        self.init_method=init_method
        self.valid_methods = ["xavier" , "kaiming" , "orthogonal"]
        if not(self.init_method in self.valid_methods):
            self.init_method="xavier"
            print("Initialization init_method {} not supported. Please use one of {} --> {} used".format(init_method, self.valid_methods, self.init_method))
            
    def __call__(self, tensor, gain=1, a=0, mode='fan_in'):
        if self.init_method=="xavier":
            return xavier_normal_(tensor, gain=gain)
        elif self.init_method=="orthogonal":
            return orthogonal_(tensor, gain=gain)
        else:
            return kaiming_normal_(tensor, a=a, mode=mode)
   
###############################################################################
# Class GlobalPoolingBlock
###############################################################################
class GlobalPoolingBlock(nn.Module):
    def __init__(self, size=1, pool=0):
        super(GlobalPoolingBlock, self).__init__()
        
        self.size = size
        self.pool = pool
                
        if self.pool==0:
            self.g_pooling = nn.AdaptiveMaxPool3d(self.size)
        elif self.pool==1:
            self.g_pooling = nn.AdaptiveAvgPool3d(self.size)
        else:
            print("pool={} != 0 or 1 --> pool=0 (nn.AdaptiveMaxPool3d)".format(self.pool))
            self.pool = 0
            self.g_pooling = nn.AdaptiveAvgPool3d(self.size)  
        
    def forward(self, x):
            return self.g_pooling(x)
        
    def init(self, init_method=None):
        pass
        
    def name(self):
        return "GlobalPoolingBlock" + "_" + str(self.size) + "_" + str(self.pool)
        
###############################################################################        
# Class SENet (Squeeze-and-Excitation Network) from <http://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html>
###############################################################################
class SENet(nn.Module):
    def __init__(self, in_feature_maps=64, squeeze_ratio=16, pool=1, norm=True, init_method="kaiming"):
        super(SENet, self).__init__()
        
        self.in_feature_maps = in_feature_maps
        self.squeeze_ratio = squeeze_ratio
        self.pool = pool
        self.norm=bool(norm)
        hidden_features = max(2,int(np.ceil(in_feature_maps / self.squeeze_ratio))) # <=1 non-sense
        
        self.fc = nn.Sequential()
        
        # Global/Adaptative Pooling
        self.fc.add_module('pool',GlobalPoolingBlock(1,pool=self.pool))
            
        # Squeezing Conv (conv1x1x1)
        self.fc.add_module('conv1', nn.Conv3d(in_feature_maps, hidden_features, 1, bias=not(self.norm)))
            
        if self.norm:
            self.fc.add_module('norm1', nn.BatchNorm3d(hidden_features))
        self.fc.add_module('relu', nn.ReLU())
        
        # Second Conv (Excitation ?) (conv1x1x1)
        self.fc.add_module('conv2', nn.Conv3d(hidden_features, in_feature_maps, 1, bias=not(self.norm)))
            
        if self.norm:
            self.fc.add_module('norm2', nn.BatchNorm3d(in_feature_maps))
        self.fc.add_module('sigmoid', nn.Sigmoid())
        
        self.init(init_method)

    def forward(self, x):        
        return (x * self.fc(x))
    
    def init(self, init_method="kaiming"):
        self.initializer = Initializer(init_method)
                    
        for i,m in enumerate(self.fc):
            if ('Conv' in m.__class__.__name__):
                if i==len(self.fc)-2 or i==len(self.fc)-3:
                    for p in m.parameters():
                        if p.ndimension()>1:
                            p = self.initializer(p, gain=calculate_gain('sigmoid'), a=0)
                else:
                    for p in m.parameters():
                        if p.ndimension()>1:
                            p = self.initializer(p, gain=calculate_gain('relu'), a=0)
    
    def name(self):
        return "SENet_" + str(self.in_feature_maps) + "_" + str(self.squeeze_ratio)

###############################################################################
# Class Conv3dBlock: on crée un block de conv (+ BN relu dropout...)
###############################################################################
class Conv3dBlock(nn.Module):
    def __init__(self,
                 in_feature_maps,
                 out_feature_maps=None,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 norm=nn.BatchNorm3d,
                 activation=nn.ReLU,
                 dp=False,
                 se=False,
                 init_method="kaiming"
                 ):
        super(Conv3dBlock, self).__init__()
        
        # TODO: à virer une fois qu'on aura viré les fonctions .name des Networks
        self.in_feature_maps = in_feature_maps
        self.out_feature_maps = out_feature_maps or in_feature_maps
        self.min_in_out_feature_maps = min(self.in_feature_maps, self.out_feature_maps)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
                
        self.activation = activation
        
###############################################################################        
        # Normalization parameter
        if isinstance(norm,bool):
            if norm:
                norm_class = nn.BatchNorm3d
        else:
            try:
                if norm.__name__ in ['BatchNorm3d','InstanceNorm3d','LayerNorm']: # don't support GroupNorm
                    norm_class = norm
                    norm=True
                elif 'norm' in norm.__name__.casefold():
                    print("Warning: norm.__name__<{}> unrecognized ==> norm=nn.BatchNorm3d".format(norm.__name__))
                    norm_class = nn.BatchNorm3d
                    norm=True
                else:
                    print("Warning: norm<{}> doesn't look like a norm ==> norm=False".format(norm.__name__))
                    norm=False                    
            except AttributeError:
                print("Warning: norm<{}> doesn't look like a norm ==> norm=False".format(norm))
                norm=False
                
        kernel_size, stride, padding, dilation         
        
        # Kernel parameter
        if not(isinstance(kernel_size,int)):
            print("Warning: only int kernel_size accepted! ==> kernel_size=3")
            kernel_size = 3
        elif kernel_size<1:
            print("Warning: only kernel_size > 0 accepted! ==> kernel_size=3")
            kernel_size = 3
                    
        # Stride parameter
        if not(isinstance(stride,int)):
            print("Warning: only int stride accepted! ==> stride=1")
            stride = 1
        elif stride<1:
            print("Warning: only stride > 0 accepted! ==> stride=1")
            stride = 1
            
        # Padding parameter
        if not(isinstance(padding,int)):
            print("Warning: only int padding accepted! ==> padding=0")
            padding = 0
        elif padding<0:
            print("Warning: only dilation >= 0 padding! ==> padding=0")
            padding = 0
            
        # Dilation parameter
        if not(isinstance(dilation,int)):
            print("Warning: only int dilation accepted! ==> dilation=1")
            dilation = 1
        elif dilation<1:
            print("Warning: only dilation > 0 accepted! ==> dilation=1")
            dilation = 1
            
        # Groups parameter
        if self.in_feature_maps % groups != 0:
            print("Warning: in_feature_maps % groups != 0 ==> groups=1")
            groups = 1
        if self.out_feature_maps % groups != 0:
            print("Warning: out_feature_maps % groups != 0 ==> groups=1")
            groups = 1
                        
        # Dropout parameter
        if isinstance(dp, float) and 0.0<=dp<1.0:
            dropout_probability=dp
            dp=True
        elif isinstance(dp, bool) and dp==True:
            print("Warning: dropout_probability not specified! ==> dropout_probability=0.1")
            dropout_probability=0.1
            dp=True
        else:
            dp=False
                    
        # SENet parameter
        if isinstance(se, int) and se>1:
            squeeze_ratio=se
            se=True
        elif isinstance(se, bool) and se==True:
            print("Warning: SENet squeeze_ratio not specified! ==> squeeze_ratio=16")
            squeeze_ratio=16
            se=True
        else:
            se=False
            
        self.name = "Conv3dBlock_" + str(self.kernel_size)+2*("x"+str(self.kernel_size)) + "_" + str(self.in_feature_maps)
       
###############################################################################
        # Create Layers
        self.conv = nn.Sequential()
        
        # BatchNorm
        if norm:
            self.conv.add_module(norm_class.__name__.casefold(),norm_class(in_feature_maps))
            
        # Non-Linearity (Activation)
        if self.activation!=None:
            self.conv.add_module(activation.__name__.casefold(),activation())
        
        # Spatial Dropout
        if dp:
            self.conv.add_module("drop",nn.Dropout3d(dropout_probability))
            
        # Convolution
        self.conv.add_module("conv",nn.Conv3d(in_feature_maps, self.out_feature_maps, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
        
        # Squeeze and Excitation Net
        if se:
            self.conv.add_module("senet",SENet(self.out_feature_maps, squeeze_ratio))
###############################################################################
             
        # Initialize weigths
        self.init(init_method)
        
    def forward(self, x):
        return self.conv(x)
    
    def init(self, init_method="kaiming"):
        initializer = Initializer(init_method)
            
        for i,m in enumerate(self.conv):
            for p in m.parameters():  
                if p.ndimension()>1:
                    if ('Leaky' in self.activation.__class__.__name__):
                        p = initializer(p, gain=calculate_gain('leaky_relu',param=0.1), a=0.1)
                    elif ('RReLU' in self.activation.__class__.__name__):
                        p = initializer(p, gain=calculate_gain('leaky_relu',param=0.229), a=0.229) # around 0.229
                    else:
                        p = initializer(p, gain=calculate_gain('relu'), a=0)
    
    def name(self):
        return self.name