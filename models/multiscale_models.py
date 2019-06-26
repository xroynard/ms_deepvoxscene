#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xavier Roynard
"""

from __future__ import print_function, division

import torch
import torch.nn as nn

from .conv_base import VoxNetBase, ModVoxNetBase, VGGLikeNetBase, ModVGGLikeNetBase, BasicFC

#%% 
###############################################################################
# Class LateFusionMultiScaleTemplate
###############################################################################
class LateFusionMultiScaleTemplate(nn.Module):
    def __init__(self, nb_channels=1, nb_classes=64, nb_scales=3, nb_hidden=128, init_method="orthogonal", encoder_class=VoxNetBase, fc_class=BasicFC):
        super(LateFusionMultiScaleTemplate, self).__init__()
        
        self.nb_channels = nb_channels
        self.nb_classes = nb_classes
        self.nb_scales = nb_scales
        self.nb_hidden = nb_hidden
                
        # Conv Layers
        self.conv = nn.ModuleList([encoder_class(self.nb_channels, self.nb_hidden) for _ in range(self.nb_scales)])
        
        # Fully Connected Layers
        self.fc = fc_class(self.nb_scales,self.nb_hidden,self.nb_classes)
        
        self.init(init_method)
        
    def forward(self, x):
        """Assumes x is a sequence of Tensors with shape: n_batch * n_features * grid_size * grid_size * grid_size"""        
        if isinstance(x,list):
            assert(len(x)==self.nb_scales)
        elif self.nb_scales==1:
            x = [x]
        else:
            raise Exception('input is not a list OR nb_scales>1')
        
        # Conv Layers
        x = [self.conv[i](y) for i,y in enumerate(x)]
        
        # Concatenate
        z_cat = torch.cat([y for i,y in enumerate(x)], dim=1)
        
        # FC
        z = self.fc(z_cat)            
        return z.squeeze(4).squeeze(3).squeeze(2)
    
    def init(self, init_method="orthogonal"):
        
        for m in self.conv:
            m.init(init_method=init_method)
                                
        self.fc.init(init_method=init_method)
                    
#%%
   
###############################################################################
# Class VoxNet
###############################################################################
class VoxNet(nn.Module):
    def __init__(self, nb_channels=1, nb_classes=64, nb_scales=3, nb_hidden=128, init_method="orthogonal"):
        super(VoxNet, self).__init__()
        
        self.model_name = "VoxNet" + "_" + str(nb_channels) + "_" + str(nb_classes) + "_" + str(nb_scales)
        
        self.nb_channels = nb_channels
        self.nb_classes = nb_classes
        self.nb_scales = nb_scales
        self.nb_hidden = nb_hidden
        
        self.template = LateFusionMultiScaleTemplate(nb_channels=nb_channels,
                                                     nb_classes=nb_classes,
                                                     nb_scales=nb_scales,
                                                     nb_hidden=nb_hidden,
                                                     init_method=init_method,
                                                     encoder_class=VoxNetBase,
                                                     fc_class=BasicFC,
                                                     )
        
        self.template.init(init_method)
        
    def forward(self, x):
        return self.template(x)
        
    def init(self, init_method="orthogonal"):
        self.template.init(init_method)
        
    def name(self):
        return self.model_name
    
###############################################################################
# Class ModVoxNet
###############################################################################
class ModVoxNet(nn.Module):
    def __init__(self, nb_channels=1, nb_classes=64, nb_scales=3, nb_hidden=128, init_method="orthogonal"):
        super(ModVoxNet, self).__init__()
        
        self.model_name = "ModVoxNet" + "_" + str(nb_channels) + "_" + str(nb_classes) + "_" + str(nb_scales)
        
        self.nb_channels = nb_channels
        self.nb_classes = nb_classes
        self.nb_scales = nb_scales
        self.nb_hidden = nb_hidden
        
        self.template = LateFusionMultiScaleTemplate(nb_channels=nb_channels,
                                                     nb_classes=nb_classes,
                                                     nb_scales=nb_scales,
                                                     nb_hidden=nb_hidden,
                                                     init_method=init_method,
                                                     encoder_class=ModVoxNetBase,
                                                     fc_class=BasicFC,
                                                     )
        
        self.template.init(init_method)
        
    def forward(self, x):
        return self.template(x)
        
    def init(self, init_method="orthogonal"):
        self.template.init(init_method)
        
    def name(self):
        return self.model_name
    
###############################################################################
# Class VGGLikeNet
###############################################################################
class VGGLikeNet(nn.Module):
    def __init__(self, nb_channels=1, nb_classes=64, nb_scales=3, nb_hidden=128, init_method="orthogonal"):
        super(VGGLikeNet, self).__init__()
        
        self.model_name = "VGGLikeNet" + "_" + str(nb_channels) + "_" + str(nb_classes) + "_" + str(nb_scales)
        
        self.nb_channels = nb_channels
        self.nb_classes = nb_classes
        self.nb_scales = nb_scales
        self.nb_hidden = nb_hidden
        
        self.template = LateFusionMultiScaleTemplate(nb_channels=nb_channels,
                                                     nb_classes=nb_classes,
                                                     nb_scales=nb_scales,
                                                     nb_hidden=nb_hidden,
                                                     init_method=init_method,
                                                     encoder_class=VGGLikeNetBase,
                                                     fc_class=BasicFC,
                                                     )
        
        self.template.init(init_method)
        
    def forward(self, x):
        return self.template(x)
        
    def init(self, init_method="orthogonal"):
        self.template.init(init_method)
        
    def name(self):
        return self.model_name
    
###############################################################################
# Class ModVGGLikeNet
###############################################################################
class ModVGGLikeNet(nn.Module):
    def __init__(self, nb_channels=1, nb_classes=64, nb_scales=3, nb_hidden=128, init_method="orthogonal"):
        super(ModVGGLikeNet, self).__init__()
        
        self.model_name = "ModVGGLikeNet" + "_" + str(nb_channels) + "_" + str(nb_classes) + "_" + str(nb_scales)
        
        self.nb_channels = nb_channels
        self.nb_classes = nb_classes
        self.nb_scales = nb_scales
        self.nb_hidden = nb_hidden
        
        self.template = LateFusionMultiScaleTemplate(nb_channels=nb_channels,
                                                     nb_classes=nb_classes,
                                                     nb_scales=nb_scales,
                                                     nb_hidden=nb_hidden,
                                                     init_method=init_method,
                                                     encoder_class=ModVGGLikeNetBase,
                                                     fc_class=BasicFC,
                                                     )
        
        self.template.init(init_method)
        
    def forward(self, x):
        return self.template(x)
        
    def init(self, init_method="orthogonal"):
        self.template.init(init_method)
        
    def name(self):
        return self.model_name

###############################################################################
#%% Dict of models
###############################################################################
models_dict = {'VoxNet':VoxNet,
               'ModVoxNet':ModVoxNet,
               'VGGLikeNet':VGGLikeNet,
               'ModVGGLikeNet':ModVGGLikeNet,
               }