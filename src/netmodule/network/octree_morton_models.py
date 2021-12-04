#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 18:06:48 2017

@author: Xavier Roynard
"""

from __future__ import print_function, division

import time
import copy
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import calculate_gain

print("multiscale_segmentation_voxel_models.py: __name__ --> ",__name__)

from .base_networks import Initializer, SENet, SEAttentionNet, MSSENet, Conv3dBlock, GlobalPoolingBlock, VoxNetBase, ModVoxNetBase, MiniVGG16Base, ModMiniVGG16Base, LightResNetEncoderBase, LightResNetHidden, LightResNetDecoderBase, LightResNetFC, NonNaiveMSFusionBase, BasicFC, UltimeNetEncoderBase, UltimeNetDecoderBase, ForOctreeInferenceEncoderBase, ForOctreeInferenceDecoderBase, BasicHidden, Only1BlockForOctreeEncoderBase, Only1BlockForOctreeDecoderBase, VGG1Conv3x3x3EncoderBase, VGG2Conv2x2x2EncoderBase, VGG3Conv2x2x2EncoderBase, BN_VGG1Conv3x3x3EncoderBase, BN_VGG1Conv3x3x3DecoderBase, BN_VGG2Conv2x2x2EncoderBase, BN_VGG3Conv2x2x2EncoderBase, No_VGG2Conv2x2x2EncoderBase, No_VGG3Conv2x2x2EncoderBase

sys.path.insert(0, os.path.abspath('..'))
from utils.octree import OctreeMorton

#DEBUG = True
DEBUG = False

#%% Définition des Templates
###############################################################################
# Classe OctreeMortonNet
###############################################################################
class OctreeMortonNet(nn.Module):
    def __init__(self, nb_channels=1, nb_classes=64, nb_hidden=128, leaf_size=0.1, init="orthogonal", skip_connections=True, return_segmentation=False, concatenate_skip=False, encoder_class=LightResNetEncoderBase, hidden_class=LightResNetHidden, decoder_class=LightResNetDecoderBase, fc_class=LightResNetFC):
        super(OctreeMortonNet, self).__init__()
                
        self.model_name = "OctreeMortonNet" + "_" + str(nb_channels) + "_" + str(nb_classes) + "_" + str(nb_scales)
        
        self.nb_channels = nb_channels
        self.nb_classes = nb_classes
        self.nb_hidden = nb_hidden
        
        self.skip_connections = skip_connections # 
        self.return_segmentation = return_segmentation #
        self.concatenate_skip = concatenate_skip #
        
        self.leaf_size = leaf_size
        self.octree = OctreeMorton(self.leaf_size)
               
        ######################################################################
        # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
        self.max_net_depth = 5
        
        # Initial Conv Layer (convert from input radiometric channels to latent space)
        self.init_conv = Conv3dBlock(self.nb_channels,
                                     self.nb_hidden,
                                     1)
        
        # Conv Layers
        self.conv = {depth:nn.Sequential(Conv3dBlock(self.nb_hidden,
                                              2*self.nb_hidden,
                                              norm=False,
                                              2),
                                  Conv3dBlock(2*self.nb_hidden,
                                              self.nb_hidden,
                                              norm=False,
                                              1),
                                  )
                     for depth in range(self.max_net_depth)}
        
        # # Aggragation
        # self.aggregate = 
                                  
        # Hidden Layers
        self.hidden = Conv3dBlock(self.nb_hidden,self.nb_hidden,1)
        
        # DeConv Layers
        self.deconv = {depth:nn.Sequential(Conv3dBlock(self.nb_hidden,
                                                2*self.nb_hidden,
                                                2,
                                                norm=False,
                                                transpose=True),
                                    Conv3dBlock(2*self.nb_hidden,
                                                self.nb_hidden,
                                                norm=False,
                                                1),
                                    )
                       for depth in range(self.max_net_depth)}
        
        # Fully Connected Layers
        self.final_conv = Conv3dBlock(self.nb_hidden,
                                      self.nb_classes,
                                      1)
        ######################################################################
        
        self.init(init)
        
    def forward(self, x):
        """Assumes x is a point cloud: n_points * n_features """
        
        init_index, morton_codes, reverse_index, aggregated_points = self.octree.compute_index(x)
        
        max_depth = len(morton_codes)
        
        features = {'up':{},'down':{}}
        features['up'][0] = self.init_block[0](aggregated_points[0])
        # Up pass in network 
        for depth in range(self.max_net_depth):
            # TODO : initialize <self.feature_sizes[depth]> (in __init__)
            #        and <self.net_block['up'][depth]>
            #        and <self.init_block[depth]>
            #        and <self.aggregate[depth]>
            
            temp_conv_input = torch.zeros((morton_codes[depth].size(0),features['up'][depth-1].size(),2,2,2),device=x.device,dtype=x.dtype)
            temp_conv_input[reverse_index[depth],
                            :,
                            (morton_codes[depth-1] & 0b001),
                            (morton_codes[depth-1] & 0b010)>>1,
                            (morton_codes[depth-1] & 0b100)>>2] = features['up'][depth-1].squeeze()
            features['up'][depth] = self.net_block['up'][depth](temp_conv_input)
            
            features['up'][depth] = self.aggregate[depth](features['up'][depth], self.init_block[depth](aggregated_points[depth]))
            
            
        # TODO : between up and down pass --> Aggregate over all max depth of net Then FC
        features['down'][self.max_net_depth-1] = torch.mean(features['up'][self.max_net_depth-1], 0).unsqueeze(0)
        
        # Down pass in network
        for depth in range(self.max_net_depth-1,0,-1):
            # TODO : initialize <self.aggregate[depth]> (in __init__)
            # and  <self.net_block['down'][depth]>
            temp_conv_input = self.aggregate[depth](features['up'][depth], features['down'][depth])
            features['down'][depth-1] = self.net_block['down'][depth](temp_conv_input)[reverse_index[depth],
                                                                                :,
                                                                                (morton_codes[depth-1] & 0b001),
                                                                                (morton_codes[depth-1] & 0b010)>>1,
                                                                                (morton_codes[depth-1] & 0b100)>>2].unsqueeze(2).unsqueeze(3).unsqueeze(4)
            
        # Final conv1x1x1 block to get classes            
        features['down'][0] = self.final_block(features['down'][0])
            
        output = {'predicted':features['down'][0].squeeze(4).squeeze(3).squeeze(2),
                  'index':init_index}
        return 
        ######################################################################
        # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
        # # Conv Laye
        # x = [self.conv[i](y) for i,y in enumerate(x)]
        
        # # Concatenate
        # if self.skip_connections:
        #     z_cat = torch.cat([y[0] for i,y in enumerate(x)], dim=1)
        # else:rs
        #     z_cat = torch.cat([y for i,y in enumerate(x)], dim=1)
        
        # if self.return_segmentation: # On fait de la segmentation
        #     # Hidden Layers
        #     z = self.hidden(z_cat)
            
        #     # Devonv Layers
        #     if self.skip_connections:
        #         z = [self.deconv[i](z, inter=y[1]) for i,y in enumerate(x)]
        #     else:
        #         z = [self.deconv[i](z) for i,_ in enumerate(x)]
                
        #     return z
            
        # else: # On fait de la classificiation
        #     # Fully Connected Layers
        #     z = self.fc(z_cat)
        #######################################################################
        
    def init(self, method="xavier"):
        
        for m in self.conv:
            m.init(method=method)
            
        self.hidden.init(method=method)
            
        for m in self.deconv:
            m.init(method=method)
                                
        self.fc.init(method=method)
                   
###############################################################################
# Classe OctreeMortonNetOneBlock
###############################################################################
class OctreeMortonNetOneBlock(nn.Module):
    def __init__(self, nb_channels=1, nb_classes=64, nb_hidden=128, leaf_size=0.1, init="orthogonal", skip_connections=True, return_segmentation=False, concatenate_skip=False, encoder_class=LightResNetEncoderBase, hidden_class=LightResNetHidden, decoder_class=LightResNetDecoderBase, fc_class=LightResNetFC):
        super(OctreeMortonNetOneBlock, self).__init__()
        
        self.model_name = "OctreeMortonNetOneBlock" + "_" + str(nb_channels) + "_" + str(nb_classes) + "_" + str(nb_scales)
        
        self.nb_channels = nb_channels
        self.nb_classes = nb_classes
        self.nb_hidden = nb_hidden
        
        self.skip_connections = skip_connections # 
        self.return_segmentation = return_segmentation #
        self.concatenate_skip = concatenate_skip #
        
        self.leaf_size = leaf_size
        self.octree = OctreeMorton(self.leaf_size)
               
        ######################################################################
        # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
        
        # Initial Conv Layer (convert from input radiometric channels to latent space)
        self.init_conv = Conv3dBlock(self.nb_channels,
                                     self.nb_hidden,
                                     1)
        
        # Conv Layers
        self.conv = nn.Sequential(Conv3dBlock(self.nb_hidden,
                                              2*self.nb_hidden,
                                              norm=False,
                                              2),
                                  Conv3dBlock(2*self.nb_hidden,
                                              self.nb_hidden,
                                              norm=False,
                                              1),
                                  )
                                          
        # Hidden Layers
        self.hidden = Conv3dBlock(self.nb_hidden,self.nb_hidden,1)
        
        # DeConv Layers
        self.deconv = nn.Sequential(Conv3dBlock((1+if self.cat_aggregate)*self.nb_hidden,
                                                2*self.nb_hidden,
                                                2,
                                                norm=False,
                                                transpose=True),
                                    Conv3dBlock(2*self.nb_hidden,
                                                self.nb_hidden,
                                                norm=False,
                                                1),
                                    )
        
        # Fully Connected Layers
        self.final_conv = Conv3dBlock(self.nb_hidden,
                                      self.nb_classes,
                                      1)
        ######################################################################
        
        self.init(init)
        
    def forward(self, x):
        """Assumes x is a point cloud: n_points * n_features """
        
        init_index, morton_codes, reverse_index, aggregated_points = self.octree.compute_index(x)
        
        max_depth = len(morton_codes)
        
        features = {'up':{},'down':{}}
        features['up'][0] = self.init_block[0](aggregated_points[0])
        output_features = {}
        # Up pass in network 
        for depth in range(1,max_depth):
            temp_conv_input = torch.zeros((morton_codes[depth].size(0),features['up'][depth-1].size(),2,2,2),device=x.device,dtype=x.dtype)
            temp_conv_input[reverse_index[depth],
                            :,
                            (morton_codes[depth-1] & 0b001),
                            (morton_codes[depth-1] & 0b010)>>1,
                            (morton_codes[depth-1] & 0b100)>>2] = features['up'][depth-1].squeeze()
            features['up'][depth] = self.conv(temp_conv_input)
            
            features['up'][depth] = 0.5 * (features['up'][depth] + self.init_conv(aggregated_points[depth]))
            
            
        # Between up and down pass
        features['down'][max_depth-1] = self.hidden(features['up'][max_depth-1])
        
        # Down pass in network
        for depth in range(max_depth-1,0,-1):
            temp_conv_input = features['down'][depth]
            if self.skip_connections:
                if self.cat_aggregate:
                    temp_conv_input = torch.cat((temp_conv_input,features['up'][depth]),1)
                else:
                    temp_conv_input = 0.5 * (temp_conv_input + features['up'][depth])
            features['down'][depth-1] = self.deconv(temp_conv_input)[reverse_index[depth],
                                                                     :,
                                                                     (morton_codes[depth-1] & 0b001),
                                                                     (morton_codes[depth-1] & 0b010)>>1,
                                                                     (morton_codes[depth-1] & 0b100)>>2].unsqueeze(2).unsqueeze(3).unsqueeze(4)
            
            # Final conv1x1x1 block to get classes
            output_features[depth-1] = self.final_conv(features['down'][depth-1]).squeeze(4).squeeze(3).squeeze(2)
            
        output = {'predicted':output_features,
                  'index':{'init_index':init_index,
                           'morton_codes':morton_codes,
                           'reverse_index':reverse_index,
                           'aggregated_points':aggregated_points
                           }
                  }
        return output
        ######################################################################
        # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
        # # Conv Layer
        # x = [self.conv[i](y) for i,y in enumerate(x)]
        
        # # Concatenate
        # if self.skip_connections:
        #     z_cat = torch.cat([y[0] for i,y in enumerate(x)], dim=1)
        # else:
        #     z_cat = torch.cat([y for i,y in enumerate(x)], dim=1)
        
        # if self.return_segmentation: # On fait de la segmentation
        #     # Hidden Layers
        #     z = self.hidden(z_cat)
            
        #     # Devonv Layers
        #     if self.skip_connections:
        #         z = [self.deconv[i](z, inter=y[1]) for i,y in enumerate(x)]
        #     else:
        #         z = [self.deconv[i](z) for i,_ in enumerate(x)]
                
        #     return z
            
        # else: # On fait de la classificiation
        #     # Fully Connected Layers
        #     z = self.fc(z_cat)
        #######################################################################
        
    def init(self, method="xavier"):
        
        for m in self.conv:
            m.init(method=method)
            
        self.hidden.init(method=method)
            
        for m in self.deconv:
            m.init(method=method)
                                
        self.fc.init(method=method)
#%% Instanciation des template      

##----------------------------------------------------------------------------#
##---# Class OctNet
##----------------------------------------------------------------------------#
class OctNet(nn.Module):
    def __init__(self, nb_channels=1, nb_classes=64, nb_hidden=486, init="orthogonal", skip_connections=True, return_segmentation=False, concatenate_skip=False):
        super(OctNet, self).__init__()
        
        self.model_name = "OctNet" + "_" + str(nb_channels) + "_" + str(nb_classes) + "_" + str(nb_scales)
        
        self.nb_channels = nb_channels
        self.nb_classes = nb_classes
        self.nb_scales = nb_scales
        self.nb_hidden = 486 # Attention à changer si on change nb_maps ou nb_down dans ForOctreeInferenceEncoderBase ...
        
        self.skip_connections = skip_connections
        self.concatenate_skip = concatenate_skip
        
        self.template = OctreeMortonNet(nb_channels=nb_channels,
                                                     nb_classes=nb_classes,
                                                     nb_scales=nb_scales,
                                                     nb_hidden=self.nb_hidden,
                                                     init=init,
                                                     skip_connections=skip_connections,
                                                     return_segmentation=return_segmentation,
                                                     concatenate_skip=concatenate_skip,
                                                     encoder_class=ForOctreeInferenceEncoderBase,
                                                     hidden_class=BasicHidden,
                                                     decoder_class=ForOctreeInferenceDecoderBase,
                                                     fc_class=BasicFC,
                                                     )
        
        self.template.init(init)
        
    def forward(self, x):
        return self.template(x)
        
    def init(self, method="xavier"):
        self.template.init(method)
        
    def name(self):
        return self.model_name

###############################################################################
#%% List of models
multiscale_segmentation_voxel_model_dict = {
                                            'OctreeMortonNet':OctreeMortonNet,
                                            'OctreeMortonNetOneBlock':OctreeMortonNetOneBlock,
                                            }