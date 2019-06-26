#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xavier Roynard
"""

from __future__ import print_function, division

import time
import copy
import os
import sys
import numpy as np

import torch
from torch.autograd import Variable

sys.path.insert(0, os.path.abspath('..'))
from models.conv_base.basic_blocks import Initializer, GlobalPoolingBlock, SENet, Conv3dBlock

###############################################################################
#ON_CPU = False
ON_CPU = True

#BACKWARD = False
BACKWARD = True

###############################################################################
def test_voxel_block(net, batch_size=32, in_feature_maps=9, out_feature_maps=None, grid_size=32, grid_grad=False, grid_downsample=False):
        print("\n\nTest Model : {}".format(net.name()))

        out_feature_maps = out_feature_maps or in_feature_maps

        net_size = 0
        for name, param in net.named_parameters():
            print(name,
                  type(param.data),
                  param.size(),
                  param.data.element_size()
                  )
            net_size += param.data.nelement() * param.data.element_size()
        print("{} o --> {} Go {:03d} Mo {:03d} Ko {:03d} o".format(net_size, net_size//1000000000, net_size%1000000000//1000000, net_size%1000000//1000, net_size%1000))

#        net.init("testdhgr")
        net.init("xavier")
        net.init("kaiming")    
        net.init("kaiming")      

        input = torch.randn(batch_size,in_feature_maps,grid_size,grid_size,grid_size)

        print("On CPU:")
        start_time = time.time()
        output = net(input)
        print("\tDurée Forward  : {:.4f} s".format(time.time() - start_time))
        
        grad = torch.randn_like(output)
                    
        start_time = time.time()
        output.backward(grad)
        print("\tDurée Backward : {:.4f} s".format(time.time() - start_time))
    
        best_net = copy.deepcopy(net)
        torch.save(net, "/tmp/test_save_model_checkpoint_{}.tar".format(net.name()) )
        torch.save(best_net, "/tmp/test_save_model_checkpoint_{}.tar".format(net.name()) )

        for DEVICE_ID in range(torch.cuda.device_count()):
            print("Sur GPU, device {}:".format(DEVICE_ID))
            net = net.cuda(device=DEVICE_ID)
            net.zero_grad()        
#            input = Variable(torch.randn(batch_size,in_feature_maps,grid_size,grid_size,grid_size)).cuda(device=DEVICE_ID)
#            start_time = time.time()
#            output = net(input)
#            print("\tDurée Forward  : {:.4f} s".format(time.time() - start_time))
#                
#            grad = grad.cuda(device=DEVICE_ID)
#            start_time = time.time()
#            output.backward(grad)
#            print("\tDurée Backward : {:.4f} s".format(time.time() - start_time))
        
            NB_SAMPLES = 10
            forward_time = 0.0
            backward_time = 0.0
            with torch.set_grad_enabled(False):
                for _ in range(NB_SAMPLES):
                    net.zero_grad()
                    input = torch.randn(batch_size,in_feature_maps,grid_size,grid_size,grid_size).cuda(device=DEVICE_ID)
                    t0 = time.time()
                    output = net(input)
                    forward_time += time.time() - t0
            print("\tDurée Forward no grad : {:.5f} s".format(forward_time/NB_SAMPLES))
            
            forward_time = 0.0
            with torch.set_grad_enabled(True):
                for _ in range(NB_SAMPLES):
                    net.zero_grad()
                    input = torch.randn(batch_size,in_feature_maps,grid_size,grid_size,grid_size).cuda(device=DEVICE_ID)
                    t0 = time.time()
                    output = net(input)
                    forward_time += time.time() - t0
        
                    grad = torch.randn_like(output)
                    
                    t0 = time.time()
                    output.backward(grad)
                    backward_time += time.time() - t0
            print("\tDurée Forward         : {:.5f} s".format(forward_time/NB_SAMPLES))
            print("\tDurée Backward        : {:.5f} s".format(backward_time/NB_SAMPLES))

            best_net = copy.deepcopy(net)
            torch.save(net, "test_save_model_checkpoint_{}.tar".format(net.name()) )
            torch.save(best_net, "test_save_model_checkpoint_{}.tar".format(net.name()) )

#%%  
if __name__ == '__main__':
    
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
#    cudnn.benchmark = False
        
#%%
    BATCH_SIZE = 4
    in_feature_maps = 256
    out_feature_maps = 32
    GRID_SIZE_IN = 32
    SKIP_PROBABILITY = 0.0    
    
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
#    cudnn.benchmark = False

#%%
    
#    sen = SENet(in_feature_maps=in_feature_maps)
#    test_voxel_block(sen, batch_size=BATCH_SIZE, in_feature_maps=in_feature_maps, grid_size=GRID_SIZE_IN, grid_grad=True)
    
#    sen = SENet(in_feature_maps=in_feature_maps)
#    test_voxel_block(sen, batch_size=BATCH_SIZE, in_feature_maps=in_feature_maps, grid_size=GRID_SIZE_IN, grid_grad=True)
    
#    sen = SENet(in_feature_maps=in_feature_maps)
#    test_voxel_block(sen, batch_size=BATCH_SIZE, in_feature_maps=in_feature_maps, grid_size=GRID_SIZE_IN, grid_grad=True)
    
#    sen = SENet(in_feature_maps=in_feature_maps)
#    test_voxel_block(sen, batch_size=BATCH_SIZE, in_feature_maps=in_feature_maps, grid_size=GRID_SIZE_IN, grid_grad=True)
    