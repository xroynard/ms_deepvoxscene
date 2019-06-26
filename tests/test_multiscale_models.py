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

import torch

sys.path.insert(0, os.path.abspath('..'))
from models import VoxNet, ModVoxNet, VGGLike, ModVGGLike

###############################################################################
#ON_CPU = False
ON_CPU = True

#BACKWARD = False
BACKWARD = True

###############################################################################
#%% Test function
def test_multiscale_voxel_model(net, batch_size=32, nb_channels=1, nb_classes=9, nb_scales=1, grid_size=32):
    print("\n\nTest Model : {}".format(net.name()))
        
    net_size = 0
    for name, param in net.named_parameters():
        net_size += param.data.nelement() * param.data.element_size()
    print("{} o --> {} Go {} Mo {} Ko {} o".format(net_size, net_size//1000000000, net_size%1000000000//1000000, net_size%1000000//1000, net_size%1000))
    
#    net.init("testdhgr")
    net.init("xavier")
    net.init("orthogonal")  
    net.init("kaiming")  
    
    input = [torch.rand(batch_size,nb_channels,grid_size,grid_size,grid_size) for _ in range(nb_scales)]
    for i_sc in range(nb_scales):
        input[i_sc][:,0] = torch.round(input[i_sc][:,0])
        
    if ON_CPU:
        print("On CPU:")
        input = [i.cpu() for i in input]
        net = net.cpu()
        start_time = time.perf_counter()
        output = net(input)
        print("\tForward                : {:.4f} s".format(time.perf_counter() - start_time))
            
        grad = torch.randn_like(output)
        start_time = time.perf_counter()
        output.backward(grad)
        print("\tBackward               : {:.4f} s".format(time.perf_counter() - start_time))
    
        best_net = copy.deepcopy(net)
        torch.save(net, "test_save_model_checkpoint_{}.tar".format(net.name()) )
        torch.save(best_net, "test_save_model_checkpoint_{}.tar".format(net.name()) )
            
    print("On GPU, device {}:".format(0))
    net = net.cuda(device=0)
    torch.cuda.synchronize()
    net.zero_grad()
    
    NB_SAMPLES = 10
    forward_time = 0.0
    backward_time = 0.0
    with torch.set_grad_enabled(False):
        for _ in range(NB_SAMPLES):
            input = [torch.rand(batch_size,nb_channels,grid_size,grid_size,grid_size).cuda(device=0) for _ in range(nb_scales)]
            for i_sc in range(nb_scales):
                input[i_sc][:,0] = torch.round(input[i_sc][:,0])
                
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            output = net(input)
            
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            forward_time += time.perf_counter() - start_time
    print("\tForward                : {:.4f} s".format(forward_time/NB_SAMPLES))
        
    if BACKWARD:
        with torch.set_grad_enabled(True):
            for _ in range(NB_SAMPLES):
                input = [torch.rand(batch_size,nb_channels,grid_size,grid_size,grid_size).cuda(device=0) for _ in range(nb_scales)]
                for i_sc in range(nb_scales):
                    input[i_sc][:,0] = torch.round(input[i_sc][:,0])
                    
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                output = net(input)
                
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                forward_time += time.perf_counter() - start_time
               
                grad = torch.randn_like(output)
                
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                output.backward(grad)
                
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                backward_time += time.perf_counter() - start_time
        print("\tForward ( with grad )  : {:.4f} s".format(forward_time/NB_SAMPLES))
        print("\tBackward               : {:.4f} s".format(backward_time))
    
    best_net = copy.deepcopy(net)
    torch.save(net, "test_save_model_checkpoint_{}.tar".format(net.name()) )
    torch.save(best_net, "test_save_model_checkpoint_{}.tar".format(net.name()) )

#%%  
if __name__ == '__main__':
    
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
#    cudnn.benchmark = False
        
#%% Set testing parameters
        
    BATCH_SIZE = 10
    NB_CHANNELS = 2
    NB_CLASSES = 9
    NB_SCALES = 3
    GRID_SIZE = 32
        
#%% Test models
    
    vn = VoxNet(nb_channels=NB_CHANNELS, nb_classes=NB_CLASSES, nb_scales=NB_SCALES)
    test_multiscale_voxel_model(vn, batch_size=BATCH_SIZE, nb_channels=NB_CHANNELS, nb_classes=NB_CLASSES, nb_scales=NB_SCALES, grid_size=GRID_SIZE)
    
    mvn = ModVoxNet(nb_channels=NB_CHANNELS, nb_classes=NB_CLASSES, nb_scales=NB_SCALES)
    test_multiscale_voxel_model(mvn, batch_size=BATCH_SIZE, nb_channels=NB_CHANNELS, nb_classes=NB_CLASSES, nb_scales=NB_SCALES, grid_size=GRID_SIZE)
    
    vgg = VGGLike(nb_channels=NB_CHANNELS, nb_classes=NB_CLASSES, nb_scales=NB_SCALES)
    test_multiscale_voxel_model(vgg, batch_size=BATCH_SIZE, nb_channels=NB_CHANNELS, nb_classes=NB_CLASSES, nb_scales=NB_SCALES, grid_size=GRID_SIZE)
    
    mvgg = ModVGGLike(nb_channels=NB_CHANNELS, nb_classes=NB_CLASSES, nb_scales=NB_SCALES)
    test_multiscale_voxel_model(mvgg, batch_size=BATCH_SIZE, nb_channels=NB_CHANNELS, nb_classes=NB_CLASSES, nb_scales=NB_SCALES, grid_size=GRID_SIZE)