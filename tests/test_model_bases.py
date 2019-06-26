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
from torch.autograd import Variable

sys.path.insert(0, os.path.abspath('..'))
from models.conv_base import VoxNetBase, ModVoxNetBase, VGGLikeNetBase, ModVGGLikeNetBase, BasicFC

###############################################################################
#ON_CPU = False
ON_CPU = True

#BACKWARD = False
BACKWARD = True

#%% Test model bases
    
def test_model_base(net, batch_size=32, nb_channels=9, nb_hidden=None, grid_size=32):
    print("\n\nTest Model : {}".format(net.name()))
        
    net_size = 0
    for name, param in net.named_parameters():
#            print(name, type(param.data), param.size(), param.data.element_size())
        net_size += param.data.nelement() * param.data.element_size()
    print("{} o --> {} Go {} Mo {} Ko {} o".format(net_size, net_size//1000000000, net_size%1000000000//1000000, net_size%1000000//1000, net_size%1000))
    
    net.init("testdhgr")
    net.init("xavier")
    net.init("orthogonal")    
    net.init("kaiming")      
    
    input = Variable(torch.randn(batch_size,nb_channels,grid_size,grid_size,grid_size))
        
    start_time = time.time()
    output = net(input)
    print("\tDurée Forward  : {:.4f} s".format(time.time() - start_time))
        
    grad = torch.randn_like(output)
    start_time = time.time()
    output.backward(grad)
    print("\tDurée Backward : {:.4f} s".format(time.time() - start_time))
    
    for DEVICE_ID in range(torch.cuda.device_count()):
        print("Sur GPU, device {}:".format(DEVICE_ID))
        net = net.cuda(device=DEVICE_ID)
        net.zero_grad()        
        input = Variable(torch.randn(batch_size,nb_channels,grid_size,grid_size,grid_size)).cuda(device=DEVICE_ID)
        start_time = time.time()
        output = net(input)
        print("\tDurée Forward Cuda  : {:.4f} s".format(time.time() - start_time))
            
        grad = grad.cuda(device=DEVICE_ID)
        start_time = time.time()
        output.backward(grad)
        print("\tDurée Backward Cuda : {:.4f} s".format(time.time() - start_time))
        
        best_net = copy.deepcopy(net)
        torch.save(net, "test_save_model_checkpoint_{}.tar".format(net.name()) )
        torch.save(best_net, "test_save_model_checkpoint_{}.tar".format(net.name()) )
        
#%%
if __name__ == '__main__':
    BATCH_SIZE = 4
    NB_CHANNELS = 16
    NB_CLASSES = 9
    GRID_SIZE = 32
    SKIP_PROBABILITY = 0.0
    
#%%
    vn = VoxNetBase(nb_channels=NB_CHANNELS, nb_hidden=NB_CLASSES)
    test_model_base(vn, batch_size=BATCH_SIZE, nb_channels=NB_CHANNELS, nb_hidden=NB_CLASSES, grid_size=GRID_SIZE)

    mvn = ModVoxNetBase(nb_channels=NB_CHANNELS, nb_hidden=NB_CLASSES)
    test_model_base(mvn, batch_size=BATCH_SIZE, nb_channels=NB_CHANNELS, nb_hidden=NB_CLASSES, grid_size=GRID_SIZE)

    mvgg = VGGLikeNetBase(nb_channels=NB_CHANNELS, nb_hidden=NB_CLASSES)
    test_model_base(mvgg, batch_size=BATCH_SIZE, nb_channels=NB_CHANNELS, nb_hidden=NB_CLASSES, grid_size=GRID_SIZE)

    mmvgg = ModVGGLikeNetBase(nb_channels=NB_CHANNELS, nb_hidden=NB_CLASSES)
    test_model_base(mmvgg, batch_size=BATCH_SIZE, nb_channels=NB_CHANNELS, nb_hidden=NB_CLASSES, grid_size=GRID_SIZE)