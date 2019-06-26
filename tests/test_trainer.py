#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xavier Roynard
"""

from __future__ import print_function, division

import os
import sys
import time
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.abspath('..'))
from models import VoxNet
from utils.trainer import Trainer

#%%
def test_trainer(crit, batch_size=32, nb_channels=3, nb_classes=9):
    
    # TODO
    pass
    
#    grid_size = 32
#    input = Variable(torch.randn(batch_size, nb_channels, grid_size,grid_size,grid_size))
#    net = VoxNet(nb_channels=nb_channels, nb_classes=nb_classes)
#          
#    output = net(input)
#    
#    target = Variable(torch.from_numpy(np.random.randint(nb_classes, size=batch_size)).long())
#        
#    print("\ninput  :", input.size())
#    print("output :", output[0].size(), " -- ", output[1].size())
#    print("target :", target.size())
#    
#    print("Sur CPU:")
#    start_time = time.time()
#    loss = crit(output, target)
#    print("\tLoss:{}".format(loss.data))
#    print("\tDurée Forward Loss  : {:.4f} s".format(time.time() - start_time))
#            
#    start_time = time.time()
#    loss.backward()
#    print("\tDurée Backward Loss : {:.4f} s".format(time.time() - start_time))
#            
#    for DEVICE_ID in range(torch.cuda.device_count()):
#        print("Sur GPU, device {}:".format(DEVICE_ID))
#        net = net.cuda(device=DEVICE_ID)
#        input = input.cuda(device=DEVICE_ID)
#        
#        output = net(input)
#        
#        target = target.cuda(device=DEVICE_ID)        
#        
#        start_time = time.time()
#        loss = crit(output, target)
#        print("\tLoss:{}".format(loss.data))
#        print("\tDurée Forward Loss  : {:.4f} s".format(time.time() - start_time))
#                
#        start_time = time.time()
#        loss.backward()
#        print("\tDurée Backward Loss : {:.4f} s".format(time.time() - start_time))

#%% Test losses 
if __name__ == '__main__':
    
    # TODO
#    trainer = Trainer(,,,)