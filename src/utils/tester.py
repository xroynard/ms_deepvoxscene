#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xavier Roynard
"""

from __future__ import print_function, division

import time
import os
import sys
import numpy as np

import torch
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# pour importer les modules 
sys.path.insert(0, os.path.abspath('..'))

class Tester(object):
    
    ###########################################################################
    # Initializes Tester
    ###########################################################################
    def __init__(self,
                 params,
                 dset_loader,
                 model,
                 ):
        
        self.params = params
        self.dset_loader = dset_loader
        self.model = model
    
        # Dataset size
        self.dataset_size = len(self.dset_loader.dataset)
    
    ###########################################################################
    # Tests self.model
    ###########################################################################
    def test_model(self):
        
        # Monitor total training time
        self.start_total_time = time.perf_counter()
                        
        # Set model to evaluate mode
        self.model.train(False)  
        
        # 
        running_corrects = 0
        
        # initialize outputs of self.test_model
        true_class = np.zeros((self.dataset_size,1), dtype=np.int64)-1
        pred_class = np.zeros((self.dataset_size,1), dtype=np.int64)-1
        pred_proba_class = np.zeros((self.dataset_size,self.params.getp("NB_CLASSES")), dtype=np.float32)
        
        # disable gradients
        with torch.set_grad_enabled(False):
            # Iterate over data.
            for i,data in enumerate(self.dset_loader):
                
                # get the inputs
                inputs = [d.cuda(device=self.params.getp("DEVICE_ID")) for d in data['input']]
                batch_size = inputs[0].size(0)
                                                        
                # forward
                outputs = self.model(inputs)
                if isinstance(outputs,list):
                    outputs = outputs[0]
                        
                # get predicted class
                _, preds = torch.max(outputs.data, 1)
                
                #
                preds = torch.squeeze(preds)
                
                if not(self.params.getp("USE_NO_LABELS")):
                    # get the labels
                    labels = data['label'].cuda(device=self.params.getp("DEVICE_ID"))
                    # statistics
                    batch_corrects = torch.sum(preds == labels.data)
                    running_corrects += batch_corrects
                    
                    #
                    true_class[i*self.params.getp("BATCH_SIZE"):i*self.params.getp("BATCH_SIZE") + batch_size] = labels.data.cpu().numpy().reshape( (batch_size,1) )
                    
                #
                pred_class[i*self.params.getp("BATCH_SIZE"):i*self.params.getp("BATCH_SIZE") + batch_size] = preds.cpu().numpy().reshape( (batch_size,1) )
                pred_proba_class[i*self.params.getp("BATCH_SIZE"):i*self.params.getp("BATCH_SIZE") + batch_size] = F.softmax(outputs,dim=1).data.cpu().numpy()
                
                if self.params.getp("USE_NO_LABELS"):
                    print("\r{:6.2f}%, Duration: {:.2f} s, Expected Total Duration: {:.1f} s".format(100*i/len(self.dset_loader), time.perf_counter() - self.start_total_time, len(self.dset_loader)/(i+1) * (time.perf_counter() - self.start_total_time)), end="")
                else:
                    print("\r{:6.2f}%, Batch Acc: {:6.2f}%, Duration: {:6.4f}s, Expected Total Duration: {:.1f} s".format(100*i/len(self.dset_loader), 100*batch_corrects/batch_size, time.perf_counter() - self.start_total_time, len(self.dset_loader)/(i+1) * (time.perf_counter() - self.start_total_time)), end="")
               
        if not(self.params.getp("USE_NO_LABELS")):
            accuracy = running_corrects / self.dataset_size
            
        time_elapsed = time.perf_counter() - self.start_total_time
        print('\n\nTesting complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        if not(self.params.getp("USE_NO_LABELS")):
            print('Testing Accuracy: {}'.format(accuracy))
        
        return (true_class, pred_class, pred_proba_class)
