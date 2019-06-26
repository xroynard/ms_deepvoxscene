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

# Pytorch
import torch
import torch.optim as optim

# to compute confusion matrices
from sklearn.metrics import confusion_matrix

#
sys.path.insert(0, os.path.abspath('..'))
from utils.logger import Logger

class Trainer(object):
    
    ###########################################################################
    # Initializes Trainer
    ###########################################################################
    def __init__(self,
                 parameters,
                 dset_loaders,
                 model,
                 criterion,
                 ):
        
        #
        self.parameters = parameters
        self.dset_loaders = dset_loaders
        self.model = model
        self.criterion = criterion
        
        #
        self.best_val_model = copy.deepcopy(self.model)
        self.best_train_model = copy.deepcopy(self.model)
        self.best_acc = {phase:0.0 for phase in self.parameters.getp("PHASE_LIST")}
        
        # Additionnal parameters
        self.validation = ('val' in self.parameters.getp("PHASE_LIST"))
        
        # Logger (Accuracy, Loss, Precision, Recall and Confusion Matrix)
        self.logger = Logger( self.parameters.getp("LOG_FILE_BASE"), self.validation, self.parameters.getp("NB_CLASSES") )        
                
        # Setting the Optimizer
        if self.parameters.getp("OPTIMIZER") == "Adam":
            self.optimizer = optim.Adam(model.parameters(),
                                        lr=self.parameters.getp("LR"),
                                        eps=self.parameters.getp("EPS"),
                                        weight_decay=self.parameters.getp("WD"))
        elif self.parameters.getp("OPTIMIZER") == "Adadelta":
            self.optimizer = optim.Adadelta(model.parameters(),
                                            lr=self.parameters.getp("LR"),
                                            eps=self.parameters.getp("EPS"),
                                            weight_decay=self.parameters.getp("WD"))
        elif self.parameters.getp("OPTIMIZER") == "RMSprop":
            self.optimizer = optim.RMSprop(model.parameters(),
                                           lr=self.parameters.getp("LR"),
                                           weight_decay=self.parameters.getp("WD"))
        elif self.parameters.getp("OPTIMIZER") == "SGD":
            self.optimizer = optim.SGD(model.parameters(),
                                       lr=self.parameters.getp("LR"),
                                       weight_decay=self.parameters.getp("WD"),
                                       momentum=0.9,
                                       nesterov=True)
        else:
            self.parameters.setp("OPTIMIZER","Adam")
            print("OPTIMIZER not specified --> set to {}".format(self.parameters.getp("OPTIMIZER")))
            self.optimizer = optim.Adam(model.parameters(),
                                        lr=self.parameters.getp("LR"),
                                        eps=self.parameters.getp("EPS"),
                                        weight_decay=self.parameters.getp("WD"))
                
        # Setting the Learning Rate Scheduler
        # except for None, lr_scheduler parameters are set such as, at last epoch, lr = 1e-6        
        if self.parameters.getp("LR_SCHEDULER") == None:
            self.lr_scheduler = None
        elif self.parameters.getp("LR_SCHEDULER") == "ExponentialLR":
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=np.power( 1e-6/self.parameters.getp("LR") , 1.0/(self.parameters.getp("NUM_EPOCHS")-1) ))
        elif self.parameters.getp("LR_SCHEDULER") == "CosineAnnealingLR":
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.parameters.getp("NUM_EPOCHS"), eta_min=1e-6)
        else:
            print("LR_SCHEDULER:{} doesn't exist --> set to None".format(self.parameters.getp("LR_SCHEDULER")))
            self.lr_scheduler = None
             
    ###########################################################################
    # Saves the current model and best training and validation models
    ###########################################################################
    def save_models(self):
        fold_sample_ids = "_".join(self.parameters.getp("LOG_FILE_BASE").split('_')[-4:])
        torch.save(self.best_val_model   , os.path.join(self.parameters.getp("MODEL_DIR"),   "best_val_model_checkpoint_{}.tar".format(fold_sample_ids)) )
        torch.save(self.best_train_model , os.path.join(self.parameters.getp("MODEL_DIR"), "best_train_model_checkpoint_{}.tar".format(fold_sample_ids)) )
        torch.save(self.model            , os.path.join(self.parameters.getp("MODEL_DIR"),            "model_checkpoint_{}.tar".format(fold_sample_ids)) )

    ###########################################################################
    # Trains self.model
    ###########################################################################
    def train_model(self):
        
        # Monitor total training time
        self.start_total_time = time.perf_counter()        
        
        # counters
        self.nb_processed_samples = 0
    
        # try catches KeyboardInterrupt if the user wants to stop prematurely the training
        try:
            # loop over epochs
            for epoch in range(self.parameters.getp("NUM_EPOCHS")):
                
                # trains for one epoch
                self.train_one_epoch(epoch)                
                
        # Catch exception if script is stopped by user
        except KeyboardInterrupt:
            pass
           
        # Show total training time
        elapsed_time = time.perf_counter() - self.start_total_time
        print('\n\nTraining complete in {:.0f}h {:.0f}m {:.0f}s'.format(elapsed_time // 3600, (elapsed_time % 3600) // 60, elapsed_time % 60))
        
        # Show best accuracy
        print('Best Acc: {}'.format(self.best_acc))
        
        return self.best_val_model, self.best_train_model, self.model
    
    ###########################################################################
    # Trains for one epoch (includes training and validation phases)
    ###########################################################################
    def train_one_epoch(self, epoch):
        print('\n\n### Epoch {}/{} ###'.format(epoch, self.parameters.getp("NUM_EPOCHS") - 1))
                                                
        # 
        self.epoch_loss = {}
        self.epoch_acc  = {}
        self.epoch_time = {}
        self.C = {phase:np.zeros((self.parameters.getp("NB_CLASSES"),self.parameters.getp("NB_CLASSES")), dtype=np.int64) for phase in self.parameters.getp("PHASE_LIST")}                
        
        # Randomly sample training points in each dataset for current epoch
        for ds in self.dset_loaders['train'].dataset.datasets:
            ds.randomize_samples()
            
        # Set learning rate for current epoch
        if self.lr_scheduler:
            self.lr_scheduler.step(epoch=epoch)
            
        # loop over phases ('train' and 'val')
        for phase in self.parameters.getp("PHASE_LIST"):
            
            # 
            self.model.train(phase == 'train')
            
            # 
            running_loss = 0.0
            running_corrects = 0
            nb_pts_epoch = 0
            
            # Monitor phase time
            epoch_start_time = time.perf_counter()
            
            # Passe en mode train (garde les gradients) ou self.validation (pas de gradients)
            with torch.set_grad_enabled(phase == 'train'):
                # Monitor one iteration time
                start_time = time.perf_counter()
                
                # Iterate over data
                for i,data_batch in enumerate(self.dset_loaders[phase]):
                        
                    # get the inputs and put them on device (GPU)
                    inputs = [d.cuda(device=self.parameters.getp("DEVICE_ID"), non_blocking=True) for d in data_batch['input']]
                    batch_size = inputs[0].size(0)
                    
                    if phase=="train":
                        for d in inputs:
                            d.requires_grad_() 
                                
                    # get the labels and put them on device (GPU)
                    labels = data_batch['label'].cuda(device=self.parameters.getp("DEVICE_ID"), non_blocking=True)
                    
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
    
                    # inference (forward pass)
                    outputs = self.model(inputs)
                        
                    # compute loss
                    loss = self.criterion(outputs, labels)
    
                    # backward + optimize (only in training phase)
                    if phase == 'train':
                        # compute gradients (backward pass)
                        loss.backward()
                        
                        if self.parameters.getp("GRADIENT_CLIPPING"):
                            # Gradient Clipping
                            for m in self.model.parameters():
                                if not (m.grad is None):
                                    if m.grad.norm()>1:
                                        m.grad = m.grad / (m.grad.norm() + 1e-8)    
                        
                        # gradient descent step
                        self.optimizer.step()
                                          
                        # update 
                        self.nb_processed_samples += batch_size  
                        
                    # Batch Statistics To Monitor Learning in Real-Time
                    batch_loss = loss.item() * batch_size
                    running_loss += batch_loss
                    
                    nb_pts_epoch += batch_size
                    true_class = labels.data                            
                    _, pred_class = torch.max(outputs.data, 1)
                    pred_class = torch.squeeze(pred_class)
                    
                    batch_corrects = torch.sum(pred_class == true_class).item()
                    running_corrects += batch_corrects
                    
                    # Compute accuracy confusion matrix
                    self.C[phase] += confusion_matrix(true_class.cpu().numpy().flatten(), pred_class.cpu().numpy().flatten(), labels=np.arange(self.parameters.getp("NB_CLASSES")))
                    
                    # Monitor batch accuracy and loss...
                    print("\r{:>5} -> {:6.2f}%, Loss : {:7.3f}, Acc : {:7.3f}%. [Batch: {:.3f}s, Epoch: {:.1f}s, Total: {:.1f}s]".format(phase, 100*i/len(self.dset_loaders[phase]), batch_loss, 100*batch_corrects/batch_size, time.perf_counter() - start_time, time.perf_counter() - epoch_start_time, time.perf_counter() - self.start_total_time), end="")
                                      
                    # start new iteration time
                    start_time = time.perf_counter()
                        
###################% End of phase (train or val)
                print("")
###########% End of with torch.set_grad_enabled(phase == 'train'):
                                        
            self.epoch_loss[phase] = running_loss / len(self.dset_loaders[phase].dataset)
            self.epoch_acc[phase]  = running_corrects / nb_pts_epoch#len(self.dset_loaders[phase].dataset)
            self.epoch_time[phase] = time.perf_counter() - epoch_start_time            
            
            # Log epoch Accuracy, Loss, Precision, Recall and Confusion Matrix
            str_line = "{:d},{:d},{:.8f},{:.8f},{:.8f},{:.4f}".format(epoch,self.nb_processed_samples,self.epoch_loss[phase],100*self.epoch_acc[phase],self.lr_scheduler.get_lr()[0] if self.lr_scheduler else 1.0,time.perf_counter() - self.start_total_time)
            for c in 100*(self.C[phase].diagonal() / self.C[phase].sum(axis=0)):
                str_line = str_line + ",{:.8f}".format(c)
            for c in 100*(self.C[phase].diagonal() / self.C[phase].sum(axis=1)):
                str_line = str_line + ",{:.8f}".format(c)
            for row in self.C[phase]:   
                for col in row:
                    str_line = str_line + ",{}".format(col)
            str_line = str_line + "\n"
            self.logger.write(str_line, phase)
            
            # deep copy self.model
            if self.epoch_acc[phase] > self.best_acc[phase]:
                self.best_acc[phase] = self.epoch_acc[phase]
                if phase == 'val':
                    self.best_val_model = copy.deepcopy(self.model)
                if phase == 'train':
                    self.best_train_model = copy.deepcopy(self.model)
                    
            # Saves models
            self.save_models()
                    
#######% End of an epoch (includes training and validation phases)
                     
        # Show confusion matrices of Training and Validation 
        print()
        for phase in self.parameters.getp("PHASE_LIST"):
            print("Phase : {:>5} --> Loss: {:.4f}, Acc: {:.2f}%. Epoch took {:6.3f}s".format(phase, self.epoch_loss[phase], 100*self.epoch_acc[phase], self.epoch_time[phase]))
        print("Confusion Matrix Training:")
        for row in self.C['train']:
            for col in row:
                print("{:10d}".format(col),end='')
            print("")
        if self.validation:
            print("Confusion Matrix Validation:")
            for row in self.C['val']:
                for col in row:
                    print("{:10d}".format(col),end='')
                print("")