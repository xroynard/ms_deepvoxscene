#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xavier Roynard
"""

#%% Imports
import os
import sys
import glob
import time

import numpy as np

from sklearn.metrics import confusion_matrix

import torch

from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.utilities import rank_zero_only

#%% Functions

#%% Classes

class ConfusionMatrix(Callback):
    """Log confusion matrix."""

    def __init__(self,
                 nb_classes):
        self.nb_classes = nb_classes

    @rank_zero_only
    def on_epoch_start(self, pl_trainer:Trainer, pl_netmodule:LightningModule):
        self.C = {phase:np.zeros((self.nb_classes, self.nb_classes), dtype=np.int64) for phase in ["val", "train"]}

    @rank_zero_only
    def on_train_batch_end(self, trainer:Trainer, pl_netmodule:LightningModule, outputs, batch, batch_idx: int) -> None:
        
        true_class = batch['label'].data                            
        _, pred_class = torch.max(outputs.data, 1)
        pred_class = torch.squeeze(pred_class)
                    
        pred_class = outputs.cpu().detach().numpy().flatten()
        true_class = true_class.cpu().detach().numpy().flatten()
        self.C["train"] += confusion_matrix(true_class, pred_class, labels=np.arange(self.nb_classes))
        
    @rank_zero_only
    def on_validation_batch_end(self, trainer:Trainer, pl_netmodule:LightningModule, outputs, batch, batch_idx: int) -> None:
        
        true_class = batch['label'].data                            
        _, pred_class = torch.max(outputs.data, 1)
        pred_class = torch.squeeze(pred_class)
                    
        pred_class = outputs.cpu().detach().numpy().flatten()
        true_class = true_class.cpu().detach().numpy().flatten()
        self.C["val"] += confusion_matrix(true_class, pred_class, labels=np.arange(self.nb_classes))
        
    @rank_zero_only
    def on_train_epoch_end(self, pl_trainer:Trainer, pl_netmodule:LightningModule) -> None:
        
        column_normalized_C = 100*(self.C["train"].diagonal() / self.C["train"].sum(axis=0))
        row_normalized_C = 100*(self.C["train"].diagonal() / self.C["train"].sum(axis=1))
        
        pl_netmodule.log("precisions/train", column_normalized_C)
        pl_netmodule.log("recalls/train", row_normalized_C)
        pl_netmodule.log("confusion_matrix/train", self.C["train"])
        
    @rank_zero_only
    def on_validation_epoch_end(self, pl_trainer:Trainer, pl_netmodule:LightningModule) -> None:
        
        column_normalized_C = 100*(self.C["val"].diagonal() / self.C["val"].sum(axis=0))
        row_normalized_C = 100*(self.C["val"].diagonal() / self.C["val"].sum(axis=1))
        
        pl_netmodule.log("precisions/val", column_normalized_C)
        pl_netmodule.log("recalls/val", row_normalized_C)
        pl_netmodule.log("confusion_matrix/val", self.C["val"])
        
        
#%% Tests

#%% Main Script
if __name__ == '__main__':
    pass