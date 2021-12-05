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

from typing import Any, List

import hydra
from omegaconf import DictConfig

import torch
from pytorch_lightning import LightningModule

#%% Functions

#%% Classes

class LitNetModule(LightningModule):
    def __init__(self,
                 network:DictConfig,
                 optimizer:DictConfig,
                 loss:DictConfig,
                 lr_scheduler:DictConfig,
                 config:DictConfig,
                 ):
        super().__init__()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        self.network = hydra.utils.instantiate(network)
        self.loss_function = hydra.utils.instantiate(loss)
        
        
        
    def training_step(self, batch:Any, batch_idx:int) -> dict:
        input = batch[0]
        target = batch[1]
        
        loss = self.loss_function(self.network(input, target))
        
        self.log("loss/train", loss.item.detach(), prog_bar=True, on_epoch=True)
        
        return {"loss": loss}
    
    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer, params=self.network.parameters())
        output = {"optimizer": optimizer,
                  "lr_scheduler": {"scheduler": hydra.utils.instantiate(self.lr_scheduler, optimizer),
                                   "interval": "epoch",
                                   "frequency": 1,
                                   }
                  }
        return output

#%% Tests

#%% Main Script
if __name__ == '__main__':
    pass