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

from typing import Optional, Tuple

import hydra
from omegaconf import DictConfig

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from src.datamodule.dataset import PointCloudDataset

import logging
log = logging.getLogger(__name__)

#%% Functions

#%% Classes

class LitDataModule(LightningDataModule):
    def __init__(self,
                 dataset:DictConfig,
                 config:DictConfig,
                 ):
        super().__init__()
        
        self.dataset = dataset
        self.config = config
        
    def prepare_data(self) -> None:
        pass
    
    def setup(self, stage: Optional[str] = None) -> None:

        log.info(f"datadir: {self.config.datadir}")
        
        train_fnames = sorted(glob.glob(os.path.join(self.config.datadir,"train","*.ply")))
        test_fnames = sorted(glob.glob(os.path.join(self.config.datadir,"test","*.ply")))
        
        log.info(f"fold_id: {self.config.fold_id}, len(train_fnames): {len(train_fnames)}")
        val_fnames = [train_fnames[self.config.fold_id]]
        train_fnames = [fname for fname in train_fnames if not(fname in val_fnames)]
        predict_fnames = test_fnames
        
        log.info(f"len(train_fnames): {len(train_fnames)}, len(val_fnames): {len(val_fnames)}, len(test_fnames): {len(test_fnames)}, len(predict_fnames): {len(predict_fnames)}")
        
        for f in train_fnames:
            log.info(f"Instantiating dataset: <{self.dataset._target_}>")
        self.train_dataset = ConcatDataset(
            [hydra.utils.instantiate(self.dataset, f) for f in train_fnames]
            )
        
        self.val_dataset = ConcatDataset(
            [hydra.utils.instantiate(self.dataset, f) for f in val_fnames]
            )
        
        # self.test_dataset = ConcatDataset(
        #     [hydra.utils.instantiate(self.dataset, f, use_no_labels=True, testing=True) for f in test_fnames]
        #     )
        
        # self.predict_dataset = ConcatDataset(
        #     [hydra.utils.instantiate(self.dataset, f, use_no_labels=True, testing=True) for f in predict_fnames]
        #     )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.config.batch_size,
                          shuffle=True,
                          num_workers=self.config.num_workers,
                          drop_last=True, # BatchNorm is not OK with size-1 batch
                          pin_memory=True
                          )
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.config.batch_size,
                          shuffle=False,
                          num_workers=self.config.num_workers,
                          drop_last=True, # BatchNorm is not OK with size-1 batch
                          pin_memory=True
                          )
    
    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset,
    #                       batch_size=self.config.batch_size,
    #                       shuffle=False,
    #                       num_workers=self.config.num_workers,
    #                       drop_last=True, # BatchNorm is not OK with size-1 batch
    #                       pin_memory=True
    #                       )
    
    # def predict_dataloader(self):
    #     return DataLoader(self.predict_dataset,
    #                       batch_size=self.config.batch_size,
    #                       shuffle=False,
    #                       num_workers=self.config.num_workers,
    #                       drop_last=True, # BatchNorm is not OK with size-1 batch
    #                       pin_memory=True
    #                       )
        
#%% Tests

#%% Main Script
if __name__ == '__main__':
    pass