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

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities import rank_zero_only

#%% Functions

#%% Classes

class RandomizeDataset(Callback):
    """Sample new points in each dataset."""

    def __init__(self):
        pass

    @rank_zero_only
    def on_epoch_start(self, pl_trainer:Trainer, pl_netmodule):
        for ds in pl_trainer.train_dataloader.dataset.datasets:
            ds.randomize_samples()
        

#%% Tests

#%% Main Script
if __name__ == '__main__':
    pass