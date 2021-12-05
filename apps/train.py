#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xavier Roynard
"""

#%% Imports
from typing import List, Optional

import hydra
from omegaconf import DictConfig

from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase

from src.utils import utils
from src.datamodule.pl_datamodule import LitDataModule
from src.netmodule.pl_netmodule import LitNetModule

import logging
log = logging.getLogger(__name__)


#%% Functions

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
              
    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        log.info("seed: {}".format(config.seed))
        seed_everything(config.seed, workers=True)
        
    # Init pl_datamodule
    log.info(f"Instantiating pl_datamodule: <{config.pl_datamodule._target_}>")
    pl_datamodule: LightningDataModule =  hydra.utils.instantiate(config.pl_datamodule)
    
    # Init pl_netmodule 
    log.info(f"Instantiating pl_netmodule: <{config.pl_netmodule._target_}>")
    pl_netmodule: LightningModule = hydra.utils.instantiate(config.pl_netmodule)    
            
    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback: <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger: <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))
                
    # Init pl_trainer
    log.info(f"Instantiate pl_trainer: <{config.trainer._target_}>")
    pl_trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger, _convert_="partial")
        
    # Train the model
    pl_trainer.fit(model=pl_netmodule, datamodule=pl_datamodule)
            
#%% Main Script
if __name__ == '__main__':
    main()