#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xavier Roynard
"""

from __future__ import print_function, division

import os
import glob
import sys
import argparse
import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
# cudnn optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# to import modules
sys.path.insert(0, os.path.abspath('..'))
from input import PointCloudDataset
import models
from utils.trainer import Trainer
from utils.parameters import Parameters

if __name__ == '__main__':
    
###############################################################################
#%% Parses Command Line Arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config",
                        dest='CONFIG_FILE',
                        default=os.path.join(os.path.curdir, os.path.pardir, "config", "train_config.yaml"),
#                        default=os.path.join(os.path.curdir, os.path.pardir, "config", "debug_config.yaml"),
                        help="config file",
                        )
    parser.print_help()
    args = parser.parse_args()

##############################################################################
#%% Read config file
    
    # Read config file
    params = Parameters(args.CONFIG_FILE)
	    
    # Save parameters
    params.write_parameters()
    
###############################################################################
#%% Build model
    
    # Init model
    model = models.models_dict[params.getp("MODEL_NAME")](nb_channels=params.getp("NB_CHANNELS"), nb_classes=params.getp("NB_CLASSES"), nb_scales=params.getp("NB_SCALES"))
    
    # Puts the model on device (GPU)
    model = model.cuda(device=params.getp("DEVICE_ID"))

    # Initialize weights of the  model
    model.init(params.getp("INITIALIZATION"))
                                
    # Defines Loss Function
    criterion = nn.CrossEntropyLoss()   
    
###############################################################################
#%% Read dataset repository
	
    # 
    filenames = sorted(glob.glob(os.path.join(params.getp("DATASET_DIR"),"train","*.ply")))
    
    print()
    print("###############################################################################")
    print("{} FOLDS in repo {}:".format(len(filenames), os.path.join(params.getp("DATASET_DIR"),"train") ))
    for f in filenames:
        print(" --", os.path.basename(f))
               
###############################################################################
#%% loop over folds
    for i_fold,f in enumerate(filenames):
        if ("val" in params.getp("PHASE_LIST")):
            print()
            print("###############################################################################")
            print("FOLD {} --> validation file : {}".format(i_fold, os.path.basename(f)))
                            
###############################################################################
#%% loop over samples
        for i_samp in range(params.getp("NB_SAMPLES")):
                        
            # only fold 0 in case there is no validation phase (all clouds are used for training)
            if not("val" in params.getp("PHASE_LIST")) and (i_fold != 0):
                continue
            
            if ("val" in params.getp("PHASE_LIST")):
                print()
                print("###############################################################################")
                print("FOLD {} --> validation file : {} | sample : {}".format(i_fold, os.path.basename(f), i_samp))
            else:
                assert(i_samp == 0)
            
            ###############################################################################
            # Initialize seeds 
            ###############################################################################
            np.random.seed(i_samp)
            torch.manual_seed(i_samp)
            torch.cuda.manual_seed_all(i_samp)
            
###############################################################################
#%% load datasets
                        
            files = {}
            if ("val" in params.getp("PHASE_LIST")):
                files['val'] = [f]
                files['train'] = []
                for g in filenames:
                    if not(g==f):
                        files['train'].append(g)
                if len(files['train'])==0:
                    files['val'] = []
                    files['train'] = [f]
                    params.setp("PHASE_LIST",['train'])
            else:
                files['train'] = filenames
                                
            # Build Dataset(s)
            dsets = {phase: ConcatDataset([PointCloudDataset(f,
                                                             scales=params.getp("SCALES"),
                                                             grid_size=params.getp("GRID_SIZE"),
                                                             voxel_size=params.getp("VOXEL_SIZE"),
                                                             nb_pts_per_class=(params.getp("NB_POINTS_PER_CLASS") if phase=='train' else len(files['train']) * (params.getp("NB_POINTS_PER_CLASS_VAL") or params.getp("NB_POINTS_PER_CLASS"))),
                                                             nb_classes=params.getp("NB_CLASSES"),
                                                             use_class0=(params.getp("DATASET")=="s3dis"),
                                                             use_color=params.getp("USE_COLOR"),          
                                                             use_reflectance=params.getp("USE_REFLECTANCE"),          
                                                             )
                                            for f in files[phase]]
                                            )
                    for phase in params.getp("PHASE_LIST")
                    }
            # Build DataLoader(s)
            dset_loaders = {phase: DataLoader(dsets[phase],
                                              batch_size=params.getp("BATCH_SIZE"),
                                              shuffle=True,
                                              num_workers=params.getp("NUM_WORKERS"),
                                              drop_last=True, # BatchNorm n'accepte pas de batch de taille 1 !
                                              pin_memory=True #?
                                              )
                            for phase in params.getp("PHASE_LIST")}
            
            params.setp("LOG_FILE_BASE", os.path.join(params.getp("LOG_DIR"), "run_fold_{:02d}_sample_{:03d}".format(i_fold,i_samp) ) )
            
            # Dataset size
            dset_sizes = {phase: len(dsets[phase]) for phase in params.getp("PHASE_LIST")}
            dset_loaders_sizes = {phase: len(dset_loaders[phase]) for phase in params.getp("PHASE_LIST")}
            print()
            print("Dataset sizes    :", dset_sizes)
            print("Dataloaders sizes:", dset_loaders_sizes)
     
###############################################################################       
#%% Train the model
            print()
            print("###############################################################################")
            print("#----------------------------- TRAIN THE MODEL -------------------------------#")
            print("###############################################################################")
                  
            # Trainer  
            trainer = Trainer(params,
                              dset_loaders,
                              model,
                              criterion,
                              )
                
            # Train the model
            best_val_model, best_train_model, model = trainer.train_model()