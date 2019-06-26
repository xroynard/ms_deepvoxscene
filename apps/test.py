#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xavier Roynard
"""

from __future__ import print_function, division

import os
import sys
import glob
import argparse
import numpy as np

# Pytorch
import torch
from torch.utils.data import DataLoader
# cudnn optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from sklearn.metrics import confusion_matrix

# to import modules
sys.path.insert(0, os.path.abspath('..'))
from input import PointCloudDataset
#import models
from utils.tester import Tester
from utils.parameters import Parameters

if __name__ == '__main__':
    
###############################################################################
#%% Parses Command Line Arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config",
                        dest='CONFIG_FILE',
                        default=os.path.join(os.path.curdir, os.path.pardir, "config", "test_config.yaml"),
                        help="config file",
                        )
    parser.add_argument("-l", "--log_dir",
                        dest='LOG_DIR',
                        help="log directory of trained model (should contain config.yaml and subdir models/)",
                        )
    parser.print_help()
    args = parser.parse_args()
    
##############################################################################
#%% Read config file
    
    # Read config file (loads config file of trained model)
    params = Parameters(os.path.join(args.LOG_DIR, "config.yaml"))
    params.update_parameters(args.CONFIG_FILE)
	    
    # Save parameters
    params.write_parameters()
    
###############################################################################
#%% Load model
    
    print()
    
    # Init model
    model = torch.load( os.path.join(params.getp("MODEL_DIR"), "best_train_model_checkpoint_fold_00_sample_000.tar" ) )
        
    # Puts the model on device (GPU)
    model = model.cuda(device=params.getp("DEVICE_ID"))
    
###############################################################################
#%% Read dataset repository
	
    # 
    filenames = sorted( glob.glob( os.path.join(params.getp("DATASET_DIR"),"test","*.ply") ) )
    
    print()
    print("###############################################################################")
    print("{} FOLDS in repo {}:".format(len(filenames), os.path.join(params.getp("DATASET_DIR"),"test") ))
    print("Test Files:")
    for indf,f in enumerate(filenames):
        print("\t - File {:02d} -> {}".format(indf,f))
    
###############################################################################
#%% Load the testing dataset
    
    
    for f in filenames:
        dirname,fname = os.path.split(f)
        name,ext = os.path.splitext(fname)
        
        dset = PointCloudDataset(f,
                                 scales=params.getp("SCALES"),
                                 grid_size=params.getp("GRID_SIZE"),
                                 voxel_size=params.getp("VOXEL_SIZE"),
                                 nb_classes=params.getp("NB_CLASSES"),
                                 testing=True,
                                 use_no_labels=params.getp("USE_NO_LABELS"),
                                 use_class0=(params.getp("DATASET")=="s3dis"),
                                 use_color=params.getp("USE_COLOR"),          
                                 use_reflectance=params.getp("USE_REFLECTANCE"), 
                                 )
        dset_loader = DataLoader(dset,
                                 batch_size=params.getp("BATCH_SIZE"),
                                 shuffle=False,
                                 num_workers=params.getp("NUM_WORKERS"),
                                 )#, pin_memory=True) #?
                    
        # Taille des dataset
        print("Dataset sizes    :", len(dset))
        print("Dataloaders sizes:", len(dset_loader))
         
###############################################################################       
#%% Test the model
        print()
        print("###############################################################################")
        print("#------------------------------ TEST THE MODEL -------------------------------#")
        print("###############################################################################")
              
        # Tester
        tester = Tester(params,
                        dset_loader,
                        model,
                        )
            
        # Test the model
        true_class, pred_class, pred_proba_class = tester.test_model()
        
###############################################################################
#%% Save cloud with predicted classes
        
        # Saves cloud with predicted class for each point
        RESULT_CLOUD_FILE = os.path.join(params.getp("CLOUD_DIR"), "classified_cloud_" + name + "_" + params.getp("MODEL_NAME") + ".ply")
        dset.write_pred_cloud(pred_class, RESULT_CLOUD_FILE)
        
        # Saves cloud with predicted class for each point and "probability" (output of softmax layer) of of belonging to in each class
        RESULT_CLOUD_FILE = os.path.join(params.getp("CLOUD_DIR"), "classified_cloud_with_proba_" + name + "_" + params.getp("MODEL_NAME") + ".ply")
        dset.write_pred_proba_cloud(pred_proba_class, RESULT_CLOUD_FILE)
        
###############################################################################
###############################################################################
###############################################################################
#%% Compute some stats
        if true_class.max() >= 0:
            # Compute Confusion Matrix
            C = confusion_matrix(true_class, pred_class, labels=np.arange(params.getp("NB_CLASSES")))
            
            print("Confusion Matrix:")
            for row in C:
                for col in row:
                    print("{:8d}".format(col),end='')
                print("")            
            
            ###############################################################################
            TP = np.diag(C) # True Positives
            FP = np.sum(C, axis=0) - TP # False Positives
            FN = np.sum(C, axis=1) - TP # False Negatives
            TN = np.sum(C) * np.ones(C.shape[0]) - TP - FP - FN # True Negatives
            
            ###############################################################################
            print("\tOverall Accuracy: {:6.2f}%".format(100 * np.sum(np.diag(C))/np.sum(C)))
            ###############################################################################
            print("Precision:")
            S1 = C / np.sum(C,axis=0,dtype=np.float64)
            for row in S1:
                for col in row:
                    print("{:6.2f}% ".format(100 * col),end='')
                print("")
            print("\tMean Precision: {:6.2f}%".format(100 * np.mean(np.diag(S1))))
            ###############################################################################
            print("Recall:")
            S2 = (C.transpose() / np.sum(C,axis=1,dtype=np.float64)).transpose() 
            for row in S2:
                for col in row:
                    print("{:6.2f}% ".format(100 * col),end='')
                print("")   
            print("\tMean Recall: {:6.2f}%".format(100 * np.mean(np.diag(S2))))
            ###############################################################################
            print("F1:")
            S3 = 2 * S1 * S2 / (S1 + S2 + 1e-8)
            for row in S3:
                for col in row:
                    print("{:6.2f}% ".format(100 * col),end='')
                print("") 
            print("\tMean F1: {:6.2f}%".format(100 * np.mean(np.diag(S3))))
            ###############################################################################
            print()
            ###############################################################################
            print("Precision:")
            P = (TP) / (TP+FP)
            for p in P:
                print("{:6.2f}% ".format(100 * p),end='')
            print()
            print("\tMean F1: {:6.2f}%".format(100 * np.mean(P)))
            ###############################################################################
            print("Recall:")
            R = (TP) / (TP+FN)
            for r in R:
                print("{:6.2f}% ".format(100 * r),end='')
            print()
            print("\tMean F1: {:6.2f}%".format(100 * np.mean(R)))
            ###############################################################################
            print("F1:")
            F1 = (2*TP) / (2*TP+FP+FN)
            for f1 in F1:
                print("{:6.2f}% ".format(100 * f1),end='')
            print()
            print("\tMean F1: {:6.2f}%".format(100 * np.mean(F1)))
            ###############################################################################    
            IoU = (TP)/(TP+FP+FN)
            print("IoU:")
            for iou in IoU:
                print("{:6.2f}% ".format(100 * iou),end='')
            print()
            print("\tMean IoU: {:6.2f}%".format(100 * np.mean(IoU)))