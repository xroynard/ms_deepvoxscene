#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xavier Roynard
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

# to import modules
sys.path.insert(0, os.path.abspath('..'))
from utils.parameters import Parameters
    
if __name__ == '__main__':
    
###############################################################################
#%% Parses Command Line Arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-l", "--log_dir",
                        dest='LOG_DIR',
                        help="log directory of trained model (should contain config.yaml and subdir models/)",
                        )
    parser.print_help()
    args = parser.parse_args()
    
###############################################################################
#%% Read config file
    
    # Read config file (loads config file of trained model)
    training_params = Parameters(os.path.join(args.LOG_DIR, "config.yaml"))

###############################################################################
    ABSCISSE_COORDINATE = 0 # epoch 
#    ABSCISSE_COORDINATE = 1 # cpt_backward_pass
#    ABSCISSE_COORDINATE = 5 # time
        
###############################################################################
    nb_classes = training_params.getp("NB_CLASSES")
    
    LOG_DIR = training_params.getp("LOG_DIR")
    
    log_files = glob.glob( os.path.join(LOG_DIR,"*.txt") )
        
     # Training and Validation Loss and Accuracy
    print("\n 1st figure : Loss and Accuracy")
    fig1, (axes_loss,axes_acc) = plt.subplots(2,1)
    
    axes_loss.set_title("Loss")
    axes_loss.set_ylabel('Loss')
    axes_loss.set_yscale('log')
    axes_loss.set_xlabel('epoch' if ABSCISSE_COORDINATE==0 else ('number of samples' if ABSCISSE_COORDINATE==1 else 'time (s)'))
        
    axes_acc.set_title("Accuracy")
    axes_acc.set_ylabel('Accuracy')
    axes_acc.set_xlabel('epoch' if ABSCISSE_COORDINATE==0 else ('number of samples' if ABSCISSE_COORDINATE==1 else 'time (s)'))
        
    for f in log_files:
        fname = os.path.splitext(os.path.basename(f))[0]
        fold_id = fname.split("_")[2]
        sample_id = fname.split("_")[4]
        phase = "Train" if fname.split("_")[5]=="train" else "Val"
        
        log_data = np.loadtxt(f, delimiter=',',skiprows=1, ndmin=2)
        
        axes_loss.plot(log_data[:,ABSCISSE_COORDINATE], log_data[:,2], '.-', label="{} | Fold {} | Sample {} | {}".format(phase, fold_id, sample_id, training_params.getp("MODEL_NAME")))
        
        axes_acc.plot(log_data[:,ABSCISSE_COORDINATE], log_data[:,3], '.-', label="{} | Fold {} | Sample {} | {}".format(phase, fold_id, sample_id, training_params.getp("MODEL_NAME")))
        
    axes_loss.legend()
    axes_acc.legend()
            
###############################################################################
#%% TODO : Precision and Recall for each class
###############################################################################
#    # Precision and Recall for each class
#    print("\n 2nd figure : Precision and Recall for each class")
#    fig2, ((axes_train_precision,axes_val_precision),(axes_train_recall,axes_val_recall)) = plt.subplots(2,2)
#    
#    axes_train_precision.set_title("Training Precision")
#    axes_train_precision.set_ylim(0,100)
#    axes_train_precision.set_ylabel('Precision')
#    axes_train_precision.set_xlabel('epoch' if ABSCISSE_COORDINATE==0 else ('number of samples' if ABSCISSE_COORDINATE==1 else 'time (s)'))
#    
#    axes_train_recall.set_title("Training Recall")
#    axes_train_recall.set_ylim(0,100)
#    axes_train_recall.set_ylabel('Recall')
#    axes_train_recall.set_xlabel('epoch' if ABSCISSE_COORDINATE==0 else ('number of samples' if ABSCISSE_COORDINATE==1 else 'time (s)'))
#            
#    axes_val_precision.set_title("Validation Precision")
#    axes_val_precision.set_ylim(0,100) 
#    axes_val_precision.set_ylabel('Precision')
#    axes_val_precision.set_xlabel('epoch' if ABSCISSE_COORDINATE==0 else ('number of samples' if ABSCISSE_COORDINATE==1 else 'time (s)'))
#    
#    axes_val_recall.set_title("Validation Recall")
#    axes_val_recall.set_ylim(0,100)
#    axes_val_recall.set_ylabel('Recall')
#    axes_val_recall.set_xlabel('epoch' if ABSCISSE_COORDINATE==0 else ('number of samples' if ABSCISSE_COORDINATE==1 else 'time (s)'))
#    
#    for f in log_files:
#        fname = os.path.splitext(os.path.basename(f))[0]
#        fold_id = fname.split("_")[2]
#        sample_id = fname.split("_")[4]
#        phase = "Train" if fname.split("_")[5]=="train" else "Val"
#        
#        log_data = np.loadtxt(f, delimiter=',',skiprows=1, ndmin=2)
#        
#        for class_id in range(nb_classes):
#            
#            if phase=="Train":
#                axes_train_precision.plot(log_data[:,ABSCISSE_COORDINATE], log_data[:,6+class_id], '.-', label="{}".format(training_params.getp("CLASS_NAME_LIST")[class_id]))
#                
#                axes_train_recall.plot(log_data[:,ABSCISSE_COORDINATE], log_data[:,6+class_id+nb_classes], '.-', label="{}".format(training_params.getp("CLASS_NAME_LIST")[class_id]))
#            else:
#                axes_val_precision.plot(log_data[:,ABSCISSE_COORDINATE], log_data[:,6+class_id], '.-', label="{}".format(training_params.getp("CLASS_NAME_LIST")[class_id]))
#                
#                axes_val_recall.plot(log_data[:,ABSCISSE_COORDINATE], log_data[:,6+class_id+nb_classes], '.-', label="{}".format(training_params.getp("CLASS_NAME_LIST")[class_id]))
#                
#    axes_train_precision.legend()
#    axes_train_recall.legend()
#    axes_val_precision.legend()
#    axes_val_recall.legend()
      
###############################################################################
#%% TODO : Show mean accuracy over samples ...
###############################################################################               
#    MEAN_ACC[loc_MODEL_NAME]['train'] /= MEAN_ACC[loc_MODEL_NAME]['_nb']
#    if 'val' in MEAN_ACC[loc_MODEL_NAME]:
#        MEAN_ACC[loc_MODEL_NAME]['val'] /= MEAN_ACC[loc_MODEL_NAME]['_nb'] 
#        
#    print("\t plot figure {} : Mean Accuracy".format(-1))
#    plt.figure(-1)
#    
#    fig = gcf()
#    fig.suptitle("Mean Accuracy")
#    
#    plt.subplot(1,1,1)
#    for MODEL_NAME in MODEL_LIST:
#        plt.plot(MEAN_ACC[MODEL_NAME]['train'], '.-', label="Training {}".format(MODEL_NAME))
#        if 'val' in MEAN_ACC[MODEL_NAME]:
#            plt.plot(MEAN_ACC[MODEL_NAME]['val'], 'o-', label="Validation {}".format(MODEL_NAME))
#        plt.plot([0,len(MEAN_ACC[MODEL_NAME]['train'])], [80,80],'-g')
#    plt.ylabel('Mean Accuracy')
#    #        plt.yscale('log')    
#    plt.xlabel('epoch')
#    plt.ylim(0,100)
#    plt.legend()
#                            
#    
#                
#    print("-----------------------------------------------------------------------------------------------------------------------------------------")
#    print("| Fold |                   Model                  |         Phase        | _has_val | train loss |  train acc |  val loss  |   val acc  |")
#    print("-----------------------------------------------------------------------------------------------------------------------------------------")
#    for fold in FOLDS:
#        for MODEL_NAME in MODEL_LIST:
#            for phase in PRETRAIN_PHASES_LIST:
#                if (fold in RESULTS[MODEL_NAME]) and (phase in RESULTS[MODEL_NAME][fold]):
#                    print("| {:4d} | {:>40} | {:>20} | {:>8} |".format(fold,MODEL_NAME,phase,'True' if _has_val[MODEL_NAME][fold][phase] else 'False'), end="")
#                    for r in RESULTS[MODEL_NAME][fold][phase]:
#                        print("  {:9.2f} |".format(r), end="")
#                    print()
#        print("-----------------------------------------------------------------------------------------------------------------------------------------")
#    print("-----------------------------------------------------------------------------------------------------------------------------------------")
#        
#    for MODEL_NAME in MODEL_LIST:
#        try:
#            train_first_epoch = np.where(MEAN_ACC[MODEL_NAME]['train']>80)[0][0]
#        except IndexError:
#            train_first_epoch = -1
#        if 'val' in MEAN_ACC[MODEL_NAME]:
#            try:
#                val_first_epoch = np.where(MEAN_ACC[MODEL_NAME]['val']>80)[0][0]
#            except IndexError:
#                val_first_epoch = -1
#        else:
#            val_first_epoch = -1
#        print("| {:>4} | {:>40} | {:>20} | {:>8} |  {:>9} |  {:9.2f} |  {:>9} |  {:9.2f} |".format("Mean",
#                                                                                                   MODEL_NAME,
#                                                                                                   "",
#                                                                                                   "",
#                                                                                                   train_first_epoch,
#                                                                                                   np.max(MEAN_ACC[MODEL_NAME]['train']),
#                                                                                                   val_first_epoch,
#                                                                                                   np.max(MEAN_ACC[MODEL_NAME]['val']) if ('val'in MEAN_ACC[MODEL_NAME]) else 0
#                                                                                                   ))
#    print("-----------------------------------------------------------------------------------------------------------------------------------------")
            
    plt.show(block=True)
        
