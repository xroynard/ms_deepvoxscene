#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xavier Roynard
"""

from __future__ import print_function, division

import os
import sys
import time
import numpy as np

from torch.autograd import Variable
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath('..'))
from input import PointCloudDataset

#%% Test PointCloudDataset
if __name__ == '__main__':
    
###############################################################################    
    DATASET = "parislille3d"
#    DATASET = "semantic3d"
#    DATASET = "s3dis"    
        
############################################################################### 
#    DATA_DIR = os.path.join(os.path.curdir, "data", DATASET)    
    DATA_DIR = os.path.join(os.path.curdir, "../data", DATASET, "debug_datasets")
    DATASET_DIR = os.path.join(DATA_DIR,'train')
    RESULT_DIR = os.path.join(DATA_DIR,'result')
    
    GRID_SIZE = 32
    VOXEL_SIZE = 0.1
    SCALES = {1}
#    SCALES = {1,2,4}
    BATCH_SIZE = 20
    NB_CLASSES = 9
    NUM_WORKERS = 16
    NB_POINTS_PER_CLASS = 100
    NB_POINTS_TO_ADD_PER_CLASS = 100
#    NB_POINTS_PER_CLASS = 3600
#    NB_POINTS_TO_ADD_PER_CLASS = 1000
#    SEGMENTATION = True
    SEGMENTATION = False
    VOXELIZED = True
#    VOXELIZED = False
        
    dset = PointCloudDataset(DATASET_DIR,
                             grid_size=GRID_SIZE,
                             voxel_size=VOXEL_SIZE,
                             scales=SCALES,
#                             transform=voxel_augmentation,
                             nb_pts_per_class=NB_POINTS_PER_CLASS,
                             nb_pts_to_add_per_class=NB_POINTS_TO_ADD_PER_CLASS,
                             voxelized=VOXELIZED,
                             segmentation=SEGMENTATION,
#                             inRAM=True,
                             denseGrid=True
                             )

    dset_loader = DataLoader(dset,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=NUM_WORKERS,
                             collate_fn=default_collate if VOXELIZED else pc_collate
                             )#, pin_memory=True) #?
        
#    city_list = ["Lille1_1", "Lille1_2", "Lille2", "Paris"]
#    if VOXELIZED:
#        dsets = {city: PointCloudDataset(os.path.join(data_dir,city), r=1.6, transform=voxel_augmentation, voxelized=VOXELIZED, segmentation=SEGMENTATION, device=0)
#            for city in city_list}
#        dset_loaders = {city: torch.utils.data.DataLoader(dsets[city], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)#, pin_memory=True) #?
#            for city in city_list}
#    else:
#        dsets = {city: PointCloudDataset(os.path.join(data_dir,city), r=1.6, transform=pc_augmentation, voxelized=VOXELIZED, segmentation=SEGMENTATION, device=0)
#            for city in city_list}
#        dset_loaders = {city: torch.utils.data.DataLoader(dsets[city], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=pc_collate)#, pin_memory=True) #?
#            for city in city_list}
#        
#    # Taille des dataset
#    dset_sizes = {city: len(dsets[city]) for city in city_list}
#    dset_loaders_sizes = {city: len(dset_loaders[city]) for city in city_list}
#    print("dset_sizes         :", dset_sizes)
#    print("dset_loaders_sizes :", dset_loaders_sizes)
#
#    # Test le data loader
#    for city in city_list:        
#        print("\n\nCity : {}".format(city))
#        start_time = time.time()
#        dataset_len = len(dsets[city])
#        dataloader_len = len(dset_loaders[city])
#        print("Taille Dataset    : {}".format(dataset_len))
#        print("Taille DataLoader : {}".format(dataloader_len))
#        print("Taille Batches    : {}".format(BATCH_SIZE))
#        for i,data in enumerate(dset_loaders[city]):
#            inputs = Variable(data['input']).cuda()
#            labels = Variable(data['label']).cuda()
#            print("\r\tDurée epoch:{:07.2f} s,{:05.2f}%,{:08d},input:{},labels:{}".format(time.time() - start_time, 100*i*BATCH_SIZE/dataset_len, i*BATCH_SIZE, inputs.size(), labels.size()), end="")
#            sys.stdout.flush()

##%% Test le data loader
#    print("\n\n")
#    dataset_len = len(dset)
#    dataloader_len = len(dset_loader)
#    print("Taille Dataset    : {}".format(dataset_len))
#    print("Taille DataLoader : {}".format(dataloader_len))
#    print("Taille Batches    : {}".format(BATCH_SIZE))
#
#    start_time = time.time()
#    for i,data in enumerate(dset_loader):
#        inputs = data['input']                
#        if isinstance(inputs,list):
#            inputs = Variable(inputs[0]).cuda()
#        else:
#            inputs = Variable(inputs).cuda()
#        labels = Variable(data['label']).cuda()
#        print("\r\tDurée epoch:{:07.2f} s,{:05.2f}%,{:08d},input:{},labels:{}".format(time.time() - start_time, 100*i*BATCH_SIZE/dataset_len, i*BATCH_SIZE, inputs.size(), labels.size()), end="")
#        sys.stdout.flush()
#        
#    pred_class = np.random.randint(0,high=NB_CLASSES,size=(len(dset),1))
#    pred_proba_class = np.random.rand(len(dset),NB_CLASSES)
#    
#    dset.write_training_points( os.path.join(RESULT_DIR, "inputDEBUG_write_training_points.ply") )
#    dset.write_pred_cloud( pred_class , os.path.join(RESULT_DIR, "inputDEBUG_write_pred_cloud.ply") )
#    dset.write_pred_proba_cloud( pred_proba_class , os.path.join(RESULT_DIR, "inputDEBUG_write_pred_proba_cloud.ply") )
#    
#    dset.new_training_points(pred_proba_class)
#           
################################################################################
################################################################################
################################################################################
#    print("\n\n")
#    dset.active() # 0 -> 1
#    dset.active() # 1 -> 0
#    dset.active() # 0 -> 1
#    dset.active(False) # 1 -> 0
#    dset.active(False) # 0 -> 0
#    dset.active(False) # 0 -> 0
#    dset.active(True) # 0 -> 1
#    dset.active(True) # 1 -> 1
#    dset.active(True) # 1 -> 1
#    print("Active Learning --> dataset_len", len(dset))
#    
#    dataset_len = len(dset)
#    dataloader_len = len(dset_loader)
#    print("Taille Dataset    : {}".format(dataset_len))
#    print("Taille DataLoader : {}".format(dataloader_len))
#    print("Taille Batches    : {}".format(BATCH_SIZE))
#    
#    start_time = time.time()
#    for i,data in enumerate(dset_loader):
#        inputs = data['input']                
#        if isinstance(inputs,list):
#            inputs = Variable(inputs[0]).cuda()
#        else:
#            inputs = Variable(inputs).cuda()
#        labels = Variable(data['label']).cuda()
#        print("\r\tDurée epoch:{:07.2f} s,{:05.2f}%,{:08d},input:{},labels:{}".format(time.time() - start_time, 100*i*BATCH_SIZE/dataset_len, i*BATCH_SIZE, inputs.size(), labels.size()), end="")
#        sys.stdout.flush()
#    
#    pred_class = np.random.randint(0,high=NB_CLASSES,size=(len(dset),1))
#    pred_proba_class = np.random.rand(NB_CLASSES,len(dset))
#    pred_proba_class = (pred_proba_class / np.sum(pred_proba_class, axis=0)).transpose()
#    
#    dset.write_training_points( os.path.join(RESULT_DIR, "inputDEBUG_write_training_points_active.ply") )
#    dset.write_pred_cloud( pred_class , os.path.join(RESULT_DIR, "inputDEBUG_write_pred_cloud_test_active.ply") )
#    dset.write_pred_proba_cloud( pred_proba_class , os.path.join(RESULT_DIR, "inputDEBUG_write_pred_proba_cloud_test_active.ply") )
#    
#    for _ in range(3):
#        print("\n")
#        pred_proba_class = np.random.rand(NB_CLASSES,len(dset))
#        pred_proba_class = (pred_proba_class / np.sum(pred_proba_class, axis=0)).transpose()
#        dset.new_training_points(pred_proba_class)
#        print("Training --> dataset_len", len(dset))
#        dset.active(True) # Pour continuer à être en active learning ET reprendre des active_points...
#            
################################################################################
################################################################################
################################################################################
#    print("\n\n")
#    dset.active(False) 
#    dset.test_grid(False)
#    dset.test(False)
#    print("Training --> dataset_len", len(dset))
#    
#    dataset_len = len(dset)
#    dataloader_len = len(dset_loader)
#    print("Taille Dataset    : {}".format(dataset_len))
#    print("Taille DataLoader : {}".format(dataloader_len))
#    print("Taille Batches    : {}".format(BATCH_SIZE))
#
#    start_time = time.time()
#    for i,data in enumerate(dset_loader):
#        inputs = data['input']                
#        if isinstance(inputs,list):
#            inputs = Variable(inputs[0]).cuda()
#        else:
#            inputs = Variable(inputs).cuda()
#        labels = Variable(data['label']).cuda()
#        print("\r\tDurée epoch:{:07.2f} s,{:05.2f}%,{:08d},input:{},labels:{}".format(time.time() - start_time, 100*i*BATCH_SIZE/dataset_len, i*BATCH_SIZE, inputs.size(), labels.size()), end="")
#        sys.stdout.flush()
#        
#    pred_class = np.random.randint(0,high=NB_CLASSES,size=(len(dset),1))
#    pred_proba_class = np.random.rand(len(dset),NB_CLASSES)
#    
#    dset.write_training_points( os.path.join(RESULT_DIR, "inputDEBUG_write_training_points.ply") )
#    dset.write_pred_cloud( pred_class , os.path.join(RESULT_DIR, "inputDEBUG_write_pred_cloud.ply") )
#    dset.write_pred_proba_cloud( pred_proba_class , os.path.join(RESULT_DIR, "inputDEBUG_write_pred_proba_cloud.ply") )
#    
#    dset.new_training_points(pred_proba_class)
           
###############################################################################
###############################################################################
###############################################################################
    print("\n\n")
    dset.test_grid() # 0 -> 1
#    dset.test_grid() # 1 -> 0
#    dset.test_grid() # 0 -> 1
#    dset.test_grid(False) # 1 -> 0
#    dset.test_grid(False) # 0 -> 0
#    dset.test_grid(False) # 0 -> 0
#    dset.test_grid(True) # 0 -> 1
#    dset.test_grid(True) # 1 -> 1
#    dset.test_grid(True) # 1 -> 1
    print("Testing on Grid --> dataset_len", len(dset))
    
    dataset_len = len(dset)
    dataloader_len = len(dset_loader)
    print("Taille Dataset    : {}".format(dataset_len))
    print("Taille DataLoader : {}".format(dataloader_len))
    print("Taille Batches    : {}".format(BATCH_SIZE))
    
    start_time = time.time()
    for i,data in enumerate(dset_loader):
        inputs = data['input']                
        if isinstance(inputs,list):
            inputs = Variable(inputs[0]).cuda()
        else:
            inputs = Variable(inputs).cuda()
        labels = Variable(data['label']).cuda()
        print("\r\tDurée epoch:{:07.2f} s,{:05.2f}%,{:08d},input:{},labels:{}".format(time.time() - start_time, 100*i*BATCH_SIZE/dataset_len, i*BATCH_SIZE, inputs.size(), labels.size()), end="")
        sys.stdout.flush()
    
    pred_class = np.random.randint(0,high=NB_CLASSES,size=(len(dset),1))
    pred_proba_class = np.random.rand(len(dset),NB_CLASSES)
    
    dset.write_training_points( os.path.join(RESULT_DIR, "inputDEBUG_write_training_points_test_grid.ply") )
    dset.write_pred_cloud( pred_class , os.path.join(RESULT_DIR, "inputDEBUG_write_pred_cloud_test_grid.ply") )
    dset.write_pred_proba_cloud( pred_proba_class , os.path.join(RESULT_DIR, "inputDEBUG_write_pred_proba_cloud_test_grid.ply") )
    
    dset.new_training_points(pred_proba_class)
            
###############################################################################
###############################################################################
###############################################################################
    print("\n\n")
    dset.test() # 0 -> 1
    dset.test() # 1 -> 0
    dset.test() # 0 -> 1
    dset.test(False) # 1 -> 0
    dset.test(False) # 0 -> 0
    dset.test(False) # 0 -> 0
    dset.test(True) # 0 -> 1
    dset.test(True) # 1 -> 1
    dset.test(True) # 1 -> 1
    print("Testing --> dataset_len", len(dset))
    
    dataset_len = len(dset)
    dataloader_len = len(dset_loader)
    print("Taille Dataset    : {}".format(dataset_len))
    print("Taille DataLoader : {}".format(dataloader_len))
    print("Taille Batches    : {}".format(BATCH_SIZE))
    
    start_time = time.time()
    for i,data in enumerate(dset_loader):
        inputs = data['input']                
        if isinstance(inputs,list):
            inputs = Variable(inputs[0]).cuda()
        else:
            inputs = Variable(inputs).cuda()
        labels = Variable(data['label']).cuda()
        print("\r\tDurée epoch:{:07.2f} s,{:05.2f}%,{:08d},input:{},labels:{}".format(time.time() - start_time, 100*i*BATCH_SIZE/dataset_len, i*BATCH_SIZE, inputs.size(), labels.size()), end="")
        sys.stdout.flush()
    
    pred_class = np.random.randint(0,high=NB_CLASSES,size=(len(dset),1))
    pred_proba_class = np.random.rand(len(dset),NB_CLASSES)
    
    dset.write_training_points( os.path.join(RESULT_DIR, "inputDEBUG_write_training_points_test.ply") )
    dset.write_pred_cloud( pred_class , os.path.join(RESULT_DIR, "inputDEBUG_write_pred_cloud_test.ply") )
    dset.write_pred_proba_cloud( pred_proba_class , os.path.join(RESULT_DIR, "inputDEBUG_write_pred_proba_cloud_test.ply") )
    
    dset.new_training_points(pred_proba_class)
