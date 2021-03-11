# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 18:47:29 2021

@author: Xavier Roynard
"""

#%% Imports

import os
import sys
import time
import glob
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


#%%
DEBUG = True
# DEBUG = False

MAX_DEPTH = 21 # 3*21 = 63 <= 64

#%% Functions
        
### method to seperate bits from a given integer 3 positions apart
# inspired from <https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/>
# torch.Tensor as input and output
def splitBy3(n):
    # we only look at the first 21 bits:AND 0b0000000000000000000000000000000000000000000111111111111111111111
    n =             n & 0b0000000000000000000000000000000000000000000111111111111111111111
    # shift left 32 bits, OR with self, AND 0b0000000000011111000000000000000000000000000000001111111111111111
    n = (n | n << 32) & 0b0000000000011111000000000000000000000000000000001111111111111111
    # shift left 16 bits, OR with self, AND 0b0000000000011111000000000000000011111111000000000000000011111111
    n = (n | n << 16) & 0b0000000000011111000000000000000011111111000000000000000011111111
    # shift left  8 bits, OR with self, AND 0b0001000000001111000000001111000000001111000000001111000000001111
    n =  (n | n << 8) & 0b0001000000001111000000001111000000001111000000001111000000001111
    # shift left  4 bits, OR with self, AND 0b0001000011000011000011000011000011000011000011000011000011000011
    n =  (n | n << 4) & 0b0001000011000011000011000011000011000011000011000011000011000011
    # shift left  2 bits, OR with self, AND 0b0001001001001001001001001001001001001001001001001001001001001001
    n =  (n | n << 2) & 0b0001001001001001001001001001001001001001001001001001001001001001
    return n

### inverse of splitBy3 : method to compact one bit over three to the 21 first bits
# inspired from <https://github.com/aavenel/mortonlib/blob/master/include/morton3d.h>
# difference : torch.Tensor as input and output
def compactBits(n):
    # we only look at 1 bit over 3 : self AND 0b0001001001001001001001001001001001001001001001001001001001001001
    n =               n & 0b0001001001001001001001001001001001001001001001001001001001001001
    # shift right  2 bits, XOR with self, AND 0b0001000011000011000011000011000011000011000011000011000011000011
    n =  (n ^ (n >> 2)) & 0b0001000011000011000011000011000011000011000011000011000011000011
    # shift right  4 bits, XOR with self, AND 0b0001000000001111000000001111000000001111000000001111000000001111
    n =  (n ^ (n >> 4)) & 0b0001000000001111000000001111000000001111000000001111000000001111
    # shift right  8 bits, XOR with self, AND 0b0000000000011111000000000000000011111111000000000000000011111111
    n =  (n ^ (n >> 8)) & 0b0000000000011111000000000000000011111111000000000000000011111111
    # shift right 16 bits, XOR with self, AND 0b0000000000011111000000000000000000000000000000001111111111111111
    n = (n ^ (n >> 16)) & 0b0000000000011111000000000000000000000000000000001111111111111111
    # shift right 32 bits, XOR with self, AND 0b0000000000000000000000000000000000000000000111111111111111111111
    n = (n ^ (n >> 32)) & 0b0000000000000000000000000000000000000000000111111111111111111111
    return n

def sortMortonCodesAndFeatures(morton_codes, features):    
    # print("size of <morton_codes>               :{}".format(morton_codes.size()))
    # print("size of <features>                   :{}".format(features.size()))
    
    with torch.no_grad():
        #
        morton_code_sorted = torch.sort(morton_codes)
    #
    features_sorted = features[morton_code_sorted.indices]
    # print("size of <morton_code_sorted.values>  :{}".format(morton_code_sorted.values.size()))
    # print("size of <morton_code_sorted.indices> :{}".format(morton_code_sorted.indices.size()))
    # print("size of <features_sorted>            :{}".format(features_sorted.size()))
        
    return morton_code_sorted.values, morton_code_sorted.indices, features_sorted

def uniqueMortonCodesAndFeatures(morton_codes, features):    
    # print("size of <morton_codes>               :{}".format(morton_codes.size()))
    # print("size of <features>                   :{}".format(features.size()))
        
    with torch.no_grad():
        #
        unique_morton_code, reverse_morton_code = torch.unique_consecutive(morton_codes, return_inverse=True)
        # print("size of <unique_morton_code>         :{}".format(unique_morton_code.size()))
        # print("size of <reverse_morton_code>        :{}".format(reverse_morton_code.size()))
        
        features_size = features.size(1)
        init_agg = torch.zeros((unique_morton_code.size(0),features_size),device=morton_codes.device,dtype=features.dtype)
        # print("size of <init_agg>                   :{}".format(init_agg.size()))
    
    features_aggregated = init_agg.scatter_add_(0, reverse_morton_code.unsqueeze(1).expand(-1,features_size), features)
    # print("size of <features_aggregated>        :{}".format(features_aggregated.size()))
    
    return unique_morton_code, reverse_morton_code, features_aggregated
    
#%% Classes

class OctreeMorton(nn.Module):
    """
    Octree Class based on Morton code. 
    Not dynamic --> each time self.compute_index(new_cloud) is called, the older clouds are forgotten
    """
    
    def __init__(self, leaf_size=torch.zeros(3)+0.1):
        super(OctreeMorton, self).__init__()
                
        # Contient les codes de Morton de chaque feuille
        # TODO : check leaf_size is a tensor of size 3 --> else convert it
        if isinstance(leaf_size, torch.Tensor):
            self.leaf_size = leaf_size
        else:
            self.leaf_size = torch.zeros(3) + float(leaf_size)
        self.origin = None
        self.max_depth = None
                        
    def __computeIJKFromXYZ__(self, xyz):
        with torch.no_grad():
            return torch.floor((xyz - self.origin) / self.leaf_size)
                
    def __computeMortonCode__(self):
        
        with torch.no_grad():
            #
            self.cloud_ijk = self.__computeIJKFromXYZ__(self.cloud_xyz).to(torch.int64)
            
            #
            ijk_splitBy3 = splitBy3(self.cloud_ijk)
            self.morton_codes[0] = ijk_splitBy3[:,0] | ijk_splitBy3[:,1] << 1 | ijk_splitBy3[:,2] << 2;
            # TODO : check usefullness of <torch.zeros(self.nb_points,device=self.device,dtype=torch.int64)> 
            # self.morton_codes[0] = torch.zeros(self.nb_points,device=self.device,dtype=torch.int64) | ijk_splitBy3[:,0] | ijk_splitBy3[:,1] << 1 | ijk_splitBy3[:,2] << 2;
        
        #
        self.morton_codes[0], self.init_index['init_reverse_sort'], self.aggregated_points[0] = sortMortonCodesAndFeatures(self.morton_codes[0], self.aggregated_points[0])
        
        #
        self.morton_codes[0], self.init_index['init_reverse_unique'], self.aggregated_points[0] = uniqueMortonCodesAndFeatures(self.morton_codes[0], self.aggregated_points[0])
        
    def __compute_full_index__(self):
        """
        compute index at each depth
        """
        if DEBUG:
            print("#---# DEPTH: 00 | Nb Morton Codes : {:09d} #---#".format(self.morton_codes[0].size(0)))
            
        for depth in range(1,MAX_DEPTH):
            if self.morton_codes[depth-1].size(0) > 1:
                # Compute unique morton codes at <depth>
                self.morton_codes[depth], self.reverse_index[depth], self.aggregated_points[depth] = uniqueMortonCodesAndFeatures(self.morton_codes[depth-1] >> 3, self.aggregated_points[depth-1])
                if DEBUG:
                    print("#---# DEPTH: {:02d} | Nb Morton Codes : {:09d} #---#".format(depth, self.morton_codes[depth].size(0)))
                
            else:
                if DEBUG:
                    print("max_depth:",depth)
                assert(self.morton_codes[depth-1].size(0)==1)
                # Get max_depth of tree
                self.max_depth = depth
                    
                # Stop after max_depth
                break
            
    def compute_index(self, new_cloud):
        """
        input: new_cloud
        output: index_dict
        """
        #---------------------------------------------------------------------#
        # What will be returned by <self.compute_index> method
        self.init_index = {'init_reverse_sort':None,
                           'init_reverse_unique':None}
        self.morton_codes = {d:None for d in range(MAX_DEPTH)}
        self.reverse_index = {d:None for d in range(MAX_DEPTH)}
        self.aggregated_points = {d:None for d in range(MAX_DEPTH)}
        
        #---------------------------------------------------------------------#
        # Get device and dtype
        self.device = new_cloud.device
        self.dtype  = new_cloud.dtype
        
        self.leaf_size = self.leaf_size.to(self.device, self.dtype)
        
        with torch.no_grad():
            #---------------------------------------------------------------------#
            # Extract points XYZ
            if DEBUG:
                torch.cuda.synchronize()
                start_time = time.perf_counter()
            self.nb_points = new_cloud.size(0)
            self.cloud_xyz = new_cloud[:,:3]
            # Extract features and
            # Add points to features (to compute nb_points, barycenter and variance matrix)
            self.aggregated_points[0] = torch.cat((torch.ones((self.nb_points,1), device=self.device, dtype=self.dtype),
                                                   self.cloud_xyz,
                                                   torch.pow(self.cloud_xyz, 2.0),
                                                   new_cloud[:,3:]),
                                                  1)
            if DEBUG:
                torch.cuda.synchronize()
                print("extract cloud and features : {:.3f} ms".format((time.perf_counter()-start_time)*1000.0))
            
            #---------------------------------------------------------------------#
            # Compute min of XYZ
            if DEBUG:
                torch.cuda.synchronize()
                start_time = time.perf_counter()
            self.min_xyz = torch.min(self.cloud_xyz,0).values
            if DEBUG:
                torch.cuda.synchronize()
                print("torch.min(self.cloud_xyz,0).values : {:.3f} ms".format((time.perf_counter()-start_time)*1000.0))
            
            #---------------------------------------------------------------------#
            # Compute origin
            if DEBUG:
                torch.cuda.synchronize()
                start_time = time.perf_counter()
            self.origin = self.min_xyz - 0.5 * self.leaf_size
            if DEBUG:
                torch.cuda.synchronize()
                print("Compute origin                      : {:.3f} ms".format((time.perf_counter()-start_time)*1000.0))
        
        #---------------------------------------------------------------------#
        # Compute Morton Index of each point
        if DEBUG:
            torch.cuda.synchronize()
            start_time = time.perf_counter()
        self.__computeMortonCode__()
        if DEBUG:
            torch.cuda.synchronize()
            print("self.__computeMortonCode__()        : {:.3f} ms".format((time.perf_counter()-start_time)*1000.0))
        
        #---------------------------------------------------------------------#
        # Compute Morton Index at each depth
        if DEBUG:
            torch.cuda.synchronize()
            start_time = time.perf_counter()
        self.__compute_full_index__()
        if DEBUG:
            torch.cuda.synchronize()
            print("self.__compute_full_index__()                      : {:.3f} ms".format((time.perf_counter()-start_time)*1000.0))
        
        # TODO : check it is really usefull to return <self.morton_codes>
        return self.init_index, self.morton_codes, self.reverse_index, self.aggregated_points
            
    def get_leaf_cloud(self):
        """
        should return one point for each morton code at depth 0 (i.e. for each leaf)
        """
                
        IJK = torch.cat((compactBits(self.morton_codes[0]).unsqueeze(1),
                         compactBits(self.morton_codes[0]>>1).unsqueeze(1),
                         compactBits(self.morton_codes[0]>>2).unsqueeze(1)),
                        1)
        XYZ_center = (IJK.to(torch.float32) + 0.5)* self.leaf_size + self.origin
        nb_points = self.aggregated_points[0][:,0].unsqueeze(1).to(torch.float64)
        
        # Compute Barycenters
        XYZ_barycenter = self.aggregated_points[0][:,1:4].to(torch.float64) / nb_points
        
        # Compute Variance
        var = (self.aggregated_points[0][:,4:7].to(torch.float64) / nb_points) 
        mean_square = torch.pow(XYZ_barycenter.to(torch.float64),2)
        var = var - mean_square
        var = F.relu(var)
        
        # Compute Standard Deviation
        std = torch.sqrt(var)
        
        reflectance = self.aggregated_points[0][:,7].to(torch.float64).unsqueeze(1) / nb_points
        XYZ_offset = XYZ_barycenter - XYZ_center
        
        # nb_points = nb_points.to(torch.int64)
        
        return XYZ_center, XYZ_barycenter.to(torch.float32), nb_points.to(torch.float32), var.to(torch.float32), std.to(torch.float32), reflectance.to(torch.float32), XYZ_offset.to(torch.float32)
    
    def get_full_octree_cloud(self):
        """
        should return one point for each morton code at each depth
        """     
        full_cloud = torch.cat((self.aggregated_points[0], torch.zeros((self.aggregated_points[0].size(0),1), device=self.device, dtype=self.dtype)),1)
        full_IJK = torch.cat((compactBits(self.morton_codes[0]   ).unsqueeze(1),
                              compactBits(self.morton_codes[0]>>1).unsqueeze(1),
                              compactBits(self.morton_codes[0]>>2).unsqueeze(1)),
                             1)
        
        # Loop over depth
        for d in range(1,self.max_depth):
            temp = torch.cat((self.aggregated_points[d], torch.zeros((self.aggregated_points[d].size(0),1), device=self.device, dtype=self.dtype)+d),1)
            full_cloud = torch.cat((full_cloud,temp),0)

            # TODO : check bit positions
            m_codes = self.morton_codes[d] << (3*d)
            temp = torch.cat(((compactBits(m_codes   )).unsqueeze(1),
                              (compactBits(m_codes>>1)).unsqueeze(1),
                              (compactBits(m_codes>>2)).unsqueeze(1)),
                             1)
            full_IJK = torch.cat((full_IJK,temp),0)
                
        depth = full_cloud[:,-1].to(torch.int32)
        voxel_size = (torch.ones(depth.size(0), device=depth.device, dtype=torch.int32)<<depth).unsqueeze(1)
        
        # TODO : check this is right XYZ_center (miss leaf_size*depth ?)
        XYZ_center = (full_IJK.to(torch.float32) + 0.5 * (self.leaf_size.unsqueeze(0) * voxel_size)) + self.origin
        
        nb_points = full_cloud[:,0].unsqueeze(1).to(torch.float64)
        
        # Compute Barycenters
        XYZ_barycenter = full_cloud[:,1:4].to(torch.float64) / nb_points
        
        # Compute Variance
        var = (full_cloud[:,4:7].to(torch.float64) / nb_points) 
        mean_square = torch.pow(XYZ_barycenter.to(torch.float64),2)
        var = var - mean_square
        var = F.relu(var)
        
        # Compute Standard Deviation
        std = torch.sqrt(var) / (self.leaf_size.unsqueeze(0) * voxel_size)
        
        reflectance = full_cloud[:,7].to(torch.float64).unsqueeze(1) / nb_points
        XYZ_offset = XYZ_barycenter - XYZ_center
        
        # nb_points = nb_points.to(torch.int64)
                
        return XYZ_center, XYZ_barycenter.to(torch.float32), nb_points.to(torch.float32), var.to(torch.float32), std.to(torch.float32), reflectance.to(torch.float32), XYZ_offset.to(torch.float32), depth.to(torch.float32)
        