#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xavier Roynard
"""

from __future__ import print_function, division

import time
import os
import sys
import glob

import numpy as np

import torch
from torch.utils.data import Dataset

from sklearn.neighbors import KDTree

# pour importer les modules 
sys.path.insert(0, os.path.abspath('..'))
from utils.ply_utils import read_ply, write_ply

###############################################################################
def pc_augmentation(clouds):
    
    # random flip / miror
    flip = np.random.randint(2)
    # random rotation around Z-axis
    theta = 2 * np.pi * np.random.rand()
    R = np.array([[np.cos(theta) , -np.sin(theta)] , [np.sin(theta) , np.cos(theta)]])
    # noise (on each point)
    NOISE = 0.01 # in meters
    # occlusion (randomly removes points), must be < 1.0 and >= 0.0
    OCCLUSION_RATE = 0.05
    # artefacts (randomly adds points), must be >= 0.0
    ARTEFACT_RATE = 0.05
    # scale the whole neigborhood, must be < 1.0 and >= 0.0
    SCALE_RANGE = 0.05 #  0.1 --> +/-10%  |  0.05 --> +/-5%
    scale = (2 * np.random.rand() - 1) * SCALE_RANGE + 1
    
    for key in clouds:
        nb_pts = clouds[key].shape[0]
        if nb_pts>1:
            nb_channels = clouds[key].shape[1]-4
            min_cloud  = np.min(clouds[key][:,:3],axis=0)
            max_cloud  = np.max(clouds[key][:,:3],axis=0)
            
            # random flip / miror
            if flip:
                temp = np.copy(clouds[key][:,0])
                clouds[key][:,0] = np.copy(clouds[key][:,1])
                clouds[key][:,1] = temp
                        
            # random rotation around Z-axis
            clouds[key][:,:2] = np.dot(clouds[key][:,:2], R)
                    
            # noise (on each point)            
            clouds[key][1:,:3] += NOISE * np.random.randn(nb_pts-1,3)
                        
            # occlusion (randomly removes points),
            perm_inds = np.random.permutation(np.arange(1,nb_pts))[:-np.random.randint(0, high=int(OCCLUSION_RATE*nb_pts)+1)-1]
            clouds[key] = np.vstack( (clouds[key][0],clouds[key][perm_inds]) )
            nb_pts = clouds[key].shape[0]
                    
            # artefacts (randomly adds points)
            nb_random_points = np.random.randint(0, high=int(ARTEFACT_RATE*nb_pts)+1)
            random_points = (max_cloud - min_cloud)*np.random.rand(nb_random_points,3) + min_cloud
            if nb_channels==0:
                random_classes = np.random.randint(0, high=np.max(clouds[key][:,3])+1, size=(nb_random_points,1))
                clouds[key] = np.vstack( ( clouds[key] , np.hstack( (random_points,random_classes) ) ) )
            nb_pts = clouds[key].shape[0]            
                
            # scale the whole neigborhood
            clouds[key][:,:3] = scale * clouds[key][:,:3]
                         
    return clouds

class PointCloudDataset(Dataset):
    """Point Cloud Dataset."""

    def __init__(self, 
                 point_cloud_file,
                 grid_size=None,
                 voxel_size=None,
                 scales=None,
                 nb_pts_per_class=0,
                 nb_classes=0,
                 testing=False,
                 use_no_labels=False,
                 use_class0=False,
                 use_color=False,
                 use_reflectance=False,
                 **kwargs
                 ):
        """
        Args:
            point_cloud_file (string): Directory with all the poinclouds in ply-files.
            
            grid_size (int): Optional number of voxels in each dimension (must be even)
            voxel_size (int): Optional size of a voxel (meters)
            
            scales (list of int or float): list of scales for 3D grids in input of network
            nb_pts_per_class (int): number of points sampled in each class before each epoch
            nb_classes (int): number of classes in the dataset
            
            testing (bool): Optional set True so that __getitem__ iterates over the whole point cloud.
            
            use_no_labels (bool): set to True if the point cloud doesn't provide a label/class for each point
            use_class0 (bool): set to True if the class 0 is not used for "unclassified" points (for instance for S3DIS dataset)
            use_color (bool): set to True if dataset provides color for each point and you want color channels in input to your network
            use_reflectance (bool): set to True if dataset provides reflectance for each point and you want a reflectance channel in input to your network
        """
        
        self.point_cloud_file = point_cloud_file
        self.testing = testing
        self.use_no_labels = use_no_labels
        self.use_class0 = use_class0
        self.use_color = use_color
        self.use_reflectance = use_reflectance
            
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.scales = scales
        self.nb_pts_per_class = nb_pts_per_class
        self.nb_classes = nb_classes
        assert(nb_classes>0)
        
        # Checks self.scales is a list or an int or a float
        # And converts it to a list
        def canbeconvertedtolist(seq):
            try:
                list(seq)
            except:
                return False
            return True
        if not(self.scales):
            self.scales = [1]
            print("scales not set --> scales = {} used".format(self.scales))  
        elif (isinstance(self.scales, int) or isinstance(self.scales, float)):
            self.scales = [self.scales]
        elif not(canbeconvertedtolist(self.scales)):
            self.scales = [1]
            print("scales is '{}' not an int or a float or a sequence --> scales = {} used".format(type(scales), self.scales)) 
                                    
        # Checks each element of self.scales is an int or a float >0
        self.scales = list(self.scales)
        for i,s in enumerate(self.scales):
            if not(isinstance(s, int) or isinstance(s, float)):
                self.scales[i] = 1
                print("a value in scales is '{}' not int or float --> value changed to {}".format(type(s),self.scales[i]))
            elif s<=0:
                self.scales[i] = 1
                print("a value in scales is {} not >0 --> value changed to {}".format(s,self.scales[i]))
        self.scales = np.unique(self.scales)
            
        self.scales = sorted(list(self.scales))
        self.nb_scales = len(self.scales)
        self.min_scale = self.scales[0]
        self.max_scale = self.scales[-1]
            
        # Checks voxel_size is an int or a float >0
        if not(self.voxel_size):
            self.voxel_size = 0.1
            print("voxel_size not set --> voxel_size = {} used".format(self.voxel_size))            
        elif not(isinstance(self.voxel_size, float) or isinstance(self.voxel_size, int)):
            self.voxel_size = 0.1
            print("voxel_size is '{}' not a float or an int --> voxel_size = {} used".format(type(voxel_size), self.voxel_size))
        elif self.voxel_size <= 0.0:
            self.voxel_size = 0.1
            print("voxel_size is {} not >0 --> voxel_size = {} used".format(voxel_size, self.voxel_size))
        else:
            self.voxel_size = float(self.voxel_size)
          
        # Checks grid_size is an int >0 and even    
        if not(self.grid_size):
            self.grid_size = 32
            print("grid_size not set --> grid_size = {} used".format(self.grid_size))            
        elif not(isinstance(self.grid_size, int)):
            self.grid_size = 32
            print("grid_size is '{}' not an int --> grid_size = {} used".format(type(grid_size), self.grid_size))
        elif self.grid_size <= 1 or self.grid_size%2!=0:
            self.grid_size = 32
            print("grid_size is {} not >0 or not even --> grid_size = {} used".format(grid_size, self.grid_size))
           
        # Set radius search
        self.r = np.sqrt(3) * self.voxel_size * self.grid_size * self.max_scale / 2
                                        
        # Find all subsampled point clouds
        if isinstance(self.point_cloud_file,str):
            if os.path.isfile(self.point_cloud_file) and self.point_cloud_file.split('.')[-1]=="ply":
                # Looks for any directory looking like sub_*cm/ in directory containing point_cloud_file
                self.subsampled_pathnames = np.array(glob.glob(os.path.join(os.path.dirname(self.point_cloud_file),"sub_*cm")))
                
                self.subsampled_pathnames = sorted(self.subsampled_pathnames[np.where([os.path.isdir(f) for f in self.subsampled_pathnames])[0]])
                if len(self.subsampled_pathnames)>0: # if there are some repo 'sub_*cm'
                    self.scale_to_subsampfile = {}
                    for sc in self.scales:
                        best_subsamp = 0
                        for sp in self.subsampled_pathnames:
                            subsamp = float(os.path.basename(sp)[4:-2])
                            if 2*subsamp <= 100*sc*self.voxel_size: # *100 to convert to cm !
                                if subsamp > best_subsamp:
                                    self.scale_to_subsampfile[sc] = sp
                                    best_subsamp = subsamp
                            if best_subsamp==0:
                                self.scale_to_subsampfile[sc] = self.point_cloud_file
                    self.subsampfile_to_scale = {}
                    
                    for sc in self.scale_to_subsampfile:
                        if os.path.isfile( os.path.join(self.scale_to_subsampfile[sc],os.path.basename(self.point_cloud_file)) ):
                            self.scale_to_subsampfile[sc] = os.path.join(self.scale_to_subsampfile[sc],os.path.basename(self.point_cloud_file))
                            self.subsampfile_to_scale[self.scale_to_subsampfile[sc]] = sc
                        else:
                            self.scale_to_subsampfile[sc] = self.point_cloud_file
                            self.subsampfile_to_scale[self.scale_to_subsampfile[sc]] = sc
                            print("!Warning: No file {} subsampled at scale {} cm!!!".format(os.path.basename(self.point_cloud_file),0.5*100*sc*self.voxel_size ))

                    for sc in self.scale_to_subsampfile:
                        self.subsampfile_to_scale[self.scale_to_subsampfile[sc]] = max(sc,self.subsampfile_to_scale[self.scale_to_subsampfile[sc]])

                    self.lowest_subsamp_file = self.scale_to_subsampfile[self.min_scale]
        
                else: # if there is no repo 'sub_*cm'
                    self.scale_to_subsampfile = {sc:self.point_cloud_file for sc in self.scales}
                    self.subsampfile_to_scale = {self.point_cloud_file:self.max_scale}
                    self.lowest_subsamp_file = self.point_cloud_file
            elif os.path.isdir(self.point_cloud_file):
                print("!Warning: {} is a directory, file expected!".format(self.point_cloud_file))
            else:
                print("!Warning: this is not a point cloud(.ply) : {}!".format(self.point_cloud_file))
        else:
            print("!Warning: {} in not a string, file name expected!".format(self.point_cloud_file))
                        
        # Checks that all requierd fields are in the point clouds
        for f in self.subsampfile_to_scale:
            pc = read_ply(f)
            if self.use_color and not(('red' in pc.dtype.fields) and ('green' in pc.dtype.fields) and ('blue' in pc.dtype.fields)):
                print("!Warning: cloud {} doesn't contain the color fields (<red> <green> <blue>) ! ==> self.use_color = False".format(f))
                self.use_color = False
            if self.use_reflectance and not('reflectance' in pc.dtype.fields):
                print("!Warning: cloud {} doesn't contain the <reflectance> field ! ==> self.use_reflectance = False".format(f))
                self.use_reflectance = False

        # Reads point clouds and puts them in np.array
        self.pc = {}
        for ssfile in self.subsampfile_to_scale:
            self.pc[ssfile] = np.array([], order='C')
        
        for ssfile in self.subsampfile_to_scale:
            f_pc = read_ply(ssfile)
            # Reads 3D coordinates
            tmp_pc = np.vstack((f_pc['x'],f_pc['y'],f_pc['z']))
            
            # Reads class channel
            if not(self.use_no_labels):
                tmp_pc = np.vstack( (tmp_pc, f_pc['class']) )
                              
            # Reads color channel
            if self.use_color:
                tmp_pc = np.vstack( (tmp_pc, f_pc['red'], f_pc['green'], f_pc['blue']) )
                
            # Reads reflectance channel
            if self.use_reflectance:
                tmp_pc = np.vstack( (tmp_pc, f_pc['reflectance']) )

            self.pc[ssfile] = np.vstack((self.pc[ssfile].reshape((len(self.pc[ssfile]),tmp_pc.shape[0])),tmp_pc.transpose()))
            del f_pc
            del tmp_pc

        # Compute number of channels in input of network
        self.nb_channels = 1 + 3 * self.use_color + self.use_reflectance
            
        # 
        if not(self.use_no_labels):
            if not(self.use_class0):
                for ssfile in self.subsampfile_to_scale:
                    # removes points with class 0 : unclassified or unknown
                    self.pc[ssfile] = self.pc[ssfile][np.where(self.pc[ssfile][:,3] != 0)]
                    # Rescale classes from 0 to 8 (instead of 1 to 9)
                    self.pc[ssfile][:,3] -= 1
    
            # Classes actuelly available in the dataset
            self.classes, self.pts_par_classe = np.unique(self.pc[self.lowest_subsamp_file][:,3], return_counts=True)
            self.classes = self.classes.astype(int)
            if (self.nb_classes != np.max(self.classes)+1):
                print("!Warning: this dataset doesn't contain all expected classes:")
                print("\tnumber of expected classes: {}".format(self.nb_classes))
                print("\tmax available class       : {}".format(np.max(self.classes)+1))
                print("\tavailable classes         : {}".format(self.classes))
            self.classe_index = {}
            for cpt,cl in enumerate(self.classes):
                self.classe_index[cl] = np.nonzero(self.pc[self.lowest_subsamp_file][:,3] == cl)[0]
                assert( self.pts_par_classe[cpt] == len(self.classe_index[cl]) )
            self.pts_par_classe = {}
            for cl in self.classes:
                self.pts_par_classe[cl] = len(self.classe_index[cl])                
                
        self.search_index = {}
        for ssfile in self.subsampfile_to_scale:            
            self.search_index[ssfile] = KDTree(np.copy(self.pc[ssfile][:,:3]).astype(np.float32))  
                
        ### Randomly find <N> points in each class
        ### they are put in : self.training_points
        self.indices = np.arange(len(self.pc[self.lowest_subsamp_file]))
        self.randomize_samples()
        
        #
        if self.testing:
            ### Find voxel centers for testing only sur on a 3D grid
            self.test_grid_mode(True)
                        
    def __len__(self):
        if self.testing:
            return len(self.testing_points)
        else:
            return self.class_index_offset[-1]
    
    ## Compute index offset for each class
    def _compute_index_offset_(self):        
        self.class_index_offset = np.zeros(np.max(self.classes)+1).astype(int)       
        for cl in self.classes:
            self.class_index_offset[cl] = len(self.training_points[cl])
        self.class_index_offset = np.cumsum(self.class_index_offset.astype(int))

    # Re-sample <self.nb_pts_per_class> training points in each class
    # can be called before each epoch
    def randomize_samples(self, nb_samples_per_class=None):
        if not(self.use_no_labels):
            if nb_samples_per_class == None:
                nb_samples_per_class = {cl:self.nb_pts_per_class for cl in self.classes}
            else:
                assert( isinstance(nb_samples_per_class, dict) )
                for cl in self.classes:
                    assert( isinstance(nb_samples_per_class[cl], int) and nb_samples_per_class[cl]>=0)
                
            self.training_points = {}
            for cl in self.classes:
                nb_pts_cl = len(self.classe_index[cl])
                assert( nb_pts_cl == self.pts_par_classe[cl] )
                if(0<nb_pts_cl):
                    self.training_points[cl] = np.random.choice(self.classe_index[cl],size=min(nb_samples_per_class[cl],nb_pts_cl),replace=False)
                    if(nb_pts_cl < nb_samples_per_class[cl]):
                        print("!Warning: class {} has only {} points in this dataset < {}".format(cl,nb_pts_cl,nb_samples_per_class[cl]))
                else:
                    print("!Warning: class {} has no samples in this dataset".format(cl))
                    self.training_points[cl] = []
            
            #
            self._compute_index_offset_()

    def test_grid_mode(self, b=None):
        if b!=None:
            self.testing = b
        else:
            self.testing = not(self.testing)
        
        if self.testing:
            print()
            print("###############################################################################")
            print("Find index of closest point to voxel center for each voxel")
            start_time = time.time()
            offset = np.min(self.pc[self.lowest_subsamp_file][:,:3], axis=0)# - (self.max_scale*self.voxel_size) * self.grid_size//2
            self.voxel_discretization_step = self.min_scale*self.voxel_size
            self.voxels, unique_counts = np.unique(np.floor((self.pc[self.lowest_subsamp_file][:,:3] - offset)/(self.voxel_discretization_step)).astype(int), return_counts=True, axis=0)
            self.voxels_centers = (self.voxel_discretization_step) * (self.voxels + 0.5) + offset
            print("Number of voxels : {}".format(len(self.voxels)))
            self.testing_points = np.zeros(len(self.voxels), dtype=int)
            
            # find index of closest point to voxel center for each voxel
            for i,v in enumerate(self.voxels_centers):
                self.testing_points[i] = self.search_index[self.lowest_subsamp_file].query(v.astype(np.float32).reshape(1, -1), 1)[1]
                print("\rFind testing points: {:6.2f}% -> {:.2f}s".format(100*i/len(self.voxels_centers),time.time() - start_time), end="")
            print()

    # Saves cloud with predicted class for each point
    def write_pred_cloud(self, pred_class, file_name):
        if self.testing:
            cloud = [tuple(row) for row in np.hstack( (self.pc[self.lowest_subsamp_file][self.testing_points],pred_class.reshape((len(pred_class),1))) )]
        else:
            cloud = np.array([],ndmin=2).reshape((0,self.pc[self.lowest_subsamp_file].shape[1]))
            for pts in self.training_points.values():
                cloud = np.vstack( (cloud,self.pc[self.lowest_subsamp_file][pts]) )                
            cloud = [tuple(row) for row in np.hstack( (cloud,pred_class.reshape((len(pred_class),1))) )]
            
        DTYPE = [('x','f4'),('y','f4'),('z','f4')]
        if not(self.use_no_labels):
            DTYPE.append( ('class','int32') )
        if self.use_color:
            DTYPE.append( ('red','i1') )
            DTYPE.append( ('green','i1') )
            DTYPE.append( ('blue','i1') )
        if self.use_reflectance:
            DTYPE.append( ('reflectance','i4') )      
        DTYPE.append( ('predicted_class','int32') )
                
        cloud = np.array(cloud, dtype=DTYPE)
        write_ply(cloud, file_name)
        
    # Saves cloud with predicted class for each point and "probability" (output of softmax layer) of of belonging to in each class
    def write_pred_proba_cloud(self, pred_proba_class, file_name):
        pred_class = np.argmax(pred_proba_class, axis=1)
        cloud = []
        if self.testing:
            cloud = [tuple(row) for row in np.hstack( (self.pc[self.lowest_subsamp_file][self.testing_points],pred_class.reshape((len(pred_class),1)),pred_proba_class) )]
        else:
            for pts in self.training_points.values():
                for i in pts:
                    cloud.append(tuple(self.pc[self.lowest_subsamp_file][i]))
            cloud = [tuple(row) for row in np.hstack( (cloud,pred_class.reshape((len(pred_class),1)),pred_proba_class) ) ]
                
        DTYPE = [('x','f4'),('y','f4'),('z','f4')]
        if not(self.use_no_labels):
            DTYPE.append( ('class','int32') )
        if self.use_color:
            DTYPE.append( ('red','i1') )
            DTYPE.append( ('green','i1') )
            DTYPE.append( ('blue','i1') )                
        if self.use_reflectance:
            DTYPE.append( ('reflectance','i4') )      
        DTYPE.append( ('predicted_class','int32') )
        for p in range(pred_proba_class.shape[1]):
            DTYPE.append( ('predicted_proba'+str(p),'f4') )
                
        cloud = np.array(cloud, dtype=DTYPE)
        write_ply(cloud, file_name)

    def __getitem__(self, idx):
        
        # Get class and index of point in the point cloud
        cl = 0
        if self.testing:
            index = self.testing_points[idx]
            if not(self.use_no_labels):
                cl = self.pc[self.lowest_subsamp_file][index,3]
        else:
            while(self.class_index_offset[cl] <= idx):
                cl += 1
            index_offset = self.class_index_offset[cl]                
            index = self.training_points[cl][idx - index_offset]
                
        # Get query point for neighborhood search
        query_pt = self.pc[self.lowest_subsamp_file][index,:3].astype(np.float32)
        
        # Neighborhood search
        ms_neighborhood = {}
        for ssfile in self.subsampfile_to_scale:
            ms_neighborhood[ssfile] = np.copy( self.pc[ssfile][ self.search_index[ssfile].query_radius(query_pt.reshape(1, -1), np.sqrt(3) * self.voxel_size * self.grid_size * self.subsampfile_to_scale[ssfile] / 2, return_distance=True, sort_results=True)[0][0] ] ) 
                    
        # Centering around query point
        for ssfile in self.subsampfile_to_scale:
            ms_neighborhood[ssfile][:,:3] = ms_neighborhood[ssfile][:,:3] - query_pt
                        
        if not(self.testing):
            # Data augmentation
            ms_neighborhood = pc_augmentation(ms_neighborhood)
        
        vox_input = []
        # Loop over scales
        for i,s in enumerate(self.scales):
            # Compute indices of occupied voxels
            sneighborhood = np.copy(ms_neighborhood[self.scale_to_subsampfile[s]])
            sneighborhood[:,:3] = np.floor(sneighborhood[:,:3] / (s*self.voxel_size)) + (self.grid_size // 2)        
            sneighborhood[:,:3] = sneighborhood[:,:3].astype(int)

            # Find indices of points actually falling in the occupancy grid
            good_indexes = (sneighborhood[:,:3] < self.grid_size) & (sneighborhood[:,:3] >= 0)
            good_indexes = good_indexes[:,0] & good_indexes[:,1] & good_indexes[:,2]            
            sneighborhood = sneighborhood[good_indexes]            
                
            # Initialize the 3D voxel grid
            vox = np.zeros((self.nb_channels, self.grid_size, self.grid_size, self.grid_size))
            
            # Find unique occupied voxels
            unique_sneighborhood, unique_counts = np.unique(sneighborhood[:,:3].astype(int), return_counts=True, axis=0)
            
            # Fill the grid
            vox[0,unique_sneighborhood[:,0], unique_sneighborhood[:,1], unique_sneighborhood[:,2]] = 1.0
            if self.nb_channels > 1:
                for pt in sneighborhood:
                    vox[1:, int(pt[0]), int(pt[1]), int(pt[2])] += pt[(4 - self.use_no_labels):]
                for _i,u_pt in enumerate(unique_sneighborhood):
                    vox[1:,u_pt[0], u_pt[1], u_pt[2]] /= unique_counts[_i]

            vox_input.append(torch.from_numpy(vox).float())
                
        sample = {'input':vox_input}
        if not(self.use_no_labels):
            sample['label'] = int(cl)
                              
        return sample
