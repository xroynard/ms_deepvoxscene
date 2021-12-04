#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xavier Roynard
"""

from __future__ import print_function, division

import time
import os

# to read and write config files
import yaml 
    
###############################################################################
dataset_to_nb_classes = {"parislille3d":9,
                         "semantic3d":8,
                         "s3dis":13,
                         }

class_name_list = {}
class_name_list["parislille3d"] = ["ground","building","pole","bollard","trash can","barrier","pedestrian","car","natural"]
class_name_list["semantic3d"] = ["man-made terrain","natural terrain","high vegetation","low vegetation","buildings","hard scape","scanning artefacts","cars"]
class_name_list["s3dis"] = ["clutter","ceiling","floor","wall","column","beam","window","door","table","chair","bookcase","sofa","board"]
class_name_list["vaihingen"] = ["powerline","low vegetation","impervious surfaces","car","fence/hedge","roof","facade","shrub","tree"]

###############################################################################
class Parameters(object):
    
    def __init__(self, config_file):
        
        # Read self.config file
        instream = open(config_file, 'r')
        self.config = yaml.load(instream)
        instream.close()
    	        
        #
        self.config["DATA_DIR"] = self.config["DATA_DIR"] if ("DATA_DIR" in self.config) else os.path.join(os.getcwd(), os.path.pardir, "data")
        self.config["DATASET_DIR"] = self.config["DATASET_DIR"] if ("DATASET_DIR" in self.config) else os.path.join(self.config["DATA_DIR"], self.config["DATASET"])        
            
        self.config["RUN_DIR"] = self.config["RUN_DIR"] if ("RUN_DIR" in self.config) else os.path.join(os.getcwd(), os.path.pardir, "runs")
        if not(os.path.exists(self.config["RUN_DIR"])):
            os.mkdir(self.config["RUN_DIR"])
        
        #
        self.config["USE_COLOR"] = (self.config["USE_COLOR"] if "USE_COLOR" in self.config else None) or not(self.config["DATASET"] == "parislille3d")
        self.config["USE_REFLECTANCE"] = (self.config["USE_REFLECTANCE"] if "USE_REFLECTANCE" in self.config else None) or not(self.config["DATASET"] == "s3dis")
        
        #
        self.config["NB_CLASSES"] = dataset_to_nb_classes[self.config["DATASET"]]        
        self.config["CLASS_NAME_LIST"] = class_name_list[self.config["DATASET"]]
        self.config["NB_CHANNELS"] = 1  + 3*self.config["USE_COLOR"] + self.config["USE_REFLECTANCE"] # max 5 channels !
        self.config["NB_SCALES"] = len(self.config["SCALES"]) if "SCALES" in self.config else 0
          
        # Set LOG_DIR
        if not("LOG_DIR" in self.config):
            identifier = int(time.time() * 100000000)
            self.config["LOG_DIR"] = os.path.join(self.config["RUN_DIR"], "train_" + self.config["MODEL_NAME"] + "_" + str(identifier))
                
            if not(os.path.exists(self.config["LOG_DIR"])):
                os.mkdir(self.config["LOG_DIR"])
                
            self.config["MODEL_DIR"] = os.path.join(self.config["LOG_DIR"], "models")
            if not(os.path.exists(self.config["MODEL_DIR"])):
                os.mkdir(self.config["MODEL_DIR"])
        
    def setp(self, key, value):
        
        self.config[key] = value
        
    def getp(self, key):
        
        return self.config[key]
    
    def update_parameters(self, config_file):
        
        # Read self.config file
        instream = open(config_file, 'r')
        new_config = yaml.load(instream)
        instream.close()
        
        for key in new_config:
            self.config[key] = new_config[key]
            
        # Set new LOG_DIR
        identifier = int(time.time() * 100000000)
        self.config["LOG_DIR"] = os.path.join(self.config["RUN_DIR"], "test_" + self.config["MODEL_NAME"] + "_" + str(identifier))
            
        if not(os.path.exists(self.config["LOG_DIR"])):
            os.mkdir(self.config["LOG_DIR"])
            
        self.config["CLOUD_DIR"] = os.path.join(self.config["LOG_DIR"], "clouds")
        if not(os.path.exists(self.config["CLOUD_DIR"])):
            os.mkdir(self.config["CLOUD_DIR"])
    
    def write_parameters(self, config_file=None):   
        
        config_file = config_file or self.config["LOG_DIR"]
        
        # Write self.config file
        outstream = open(os.path.join(self.config["LOG_DIR"], "config.yaml"), 'w')
        outstream.write("#Date:{}\n".format(time.asctime()))
        outstream.write("\n")
        yaml.dump(self.config, outstream)
        outstream.close()
        
    def print_parameters(self):
        
        print("PARAMETERS:")
        print("\tDATASET: {}".format(self.config["DATASET"]))
        print("")
        print("\tDATA_DIR: {}".format(self.config["DATA_DIR"]))
        print("\tDATASET_DIR: {}".format(self.config["DATASET_DIR"]))
        print("\tRUN_DIR: {}".format(self.config["RUN_DIR"]))
        print("\tLOG_DIR: {}".format(self.config["LOG_DIR"]))
        print("\tMODEL_DIR: {}".format(self.config["MODEL_DIR"]))
        print("\tCLOUD_DIR: {}".format(self.config["CLOUD_DIR"]))
        print("")
        print("\tNB_SAMPLES: {}".format(self.config["NB_SAMPLES"]))
        print("\tOFFSET_SAMPLES: {}".format(self.config["OFFSET_SAMPLES"]))
        print("")
        print("\tNUM_WORKERS: {}".format(self.config["NUM_WORKERS"]))
        print("\tDEVICE_ID: {}".format(self.config["DEVICE_ID"]))
        print("")
        print("\tUSE_COLOR: {}".format(self.config["USE_COLOR"]))
        print("\tUSE_REFLECTANCE: {}".format(self.config["USE_REFLECTANCE"]))
        print("")
        print("\tNB_CHANNELS: {}".format(self.config["NB_CHANNELS"]))
        print("\tNB_CLASSES: {}".format(self.config["NB_CLASSES"]))
        print("\tSCALES: {}".format(self.config["SCALES"]))
        print("\tNB_SCALES: {}".format(self.config["NB_SCALES"]))
        print("\tGRID_SIZE: {}".format(self.config["GRID_SIZE"]))
        print("")
        print("\tVOXEL_SIZE: {}".format(self.config["VOXEL_SIZE"]))
        print("")
        print("\tBATCH_SIZE: {}".format(self.config["BATCH_SIZE"]))
        print("\tNB_POINTS_PER_CLASS: {}".format(self.config["NB_POINTS_PER_CLASS"]))
        print("\tNUM_EPOCHS: {}".format(self.config["NUM_EPOCHS"]))
        print("")
        print("\tPHASE_LIST: {}".format(self.config["PHASE_LIST"]))
        print("")
        print("\tINITIALIZATION: {}".format(self.config["INITIALIZATION"]))
        print("")
        print("\tEPS: {}".format(self.config["EPS"]))
        print("")
        print("\tWD: {}".format(self.config["WD"]))
        print("")
        print("\tLR: {}".format(self.config["LR"]))
        print("\tLR_SCHEDULER: {}".format(self.config["LR_SCHEDULER"]))
        print("")
        if "MODEL_NAME" in self.config:
            print("\tMODEL_NAME: {}".format(self.config["MODEL_NAME"]))
            print("")