#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xavier Roynard
"""

class Logger(object):
    
    def __init__(self,
                 log_file_base,
                 validation,
                 nb_classes,
                 ):
        
        self.log_file_base = log_file_base
        self.validation = validation
        self.nb_classes = nb_classes
        
        # Initialize log files
        log_file = open( self.log_file_base+"_train.txt" , 'w')
        log_file.write("epoch,cpt_backward_pass,loss,accuracy,lr,time,precisions,recalls,confusion_matrix,{}\n".format(self.nb_classes))
        log_file.close()  
        if self.validation:
            log_file = open( self.log_file_base+"_val.txt" , 'w')
            log_file.write("epoch,cpt_backward_pass,loss,accuracy,lr,time,precisions,recalls,confusion_matrix,{}\n".format(self.nb_classes))
            log_file.close()
        
    def write(self, line, phase="train"):
        
        assert(phase == "train" or (self.validation and phase == "val"))
        
        log_file = open( self.log_file_base+"_"+phase+".txt" , 'a')
        log_file.write(line)
        log_file.close()