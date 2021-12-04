#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xavier Roynard
"""

import numpy as np
from plyfile import PlyData, PlyElement

### return a numpy array
def read_ply(file_name):
    plydata = PlyData.read(file_name)
    return np.array(plydata.elements[0].data)

### cloud should be like np.array( ... , dtype=[('x','f4'),('y','f4'),('z','f4'), ... ])
def write_ply(cloud, file_name):    
    el = PlyElement.describe(cloud, 'vertex') 
    PlyData([el]).write(file_name)   