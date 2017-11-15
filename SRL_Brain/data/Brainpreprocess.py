#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 15:52:30 2017

@author: wuzhenglin
"""

import glob
import os
import pandas as pd
import numpy as np 
import pandas as pd 
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom
import scipy.misc
import numpy as np
from skimage.io import imread
import scipy

def resize(arr, shape):
    
    inti = arr[0]
    inti = scipy.misc.imresize(inti, shape)
    inti = inti[np.newaxis, :] 
    
    
    for i in range(arr.shape[0]):
        
        if i != 0:
            changesize = scipy.misc.imresize(arr[i], shape)
            changesize = changesize[np.newaxis, :]           
            inti = np.append(inti, changesize, axis = 0)
        
    return inti
 

def folder_travel(s, folder_path):
    
    files = os.listdir(folder_path)
    
    for each in files:
        
        if (each[0] == '.'):
            pass
        
        else:
        
            flag = os.path.isdir(os.path.join(folder_path, each))
        
            if flag:
                path = folder_path + '/' + each
                s = folder_travel(s, path)
                
            else:
                f = folder_path + '/' + each
                iter_f = iter(f)
                str = ''
                for line in iter_f:
                    str = str + line
                s.append(str)
          
         
    
    return s


def load_scan(path):
    
    
    lit = []
    lit = folder_travel(lit, path)
    print len(lit) 
    
    slices = [dicom.read_file(fil) for fil in lit]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu(slices):
    
    
    image = np.stack([scipy.misc.imresize(s.pixel_array, [128, 128]) for s in slices])

    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1,1,1]):
    

    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

def plot_3d(image, threshold=-300):

    p = image.transpose(2,1,0)
    verts, faces = measure.marching_cubes(p, threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()


def plot_ct_scan(scan):


    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(50, 50))
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone)
        





def normalize(image):
    
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0

    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def zero_center(image):
    
    PIXEL_MEAN = 0.25

    image = image - PIXEL_MEAN
    return image




if __name__ == '__main__':
    
    
    print 'Jay Chou 666'
    
    

    
    
    

    
    