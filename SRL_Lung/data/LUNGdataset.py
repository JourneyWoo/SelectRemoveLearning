#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 18:37:41 2017

@author: wuzhenglin
"""

import os 
import tensorflow as tf 
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np
import dicom as dm
from LUNGpreprocess import load_scan, get_pixels_hu, resample, normalize, zero_center, folder_travel
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



def make_dataset():
    
    path_cancer = '/Users/wuzhenglin/Python_nice/SAL_LUNG/lung_cancer_CT/DOI'
    sl = []
    l = folder_travel(sl, path_cancer)
    normalize_cancer = normalize(l)
    print normalize_cancer.shape
    zero_center_cancer = zero_center(normalize_cancer)
    resize_cancer = resize(zero_center_cancer, [128, 128])
    len_cancer = resize_cancer.shape[0]
    print 'The cancer dataset:', len_cancer
    print 'The cancer image:', resize_cancer.shape
    
    path_healthy = '/Users/wuzhenglin/Python_nice/SAL_LUNG/healthy_lung_CT'
    sl_ = []
    l_ = folder_travel(sl_, path_healthy)
    normalize_healthy = normalize(l_)
    zero_center_healthy = zero_center(normalize_healthy)
    resize_healthy = resize(zero_center_healthy, [128, 128])
    len_healthy = resize_healthy.shape[0]  
    print 'The healthy dataset:', len_healthy
    print 'The cancer image:', resize_healthy.shape
    
    
    classes={'cancer','healthy'}
    writer= tf.python_io.TFRecordWriter("cancerANDhealthy.tfrecords") 
    
    step = 1
    count = 0
    
    for index, name in enumerate(classes):
        
        if index == 0:
            print '00000000000000000000000'
            imgset = resize_cancer
            length = len_cancer
#            ind = np.array([1, 0])
            
        else:
            print '11111111111111111111111'
            imgset = resize_healthy
            length = len_healthy
#            ind = np.array([0, 1])
            
        for i in range(length): 
            
            if index == 1:
                count = count + 1
            
            step = step + 1
                      
            pix = imgset[i]
            img_raw = pix.tostring()
            
            example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    })) 
            writer.write(example.SerializeToString()) 

    print step
    print count
    writer.close()
    print 'Writer TFrecorder Finish!'
    
    
if __name__ == '__main__':
    
    make_dataset()
    
    
    
    
    
    
    
    
    