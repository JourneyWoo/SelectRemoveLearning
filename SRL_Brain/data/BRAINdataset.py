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
from Brainpreprocess import load_scan, get_pixels_hu, resample, normalize, zero_center, folder_travel
import scipy
import SimpleITK as sitk
from scipy import misc

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
    
    path_cancer = '/Users/wuzhenglin/Python_nice/SAL_BRAIN/brain_unhealthy/brain_test'
    
    l = load_scan(path_cancer)

    ll = get_pixels_hu(l)
    print ll.shape
    normalize_cancer = normalize(ll)
    print normalize_cancer.shape
    zero_center_cancer = zero_center(normalize_cancer)
    resize_cancer = resize(zero_center_cancer, [128, 128])
    len_cancer = resize_cancer.shape[0]
    print 'The cancer dataset:', len_cancer
    print 'The cancer image:', resize_cancer.shape
    
    
    path_healthy = '/Users/wuzhenglin/Python_nice/SAL_BRAIN/brain_healthy_dataset/data_test'
    
    s = []
    l_ = folder_travel(s, path_healthy)
    image_ = np.stack([scipy.misc.imresize(sitk.GetArrayFromImage(sitk.ReadImage(p))[35], [128, 128]) for p in l_])
    
    normalize_healthy = normalize(image_)
    print normalize_healthy.shape
    zero_center_healthy = zero_center(normalize_healthy)
    resize_healthy = resize(zero_center_healthy, [128, 128])
    len_healthy = resize_healthy.shape[0]  
    print 'The healthy dataset:', len_healthy
    print 'The healthy image:', resize_healthy.shape
    
    
    classes={'cancer','healthy'}
    writer= tf.python_io.TFRecordWriter("brain_test.tfrecords") 
    
    step = 1
    count = 0
    
    for index, name in enumerate(classes):
        
        if index == 0:
            print '00000000000000000000000'
            imgset = resize_cancer
            length = len_cancer

            
        else:
            print '11111111111111111111111'
            imgset = resize_healthy
            length = len_healthy

            
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
    
    
    
    
    
    
    
    
    