#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 01:56:13 2017

@author: wuzhenglin
"""

import os 
import tensorflow as tf 
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np
from Brainpreprocess import get_pixels_hu, normalize, zero_center
import scipy
import pandas as pd
import dicom
import scipy
import SimpleITK as sitk
from scipy import misc


def findgender(dcm_name, overview_df):
    

    a = overview_df[overview_df['name'] == dcm_name].head()['idx']
    b = np.asarray(a)[0]
    print b
    c = overview_df[overview_df['idx'] == b].head()['gender']
    d = np.asarray(c)[0]

    return d
        
    
   
def folder_travelwithname(s, folder_path):
    
    files = os.listdir(folder_path)
    
    for each in files:
        
        if (each[0] == '.'):
            pass
        
        else:
        
            flag = os.path.isdir(os.path.join(folder_path, each))
        
            if flag:
                path = folder_path + '/' + each
                s = folder_travelwithname(s, path)
                
            else:
                f = each
                iter_f = iter(f)
                str = ''
                for line in iter_f:
                    str = str + line
                s.append(str)
          
         
    
    return s



def make_dataset():
    
    path_csv = '/Users/wuzhenglin/Python_nice/SAL_BRAIN/brain_healthy_dataset/gender.csv'
    overview_df = pd.read_csv(path_csv)
    overview_df.columns = ['idx']+list(overview_df.columns[1:])
    overview_df['gender'] = overview_df['gender'].map(lambda x: 1 if x else 0)
    
    path_cancer = '/Users/wuzhenglin/Python_nice/SAL_BRAIN/brain_healthy_dataset/data_test'   
    lit = []
    lis = folder_travelwithname(lit, path_cancer)

    
    
    writer= tf.python_io.TFRecordWriter("brain_gender_test.tfrecords") 
    

    for i in range(len(lis)):
        
        dcm_name = lis[i]
        print dcm_name
        lab = findgender(dcm_name, overview_df)
        dcm_path = path_cancer + '/' + dcm_name
        img = scipy.misc.imresize(sitk.GetArrayFromImage(sitk.ReadImage(dcm_path))[35], [128, 128])
        normalize_cancer = normalize(img)
        zero_center_cancer = zero_center(normalize_cancer)
        resize_cancer = scipy.misc.imresize(zero_center_cancer, [128, 128])   
        img_raw = resize_cancer.tostring()
        
        example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[lab])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    })) 
        writer.write(example.SerializeToString())  
        
    
    writer.close()
    print 'Writer TFrecorder Finish!'   
        
        
  
   
if __name__ == '__main__':
    
    make_dataset()
    
    
    
    
  