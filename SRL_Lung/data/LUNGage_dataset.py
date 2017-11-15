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
from LUNGpreprocess import get_pixels_hu, normalize, zero_center
import scipy
import pandas as pd
import dicom


def findage(dcm_name, overview_df):
    
    
    a = overview_df[overview_df['dicom_name'] == dcm_name].head()['idx']
    b = np.asarray(a)[0]
    c = overview_df[overview_df['idx'] == b].head()['Age']
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
    
    path_csv = '/Users/wuzhenglin/Python_nice/SAL_LUNG/siim-medical-image-analysis-tutorial/overview.csv'
    overview_df = pd.read_csv(path_csv)
    overview_df.columns = ['idx']+list(overview_df.columns[1:])
    
    
    path_cancer = '/Users/wuzhenglin/Python_nice/SAL_LUNG/siim-medical-image-analysis-tutorial/dicom_dir'   
    lit = []
    lis = folder_travelwithname(lit, path_cancer)

    
    
    writer= tf.python_io.TFRecordWriter("age_train.tfrecords") 
    
    
    
    
    
    for i in range(0, 100):
        
        dcm_name = lis[i]
        lab = findage(dcm_name, overview_df)
        print lab
        dcm_path = path_cancer + '/' + dcm_name
        img = dicom.read_file(dcm_path)
        pix = img.pixel_array
        normalize_cancer = normalize(pix)
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
    
    
    
    
  