#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 23:38:24 2017

@author: wuzhenglin
"""
import os
import pandas as pd
import csv
from PIL import Image
import numpy as np
import tensorflow as tf
from scipy import misc
import scipy


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

  
    
def find_all_images_name():
    
    image_path = '/Users/wuzhenglin/NIH/images'
    image_list = []
    image_list = folder_travelwithname(image_list, image_path)
    
    return image_list

def find_all_images_path():
    
    image_path = '/Users/wuzhenglin/NIH/images'
    image_list = []
    image_list = folder_travel(image_list, image_path)
    
    return image_list


def find_image_name(label_name):
    
    name_list = []
    
    csv_path = '/Users/wuzhenglin/NIH/file/Data_Entry_2017.csv'     
   
    with open(csv_path, 'rb') as f:
     
        reader = csv.reader(f)
    
        for row in reader:
            
            if row[2].find(label_name)<>-1:
                name_list.append(row[1])

    return name_list

def find_gender_list(name_list):
    
    gender_list = []
    
    csv_path = '/Users/wuzhenglin/NIH/file/Data_Entry_2017.csv'    
    
    with open(csv_path, 'rb') as f:
        
        reader = csv.reader(f)
        
        for i in range(len(name_list)):
            
            for row in reader:
                
                if row[1].find(name_list[i]):
                    
                    gen = row[6]
                    
                    if gen == 'M':
                        
                        label =1
                    
                    else:
                        
                        label = 0
                    
                    gender_list.append(label) 
                    break
                      
    return gender_list          
    

def makedata_disease_healthy():
    
    disease_name = 'Edema'
    tf_name = 'Edema_healthyordisease.tfrecords'
    disease_name_list = find_image_name(disease_name)
    len_disease_list = len(disease_name_list)
    
    healthy_name = 'No Finding'
    healthy_name_list = find_image_name(healthy_name)
    len_healthy_list = len(healthy_name_list)    
    
    
    all_image_path = find_all_images_path()
    len_all_image_path = len(all_image_path)
    
    my_disease_image_path_list = []
    my_healthy_image_path_list = []
    
    for disease_name_in in range(len_disease_list):
        
        every_disease_name = disease_name_list[disease_name_in]
        
        for all_image_path_in in range(len_all_image_path):
           
            if all_image_path[all_image_path_in].find(every_disease_name)<>-1:
                
                index_of_all_image_path = all_image_path_in
                my_disease_image_path_list.append(all_image_path[index_of_all_image_path])
                break
    
    len_my_disease_image_path_list = len(my_disease_image_path_list)
    
    
    for healthy_name_in in range(len_healthy_list):
        
        every_healthy_name = healthy_name_list[healthy_name_in]
        
        for all_image_path_in in range(len_all_image_path):
           
            if all_image_path[all_image_path_in].find(every_healthy_name)<>-1:
                
                index_of_all_image_path = all_image_path_in
                my_healthy_image_path_list.append(all_image_path[index_of_all_image_path])
                break
    
    len_my_healthy_image_path_list = len(my_healthy_image_path_list)    
    
    print 'csv 中，疾病共: ', len_disease_list
    print 'csv 中，健康共: ', len_healthy_list
    
    print '找到 疾病图片: ', len_my_disease_image_path_list
    print '找到 健康图片: ', len_my_healthy_image_path_list
    
    writer= tf.python_io.TFRecordWriter(tf_name) 
    
    for i in range(0, 2):
        
        if i == 0:
        #disease   
            imgset = my_disease_image_path_list
            length = len_my_disease_image_path_list
            label = 1
        else:
        #healthy
            imgset = my_healthy_image_path_list 
#            length = len_my_healthy_image_path_list
            length = 500
            label = 0
            length = 200
        
        for j in range(length):
            print j
            im = Image.open(imgset[j])
            data = im.getdata()
            width, height = im.size
            data = np.matrix(data, dtype = 'float') / 255.0
            sh = data.shape
            if sh[0] != 1:
                print '遇见 RGB！！！！！！！！！！！！！！！！！！！！！！！！！'
                data = data[:, [0]]
           
            new_data = np.reshape(data, (height, width))            
            new_data = scipy.misc.imresize(new_data, [128, 128])            
            print '调整图片尺寸为:', new_data.shape
            
            img_raw = new_data.tostring()
            
            example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    })) 
            
            writer.write(example.SerializeToString()) 
            
    writer.close()
    print '制作完成'
    
    
    
def makedata_healthy_gender():
    
    tf_name = 'healthy_gender.tfrecords'
    
    print '制作健康图片 list'
    healthy_name = 'No Finding'
    healthy_name_list = find_image_name(healthy_name)
    len_healthy_list = len(healthy_name_list)  
    
    print '制作性别 list'    
    gender_list_csv = find_gender_list(healthy_name_list)
    len_gender_list_csv = len(gender_list_csv)
    
    print 'csv 中，健康共: ', len_healthy_list
    print 'csv 中，性别共: ', len_gender_list_csv
    
    gender_list = []
    num_name_list = []
    
    print '制作所有图片路径 list'
    all_image_path = find_all_images_path()
    len_all_image_path = len(all_image_path)
    print '所有图片共', len_all_image_path
    
    my_healthy_image_path_list = []
    
    print '制作健康图片路径 list'
    
    for healthy_name_in in range(len_healthy_list):
        
        every_healthy_name = healthy_name_list[healthy_name_in]
        
        for all_image_path_in in range(len_all_image_path):
           
            if all_image_path[all_image_path_in].find(every_healthy_name)<>-1:
                
                index_of_all_image_path = all_image_path_in
                my_healthy_image_path_list.append(all_image_path[index_of_all_image_path])
                num_name_list.append(every_healthy_name)
                break
    
    len_my_healthy_image_path_list = len(my_healthy_image_path_list) 
    gender_list = find_gender_list(num_name_list)
    len_gender_list = len(gender_list)
    print '完成！制作健康图片路径 list'
    print '找到 健康图片: ', len(num_name_list)
    print '找到 健康图片: ', len_my_healthy_image_path_list
    print '找到 gender: ', len_gender_list
    
    
    print '***********************开始制作数据集*********************************'
    writer= tf.python_io.TFRecordWriter(tf_name) 
    
    for i in range(0, 1):
        
        imgset = my_healthy_image_path_list
        labelset = gender_list
        length = len_healthy_list
#        length = 200
        
        for j in range(length):
            
            im = Image.open(imgset[j])
            data = im.getdata()
            width, height = im.size
            print width, height
            data = np.matrix(data, dtype = 'float') / 255.0
            print data.shape
            print imgset[j]
            sh = data.shape
            
            if sh[0] != 1:
                print '遇见 RGB！！！！！！！！！！！！！！！！！！！！！！！！！'
                data = data[:, [0]]
                
            new_data = np.reshape(data, (height, width))    
            print '原始尺寸为:', new_data.shape
            new_data = scipy.misc.imresize(new_data, [128, 128])            
            
            print '调整图片尺寸为:', new_data.shape
            print '******************', j
            
            label = labelset[j]
            
            img_raw = new_data.tostring()
            
            example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    })) 
            
            writer.write(example.SerializeToString()) 
            
    writer.close()
    print '***********************制作 gender_healthy 数据集完成*********************************'
 
    
    
    
    
if __name__ == '__main__':
    
#    makedata_healthy_gender()
    makedata_disease_healthy()
    