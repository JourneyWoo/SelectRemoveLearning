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
    
    
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1,1,1]):
    
    # Determine current pixel spacing
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
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    verts, faces = measure.marching_cubes(p, threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()


def plot_ct_scan(scan):
    '''
    plot a few more images of the slices
    :param scan:
    :return:
    '''
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(50, 50))
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone)
        


def get_segmented_lungs(im, plot=False):
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image.
    '''
    binary = im < 604
    if plot == True:
        plots[0].axis('off')
        plots[0].set_title('binary image')
        plots[0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].set_title('after clear border')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].set_title('found all connective graph')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].set_title(' Keep the labels with 2 largest areas')
        plots[3].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].set_title('seperate the lung nodules attached to the blood vessels')
        plots[4].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].set_title('keep nodules attached to the lung wall')
        plots[5].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].set_title('Fill in the small holes inside the binary mask of lungs')
        plots[6].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].set_title('Superimpose the binary mask on the input image')
        plots[7].imshow(im, cmap=plt.cm.bone)
    return im

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None
def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0,0,0]
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
    return binary_image


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
    
    
    
    s = load_scan('/Users/wuzhenglin/Python_nice/SAL_LUNG/lung_cancer_CT/DOI')
    print len(s)
    
    
#    path = '/Users/wuzhenglin/Python_nice/SAL_LUNG/healthy/W0001'
#    
#    first_patient = load_scan(path)
#    first_patient_pixels = get_pixels_hu(first_patient)
#    plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
#    plt.xlabel("Hounsfield Units (HU)")
#    plt.ylabel("Frequency")
#    plt.show()
#    print 'Finish HU'
#    pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])
#     pix_resampled_cancer = resample(get_pixels_hu_cancer, slices_cancer, [1,1,1])
#    plot_3d(pix_resampled, 400)
#    print 'Finish 3D'
#    plot_ct_scan(first_patient)
#    get_segmented_lungs(first_patient, plot=False)
    
    

#    image = pix_resampled[100]
#    plt.imshow(image)
#    image = segment_lung_mask(pix_resampled, fill_lung_structures=True)
#    print image[100].shape
#    plt.imshow(image)
    
#    image = normalize(pix_resampled)
#    image = zero_center(image)
    
#    path_cancer = '/Users/wuzhenglin/Python_nice/SAL_LUNG/lung_cancer_CT/DOI/R_004'
#    
#    slices_cancer = load_scan(path_cancer)
#    print '1'
#    
#    get_pixels_hu_cancer = get_pixels_hu(slices_cancer)
#    print '2'
#    print get_pixels_hu_cancer.shape
#    pix_resampled_cancer, spacing = resample(get_pixels_hu_cancer, slices_cancer, [1,1,1])
#    print '3'
#    print pix_resampled_cancer.shape
#    normalize_cancer = normalize(pix_resampled_cancer)
#    print '4'
#    print normalize_cancer.shape
#    zero_center_cancer = zero_center(normalize_cancer)
#    print '5'
#    print zero_center_cancer.shape
#    
#    resize_cancer = resize(zero_center_cancer, [64, 64])
#    
#    print resize_cancer.shape
#  
#    print resize_cancer[10].shape
#    
#    plt.imshow(resize_cancer[10])
    
    
    

    
    