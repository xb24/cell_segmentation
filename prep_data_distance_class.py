# %%
import numpy as np
# import pandas as pd

# import sys
# import cv2
#from skimage.io import imread, imread_collection
# from scipy import signal
# import scipy.misc
# from scipy.misc import imread
# from matplotlib.pyplot import imread
# from cv2 import imread
from skimage.io import imread, imsave

import os
from os import listdir

# %%
def label_map(labels,img_h,img_w, n_labels=3):
    '''Convert a 2D class label into a 3D binary label.

    Inputs: 
        labels(2D numpy.ndarray of int, shape = (img_h,img_w)): the 2D class label
        img_h(int): image height
        img_w(int): image width
        n_labels(int): number of classes.

    Outputs:
        label_map(3D numpy.ndarray of float32, shape = (img_h,img_w,n_labels)): the 3D binary label
    '''
    label_map = np.zeros([img_h, img_w, n_labels], dtype = 'float32')    
    for r in range(img_h):
        for c in range(img_w):
            label_map[r, c, labels[r][c]] = 1
    return label_map

# %%
def prep_data(path_img,path_dist,path_class, n_labels=3):
    '''Load training images and labels.

    Inputs: 
        path_img(str): the folder of training images
        path_dist(str): the folder of training GT distance
        path_class(str): the folder of training GT classes
        n_labels(int): number of classes.

    Outputs:
        data(4D numpy.ndarray of float32, shape = (n,img_h,img_w,nc_img)): all training images
        label_dist(4D numpy.ndarray of float32, shape = (n,img_h,img_w,1)): all GT distance
        label_class(4D numpy.ndarray of float32, shape = (n,img_h,img_w,n_labels)): all GT classes
        img_list(list of str): file names of all training images
    '''
    data = []
    label_dist = []
    label_class = []

    ## Find all files in the specified folders. 
    ## Remove files whose names contain 'GT'. These files are for display purpose. 
    img_list = sorted(listdir(path_img))
    img_list = [x for x in img_list if 'GT' not in x]
    dist_list=sorted(listdir(path_dist))
    dist_list = [x for x in dist_list if 'GT' not in x]
    class_list=sorted(listdir(path_class))
    class_list = [x for x in class_list if 'GT' not in x]
    i=0

    while (i<len(img_list)):
        ## load images
        img = np.array(imread(path_img +'/'+ img_list[i]))
        ## Normalize image with respect to image median.
        img=(img-np.amin(img))*1.0/(np.amax(img)-np.amin(img))
        img=img*1.0/np.median(img)
        img_h=img.shape[0]        
        img_w=img.shape[1]
        img=np.reshape(img,(img_h,img_w,1))
        data.append(img)  
        
        ## load distance
        dist =np.array(imread(path_dist + '/'+ dist_list[i]))
        ## Normalize distnace with respect to distnace median.
        ## This step is optional.
        if np.count_nonzero(dist)!=0:
            nonzero_dist=dist[dist>0]
            dist=dist*1.0/np.median(nonzero_dist)
        dist=np.reshape(dist,(img_h,img_w,1))
        label_dist.append(dist)
        
        ## load classes
        classes = np.array(imread(path_class + '/'+ class_list[i]))
        classes = label_map(classes,img_h,img_w, n_labels)
        label_class.append(classes)
        
        i+=1

    ## Convert list to array
    data=np.array(data)
    label_dist=np.array(label_dist) 
    label_class=np.array(label_class) 
    print(data.shape, label_dist.shape, label_class.shape)
    return data, label_dist, label_class, img_list


# %%
def prep_prediction_data(path_img):
    '''Load test images.

    Inputs: 
        path_img(str): the folder of test images

    Outputs:
        data(4D numpy.ndarray of float32, shape = (n,img_h,img_w,nc_img)): all test images
    '''
    data = []
    img_list = sorted(listdir(path_img))
    img_list = [x for x in img_list if 'GT' not in x]
    i=0

    while (i<len(img_list)):
        ## load images
        img = np.array(imread(path_img +'/'+ img_list[i]))
        ## Normalize image with respect to image median.
        img=(img-np.amin(img))*1.0/(np.amax(img)-np.amin(img))
        img=img*1.0/np.median(img)
        img_h=img.shape[0]        
        img_w=img.shape[1]
        img=np.reshape(img,(img_h,img_w,1))
        data.append(img)
        i+=1

    ## Convert list to array
    data=np.array(data)
    print(data.shape)
    return data, img_list
