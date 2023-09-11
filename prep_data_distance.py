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
def prep_data(path_img,path_gt):
    '''Load training images and labels.

    Inputs: 
        path_img(str): the folder of training images
        path_gt(str): the folder of training labels

    Outputs:
        data(4D numpy.ndarray of float32, shape = (n,img_h,img_w,nc_img)): all training images
        label(4D numpy.ndarray of float32, shape = (n,img_h,img_w,nc_mask)): all training labels
        img_list(list of str): file names of all training images
    '''
    data = []
    label = []

    ## Find all files in the specified folders. 
    ## Remove files whose names contain 'GT'. These files are for display purpose. 
    img_list = sorted(listdir(path_img))
    img_list = [x for x in img_list if 'GT' not in x]
    gt_list=sorted(listdir(path_gt))
    gt_list = [x for x in gt_list if 'GT' not in x]
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
        
        ## load labels
        gt =np.array(imread(path_gt + '/'+ gt_list[i]))
        ## Normalize distnace with respect to distnace median.
        ## This step is optional.
        if np.count_nonzero(gt)!=0:
            nonzero_gt=gt[gt>0]
            gt=gt*1.0/np.median(nonzero_gt)

        gt=np.reshape(gt,(img_h,img_w,1))
        label.append(gt)
        
        i+=1

    ## Convert list to array
    data=np.array(data)
    label=np.array(label) 
    print(data.shape, label.shape)
    return data, label, img_list    


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
