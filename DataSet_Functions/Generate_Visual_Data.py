'''
Created on Jun 18, 2019

@author: eljurros
'''
import h5py
import numpy as np
import matplotlib.pyplot as plt
f = h5py.File('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/DataSet/SegThor_MO_val.h5', "r")

print f.keys()
img = f['bboxes']
label = f['label']
label_estimate = f['circular']
print img.shape
import cv2 as cv
print img.shape, img.dtype
var = 0
for i in range(497):
    print i
    print img[i]
    # plt.hist(img[i].ravel(),500,[np.min(img[i]),np.max(img[i])])
    # plt.imshow(contour[i])

print var/2164
    
    
    