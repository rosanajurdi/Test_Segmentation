'''
Created on Mar 20, 2019

@author: eljurros
'''
from DataSEt_Classes import SegThorDS, SegmentationPair2D
import imageio
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from DataSet.Label_Estimate_Helper_Functions import extract_bbox, rect_mask, get__mask
sys.path.append('../medicaltorch-0.2')
gray_scale = True
typ = 'val'

i = 0
total_slices = 0
filtered_slices = 0
root_path = '/home/eljurros/Desktop/code_caroline/CT_all_cleaned'
training_dataset_path = os.path.join(root_path, typ)

Segthor_ds = SegThorDS(root_dir=training_dataset_path)
total = len(Segthor_ds.filename_pairs)
  
size = 699
file = h5py.File('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/DataSet/SegThor_MO_{}.h5'.format(typ), "w")
img_dataset = file.create_dataset('img', (size,400,400))
label_dataset = file.create_dataset('label', (size,4,400,400))
slice_name = file.create_dataset('img_id', (size,1), dtype="S10")
bbox_im_dir = file.create_dataset('bboxes', (size,400,400))
circular = file.create_dataset('circular', (size,400,400))
circular_conv = file.create_dataset('circular_conv', (size,400,400))
bbox_conv = file.create_dataset('bbox_conv', (size,400,400))
j = 0
size = 0
def get_convolution(img, mask):
    return img*mask

for i,patient_path in enumerate(Segthor_ds.filename_pairs):
    patient_name = os.path.basename(patient_path[0])
    input_filename, gt_filename, contour_filename = patient_path[0], \
                                                    patient_path[1], \
                                                    patient_path[2]

    Slicer = SegmentationPair2D(input_filename,
                                gt_filename,
                                contour_filename)
    
    
    input_data, gt_array, contour, kept_slices = Slicer.get_pair_data()    
    size += len(input_data)
    print (patient_name, len(input_data), size)
    for _,triple in enumerate(zip(input_data, gt_array, contour, kept_slices)):
        img_slice,gt,cnt = triple[0], triple[1], triple[2]
        # print('saving onto index :{}'.format(i+j))
        img = img_slice
        gt_data = gt
        rectmask= get__mask(triple[1], 'bboxes')
        circmask= get__mask(triple[1], 'circular')
        
        file['img'][i+j]= img
        file['label'][i+j] = gt_data
        file['bboxes'][i+j] = rectmask
        file['circular'][i+j] = circmask
        file['circular_conv'][i+j] = get_convolution(circmask,img)
        file['bbox_conv'][i+j] = get_convolution(rectmask,img)
        patient_name = os.path.basename(patient_path[0]).split('.nii')[0]
        sn = '{}_{}'.format(patient_name, triple[3])
        file['img_id'][i+j] = sn
        """plt.imsave('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/DataSet/tmp/{}img.png'.format(sn),img)
        plt.imsave('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/DataSet/tmp/{}gt.png'.format(sn),np.argmax(triple[1], axis=0))
        plt.imsave('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/DataSet/tmp/{}bbox.png'.format(sn), rectmask)"""
        
        j += 1
file.close()    