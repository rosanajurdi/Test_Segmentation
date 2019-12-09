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

sys.path.append('../medicaltorch-0.2')
    
gray_scale = True
typ = 'train_ancillary'

i = 0
total_slices = 0
filtered_slices = 0
root_path = '/home/eljurros/Desktop/code_caroline/CT_all_cleaned'
training_dataset_path = os.path.join(root_path, typ)

Segthor_ds = SegThorDS(root_dir=training_dataset_path)
total = len(Segthor_ds.filename_pairs)
size = 0
for patient_path in Segthor_ds.filename_pairs:
    i = i + 1
    patient_name = os.path.basename(patient_path[0])
    
    input_filename, gt_filename, contour_filename = patient_path[0], \
                                                    patient_path[1], \
                                                    patient_path[2]

    Slicer = SegmentationPair2D(input_filename,
                                gt_filename,
                                contour_filename)
    
    input,_, _,_ = Slicer.get_pair_data()
    size += len(input)
    print (patient_name, len(input), size)
    
print size