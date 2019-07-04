'''
Created on Jun 17, 2019

@author: eljurros
'''
'''
Created on Mar 20, 2019

@author: eljurros
@inspired by medical torch
'''
import cv2 as cv
import os
import sys
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
sys.path.append('../Multi_Organ_Segmentation/DataSet')
from Functions import Crop_Desk_From_Img
from PIL import Image
import random
import h5py
import numpy as np
from torchvision import transforms
train_transforms = transforms.Compose([transforms.ToPILImage(),
                                    transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(100),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()])
class SampleMetadata(object):
    def __init__(self, d=None):
        self.metadata = {} or d

    def __setitem__(self, key, value):
        self.metadata[key] = value

    def __getitem__(self, key):
        return self.metadata[key]

    def __contains__(self, key):
        return key in self.metadata

    def keys(self):
        return self.metadata.keys()


class SegThorDS(Dataset):
    """Segthor dataset.

    :param root_dir: the directory containing the training dataset.
    just a 
    """

    def __init__(self, root_dir):
        '''
        just got the input/GT pairs 
        '''
        self.root_dir = root_dir
        self.filename_pairs = []
        for patient_path, _, files in os.walk(self.root_dir, topdown=False):
            if len(files) > 1:
                input_filename = self._build_train_input_filename(patient_path, 'img')
                gt_filename = self._build_train_input_filename(patient_path, 'mask')
                contour_filename = self._build_train_input_filename(patient_path, 'contour')
                input_filename = os.path.join(patient_path, input_filename)
                gt_filename = os.path.join(patient_path, gt_filename)
                contour_filename = os.path.join(patient_path, contour_filename)
                self.filename_pairs.append((input_filename, gt_filename,
                                            contour_filename))

    @staticmethod
    def _build_train_input_filename(patient_path, im_type='img'):
        '''
        gets the img, gt names
        '''
        basename = os.path.basename(patient_path)
        if im_type == 'img':
            return "{}.nii.gz".format(basename)
        elif im_type == 'mask':
            return "GT.nii.gz"
        elif im_type == 'contour':
            return "CONTOUR.nii.gz"


class SegmentationPair2D(object):
    """
    This class is used to build 2D segmentation datasets. It represents
    a pair of two data volumes (the input data and the ground truth data).

    :param input_filename: the input filename (supported by nibabel).
    :param gt_filename: the ground-truth filename.
    """

    def __init__(self, input_filename, gt_filename, contour_filename):
        self.input_filename = input_filename
        self.gt_filename = gt_filename
        self.contour_filename = contour_filename
        self.input_handle = nib.load(self.input_filename)
        self.gt_handle = nib.load(self.gt_filename)
        self.contour_handle = nib.load(self.contour_filename)
        self.cache = True

        if len(self.input_handle.shape) > 3:
            raise RuntimeError("4-dimensional volumes not supported.")

    def filter_empty_slices(self, input_array, gt_array, contour_array):
        organ_id = 2
        new_input_array = []
        new_gt_array = []
        new_contour_array = []
        for index, pair in enumerate(zip(input_array, gt_array, contour_array)):
            if organ_id in np.unique(pair[1]):
                gt = np.isin(pair[1], organ_id).astype(int)
                new_input_array.append(pair[0])
                new_gt_array.append(gt)
                new_contour_array.append(pair[2])

        return np.array(new_input_array), np.array(new_gt_array), np.array(new_contour_array)
    
    def get_one_to_all_slices(self, input_array, gt_array, organ_id):
        '''
        function that gets heart organ slices, negative samples where the heart does not exist 
        and returns whether the heart is existent or not in a separate array.
        '''
        new_input_array = []
        new_gt_array = []
        heart_exists = []
        for index, pair in enumerate(zip(input_array, gt_array)):
            gt = np.isin(pair[1], organ_id).astype(int)
            new_input_array.append(pair[0])
            new_gt_array.append(gt)
            if organ_id in np.unique(pair[1]):
                heart_exists.append(0)
            else:
                heart_exists.append(-1)

    def all_four_organs(self, input_array, gt_array):
        new_input_array = []
        new_gt_array = []
        for index, pair in enumerate(zip(input_array, gt_array)):
            if len(np.unique(pair[1])) == 5:
                new_input_array.append(pair[0])
                new_gt_array.append(pair[1])

        return np.array(new_input_array), np.array(new_gt_array)
    
    def get_slices(self, input_array, gt_array, contour):
        new_input_array = []
        contour_array = []
        kept_slices = []
        multi_gt_array = []
        for index, pair in enumerate(zip(np.transpose(input_array), np.transpose(gt_array), 
                                         np.transpose(contour))):
            organ_1 = organ_2 = organ_3 = organ_4 = np.zeros((pair[0].shape[0], pair[0].shape[1]))
            if len(np.unique(pair[1])) >1:
                new_input_array.append(pair[0])
                kept_slices.append(index)
                if 1 in np.unique(pair[1]):
                    organ_1 = np.isin(pair[1], 1).astype(int)
                if 2 in np.unique(pair[1]):
                    organ_2 = np.isin(pair[1], 2).astype(int)
                if 3 in np.unique(pair[1]):
                    organ_3 = np.isin(pair[1], 3).astype(int)
                if 4 in np.unique(pair[1]):
                    organ_4 = np.isin(pair[1], 4).astype(int)
                gt_array = [organ_1, organ_2, organ_3, organ_4]
                gt_array = np.array(gt_array).reshape((4,pair[1].shape[0],pair[0].shape[1]))
                multi_gt_array.append(gt_array)
                contour_array.append(pair[2])
    
        return new_input_array, multi_gt_array, contour_array, kept_slices


    def get_pair_shapes(self):
        """Return the tuple (input, ground truth) representing both the input
        and ground truth shapes."""
        input_shape = self.input_handle.header.get_data_shape()

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_shape = None
        else:
            gt_shape = self.gt_handle.header.get_data_shape()

        return input_shape, gt_shape

    def crop_input_data(self, input_array, contour_array, gt_array):

        new_input_array = []
        new_gt_array = []
        new_contour_data = []
        for index, pair in enumerate(zip(input_array, contour_array,
                                         gt_array)):
            #plt.imshow(pair[0])
            #plt.imshow(pair[1])
            cropped_input, cropped_gt, contours = Crop_Desk_From_Img(pair[0], pair[1],
                                                           pair[2])
            new_input_array.append(cv.resize(cropped_input, (400,400)))
            new_gt_array.append(cv.resize(cropped_gt, (400,400)))
            new_contour_data.append(cv.resize(contours, (400,400)))

        return np.array(new_input_array), np.array(new_gt_array), np.array(new_contour_data)
 
    def get_pair_data(self):
        """Return the tuple (input, ground truth) with the data content in
        numpy array.
        applies filtering for empty slices and crops from contours
        """

        cache_mode = 'fill' if self.cache else 'unchanged'
        input_data = self.input_handle.get_fdata(cache_mode, dtype=np.float32)
        contour_data = self.contour_handle.get_fdata(cache_mode, dtype=np.float32)
        # Handle unlabeled data
        if self.gt_handle is None:
            gt_data = None
        else:
            gt_data = self.gt_handle.get_fdata(cache_mode, dtype=np.float32)

        input_data, gt_data, contour_data= self.crop_input_data(np.transpose(input_data),
                                                                 np.transpose(contour_data),
                                                                 np.transpose(gt_data))
    

        total_slices = input_data.shape[0]

        new_input_array,gt_array, contour, kept_slices = self.get_slices(np.transpose(input_data),
                                                            np.transpose(gt_data), np.transpose(contour_data))


        return new_input_array,gt_array, contour, kept_slices

    def get_pair_slice(self, slice_index, slice_axis=2):
        """Return the specified slice from (input, ground truth).

        :param slice_index: the slice number.
        :param slice_axis: axis to make the slicing.
        """
        if self.cache:
            input_dataobj, gt_dataobj = self.get_pair_data()
        else:
            # use dataobj to avoid caching
            input_dataobj = self.input_handle.dataobj

            if self.gt_handle is None:
                gt_dataobj = None
            else:
                gt_dataobj = self.gt_handle.dataobj

        if slice_axis not in [0, 1, 2]:
            raise RuntimeError("Invalid axis, must be between 0 and 2.")

        if slice_axis == 2:
            input_slice = np.asarray(input_dataobj[..., slice_index],
                                     dtype=np.float32)
        elif slice_axis == 1:
            input_slice = np.asarray(input_dataobj[:, slice_index, ...],
                                     dtype=np.float32)
        elif slice_axis == 0:
            input_slice = np.asarray(input_dataobj[slice_index, ...],
                                     dtype=np.float32)

        # Handle the case for unlabeled data
        gt_meta_dict = None
        if self.gt_handle is None:
            gt_slice = None
        else:
            if slice_axis == 2:
                gt_slice = np.asarray(gt_dataobj[..., slice_index],
                                      dtype=np.float32)
            elif slice_axis == 1:
                gt_slice = np.asarray(gt_dataobj[:, slice_index, ...],
                                      dtype=np.float32)
            elif slice_axis == 0:
                gt_slice = np.asarray(gt_dataobj[slice_index, ...],
                                      dtype=np.float32)

            gt_meta_dict = SampleMetadata({
                "zooms": self.gt_handle.header.get_zooms()[:2],
                "data_shape": self.gt_handle.header.get_data_shape()[:2],
            })

        input_meta_dict = SampleMetadata({
            "zooms": self.input_handle.header.get_zooms()[:2],
            "data_shape": self.input_handle.header.get_data_shape()[:2],
        })

        dreturn = {
            "input": input_slice,
            "gt": gt_slice,
            "input_metadata": input_meta_dict,
            "gt_metadata": gt_meta_dict,
        }

        return dreturn


class TwoD_pair_class(object):
    def __init__(self, input_img, gt_img):
        self.input_img = input_img
        self.gt_img = gt_img

        self.input_handle = self.input_img
        self.gt_handle = self.gt_img

    def get_pair_slice(self):
        dictt = {
            "input": self.input_handle,
            "gt": self.gt_handle,
        }
        return dictt


class SegTHor_2D_TrainDS(Dataset):
    def __init__(self, root_dir, transform=None, out_type='mask', filter_type = 'cc',
                 ds_mode='train', gt_mode='norm', RGB=False, size=400, model_type='Unet'):
        '''
        SEgthor dataset class 
        @parameter : 
            @root_dir: the root directory for the dataset default: ../Segthor
            @transform transformations to be implemented on the input:output or ds params, 
            @out_type on which labels we want to train our model on :
                - mask:
                - grb
                - ancillary -cc
                c_labels 
            @filter_type: the type of filter we would like to merge our predictions with as in ancillary training:
                -bb: bounding boxes
                -cc: circular shapes
            @ds_mode: what type of ds are we laoding 
                -val,
                train
                train_ancillary
                -test, 
            @gt_mode= 
            train ds_mode : the labels we are training on do we want them convolved with the output or no
            train_ancillary in ds_mode: the ancillary filters are they convolved with images or no 
            @RGB: Rgb ilage or no,
            @size=400)
            
        @configurations: 
            @ancillary_CC_Conv: 
            model_type = 'BB_Unet'
            filter_type = 'cc'
            out_type_selection = 'mask'
            gt_mode = 'conv'
            ds_mode = 'train_ancillary'
            dice_total = []
            root_dir = '../Segthor'
        '''
        h5_file = 'SegThor_MO_{}.h5'
        self.root_dir = root_dir
        self.filter_type = filter_type
        self.handlers = []
        self.indexes = []
        self.gt_mode = gt_mode
        self.RGB = RGB
        self.out_type = out_type
        self.size = size
        self.model_type = model_type
        self.transform = transform
        self.ds_mode = ds_mode
        
        if ds_mode == 'train':
            self.train_dir = os.path.join(root_dir, h5_file.format('train'))
        elif ds_mode == 'val':
            self.val_dir = os.path.join(root_dir, h5_file.format('val'))
        elif ds_mode == 'test':
            self.test_dir = os.path.join(root_dir, h5_file.format('test'))
        elif ds_mode == 'train_ancillary':
            self.train_dir = os.path.join(root_dir, h5_file.format('train_ancillary'))

        if ds_mode in ['train', 'train_ancillary', 'train_Primary']:
            self.dir = self.train_dir
        if ds_mode== 'val':
            self.dir = self.val_dir
        if ds_mode== 'test':
            self.dir = self.test_dir

        
                

    def __len__(self):
        self.f = h5py.File(self.dir, "r")
        return self.f['img'].shape[0]

    def __getitem__(self, index):
        # input directory
        with h5py.File(self.dir, "r") as self.f:
            self.img_dir = self.f['img']

            # label directory
            if self.out_type == 'mask':
                self.label_dir = self.f['label']
            
                if self.model_type == 'BB_Unet':
                    if self.filter_type == 'bb':
                        self.bbox_dir = self.f['bboxes']
                    if self.filter_type == 'bbconv':
                        self.bbox_dir = self.f['bbox_conv']
                    if self.filter_type == 'cc':
                        self.bbox_dir = self.f['circular_conv']
                    if self.bbox_dir == 'ccconv':
                        self.bbox_dir = self.f['circular']
                
                img = np.array(self.f['img'][index])
                gt = np.array(self.f['label'][index])
                name = str(self.f['img_id'])
                
                if self.transform and self.ds_mode in ['train_ancillary', 'train']:
                    input_img = train_transforms(img)
                    gt_img = train_transforms(gt)
               
                
                if self.model_type == 'Unet':
                    data_dict = {
                        'input': img,
                        'gt': gt,
                        'name': name}
                elif self.model_type == 'BB_Unet':
                    data_dict = {
                    'input': img,
                    'gt': gt,
                    'name': name, 
                    'bboxes':np.array(self.f['bboxes'][index])}
                
                self.f.close()

            
            
        return data_dict




