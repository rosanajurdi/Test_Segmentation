'''
Created on Apr 10, 2019

@author: eljurros

'''
import os
import cv2
import random as rng
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2 as cv
    
def get__mask(gt, typ ='bbox'):
    bboxes = []
    for mask_id in range(4):
        bbox = extract_bbox(gt[mask_id])
        
        if typ == 'circular':
            r, c = get_circle_attributes_from_bb(bbox)
            bboxes.append([r,c])
        else:
            bboxes.append(bbox)
        
        if typ == 'circular':
            noisy_mask = circ_mask((400,400), bboxes)
        else:
            noisy_mask = rect_mask((400,400), bboxes)
        
    return noisy_mask

def circ_mask(shape, bboxes):
    i = 0
    label_img = np.zeros(shape, np.uint8)
    color = (255, 0, 255)
    for attr in bboxes:
        if len(np.unique(attr[1])) != 1: 
            i = i+1
        if i >2:
            pass
        r,c = attr[0],attr[1]
        cv.circle(label_img, (int(c[0]), int(c[1])), int(np.ceil(r)), color, cv.FILLED)
    return label_img

def rect_mask(shape, bboxes, mode = 'norm'):
    """Given a bbox and a shape, creates a mask (white rectangle foreground, black background)
    Param:
        shape: shape (H,W) or (H,W,1)
        bbox: bbox numpy array [y1, x1, y2, x2]
    Returns:
        mask
    """
    i = 0
    mask = np.zeros(shape[:2], np.uint8)
    for bbox in bboxes:
        if len(np.unique(bbox)) != 1: 
            i = i+1
        if i >2:
            pass
        mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 255

    return mask

def extract_bbox(mask, order='y1x1y2x2'):
    """Compute bounding box from a mask.
    Param:
        mask: [height, width]. Mask pixels are either >0 or 0.
        order: ['y1x1y2x2' | ]
    Returns:
        bbox numpy array [y1, x1, y2, x2] or tuple x1, y1, x2, y2.
    """
    horizontal_indicies = np.where(np.any(mask, axis=0))[0]
    vertical_indicies = np.where(np.any(mask, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        x2 += 1
        y2 += 1
    else:
        x1, x2, y1, y2 = 0, 0, 0, 0
    if order == 'x1y1x2y2':
        return x1, y1, x2, y2
    else:
        return ([int(y1), int(x1), int(y2), int(x2)])


def get_circle_attributes_from_bb(bbox):
    x1 = bbox[1]
    x2 = bbox[3]
    y1 = bbox[0]
    y2 = bbox[2]
    # r = np.max([np.abs(x2 - x1), np.abs(y2 - y1)])
    r = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    cx = np.divide((x1 + x2), 2)
    cy = np.divide(y1 + y2, 2)
    return int(r/2), (int(cx), int(cy))

