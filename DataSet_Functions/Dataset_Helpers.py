'''
Created on Jun 17, 2019

@author: eljurros
'''
'''
Created on Mar 22, 2019

@author: ROsanaEL Jurdi
'''
import numpy as np


def Crop_Desk_From_Img(img, contours, gt_img):
    '''
    Gets an input image as well as tecountour and crops it.
    returns the cropped image
    '''

    inputt = np.tile(-1000, (img.shape[0], img.shape[0]))
    inputt[contours == np.max(contours)] = img[contours == np.max(contours)]

    # clip the image
    inputt[np.where(inputt < -1000)] = -1000
    inputt[np.where(inputt > 3000)] = 3000

    # Now crop
    (x, y) = np.where(contours == np.max(contours))
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    inputt = inputt[topx:bottomx+1, topy:bottomy+1]
    gt = gt_img[topx:bottomx+1, topy:bottomy+1]
    contours = contours[topx:bottomx+1, topy:bottomy+1]

    # normalize:
    mean = np.mean(inputt)
    std = np.std(inputt)
    norm_inputt = (inputt - mean)

    return norm_inputt, gt, contours
