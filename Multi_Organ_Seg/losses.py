'''
Created on Jun 19, 2019

@author: eljurros
'''
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
'''def dice_loss(input,target):
    """
    This function computes the sum of soft dice scores for a multiclass 2D segmentation task
    @param input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    @param target is the groundtruth, shoud have same size as the input
    """
    EPSILON=1e-8
    for i in range(4):
        pred = input[:,i,:,:]
        tr = target[:,i,:,:]
        num = torch.sum(pred*tr)+EPSILON
        denum = np.sum(pred) + np.sum(tr) + EPSILON
        d = -2.00*(num/denum)'''
        
        
def dice_loss(input,target):
    """
    This function computes the sum of soft dice scores for a multiclass 2D segmentation task
    @param input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    @param target is the groundtruth, shoud have same size as the input
    """
    EPSILON=1e-8
    input = F.softmax(input,1)
    num = torch.sum(input*target)+EPSILON
    denum = torch.sum(input) + torch.sum(target) + EPSILON
    d = -2.00*(num/denum)
    
    return d
        
        
        