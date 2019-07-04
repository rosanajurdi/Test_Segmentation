from collections import defaultdict

from scipy import spatial
import numpy as np
import torch.nn.functional as F
import torch
def threshold_predictions(predictions, thr=0.999):
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds

def dice_score_2(prediction, groundtruth):
    pflat = prediction.flatten()
    gflat = groundtruth.flatten()
    d = (1 - spatial.distance.dice(pflat, gflat)) * 100.0
    return d

class MetricManager(object):
    def __init__(self, metric_fns):
        self.metric_fns = metric_fns
        self.result_dict = defaultdict(float)
        self.num_samples = 0 
    
    def __call__(self, prediction, ground_truth):
        self.num_samples += len(prediction)
        for metric_fn in self.metric_fns:
            for p, gt in zip(prediction, ground_truth):
                res = metric_fn(p, gt)
                dict_key = metric_fn.__name__
                self.result_dict[dict_key] += res
            
    def get_results(self):
        res_dict = {}
        for key, val in self.result_dict.items():
            res_dict[key] = val / self.num_samples
        return res_dict
    
    def reset(self):
        self.num_samples = 0
        self.result_dict = defaultdict(float)
        

def numeric_score(prediction, groundtruth):
    """Computation of statistical numerical scores:

    * FP = False Positives
    * FN = False Negatives
    * TP = True Positives
    * TN = True Negatives

    return: tuple (FP, FN, TP, TN)
    """
    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))
    return FP, FN, TP, TN

def dice_per_organ(prediction, groundtruth, i):
    pred = prediction[:,i,:,:]
    tr = groundtruth[:,i,:,:]
    num = torch.sum(pred*tr)
    denum = torch.sum(pred) + torch.sum(tr)
    d = 200*(num/denum)
    
    return d

def dice_score(prediction, groundtruth):
    dice = []
    dice_dict = {}
    for i in range(4):
        pred = prediction[:,i,:,:]
        tr = groundtruth[:,i,:,:]
        num = torch.sum(pred*tr)
        denum = torch.sum(pred) + torch.sum(tr)
        d = 200*(num/denum)
        dice.append(d.detach().numpy())
    
    dice_dict = {'organ_0': dice[0], 'organ_1': dice[1], 
              'organ_2':dice[2], 'organ_3': dice[3]}
    return np.mean(dice), dice_dict


def jaccard_score(prediction, groundtruth):
    pflat = prediction.flatten()
    gflat = groundtruth.flatten()
    return (1 - spatial.distance.jaccard(pflat, gflat)) * 100.0


def hausdorff_score(prediction, groundtruth):
    return spatial.distance.directed_hausdorff(prediction, groundtruth)[0]


def precision_score(prediction, groundtruth):
    # PPV
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FP) <= 0.0:
        return 0.0

    precision = np.divide(TP, TP + FP)
    return precision * 100.0


def recall_score(prediction, groundtruth):
    # TPR, sensitivity
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FN) <= 0.0:
        return 0.0
    TPR = np.divide(TP, TP + FN)
    return TPR * 100.0


def specificity_score(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TN + FP) <= 0.0:
        return 0.0
    TNR = np.divide(TN, TN + FP)
    return TNR * 100.0


def intersection_over_union(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FP + FN) <= 0.0:
        return 0.0
    return TP / (TP + FP + FN) * 100.0


def accuracy_score(prediction, groundtruth):
    pred_thresh = threshold_predictions(prediction)
    FP, FN, TP, TN = numeric_score(pred_thresh, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)
    return accuracy * 100.0
