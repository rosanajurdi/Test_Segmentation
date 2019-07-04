import sys
import os
import torch.nn as nn
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from DataSet import DataSEt_Classes
from torch import optim
import time
import tqdm
import torch
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
import warnings
import models, losses
from torch.autograd import Variable
from metrics import dice_score, dice_per_organ
from torch.utils.data import DataLoader
filter_type = 'bb'
out_type_selection = 'mask'
import matplotlib.pyplot as plt
import gc
gt_mode = 'norm'
size = 400
import numpy as np
model_type = 'Unet'
initial_lr = 0.001
num_epochs = 550
opt_loss = 10000

def threshold(array):
    array = (array > 0.89) * 1.0
    return array
# print(torch.cuda.is_available())
# print(torch.cuda.current_device())
dirr = '/home/eljurros/spare-workplace/Multi_Organ_Segmentation/DataSet'

if model_type == 'Unet':
    model = models.Unet(drop_rate=0.4, bn_momentum=0.1)
    model = torch.load('/home/eljurros/Desktop/myria_scripts/checkpnt_BEST.ckpt')

elif model_type == 'BB_Unet':
    model = models.BB_Unet(drop_rate=0.4, bn_momentum=0.1)


optimizer = optim.Adam(model.parameters(), lr=initial_lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
train_transform = None
txt_file_path ='../Multi_Organ_Segmentation/results_MO_newest_edition_2.txt'       

dice = organ_0 = organ_1 = organ_2 = organ_3 = 0
start_time = time.time()

dataset = DataSEt_Classes.SegTHor_2D_TrainDS(dirr, transform=True,out_type=out_type_selection,model_type = 'Unet', ds_mode='train', gt_mode=gt_mode, RGB=False, filter_type=filter_type, size=400)  
j = 0
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    loader = DataLoader(dataset, batch_size=1, shuffle=True,
                              pin_memory=False, num_workers=1)
    model.eval()
    print len(loader)
    for i, batch in enumerate(loader):
        input_samples, gt_samples = batch["input"], batch["gt"]
        # plt.figure()
        # plt.imshow(np.squeeze(batch["input"]))
        # plt.figure()
        #plt.imshow(np.argmax(np.squeeze(batch["gt"]), axis = 0))
        

        var_input = input_samples.reshape(-1, 1, size, size)
        var_input = torch.tensor(var_input, dtype=torch.float)
        var_input = Variable(var_input, requires_grad=True)

        if model_type == 'Unet':
            preds = model(var_input)
            # plt.figure()
            # plt.imshow(preds.detach().numpy()[0][0], axis = 0)
        elif model_type == 'BB_Unet':
            var_bbox = batch["bboxes"].reshape(-1, 1, size, size)
            var_bbox = torch.tensor(var_bbox, dtype=torch.float)
            var_bbox = Variable(var_bbox, requires_grad=True)
            preds = model(var_input, var_bbox)
        # fig = plt.figure()
        """    
        rows = 3
        columns = 4
        for i in range(4):
            axes = []
            axes.append(fig.add_subplot(rows,columns,i+1))
            plt.imshow(np.squeeze(batch["input"]))
    
        for _, layer in enumerate(np.squeeze(preds)):
            i = i + 1
            x = layer.detach().numpy() 
            axes.append(fig.add_subplot(rows,columns,i+1))
            plt.imshow(x)
            
            
        for _, layer in enumerate(np.squeeze(batch["gt"])):
            i = i + 1
            x = layer.detach().numpy()
            axes.append(fig.add_subplot(rows,columns,i+1))
            plt.imshow(x)
            
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        """
        
        # plt.savefig('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/features/{}.png'.format(layer[1]))
            # loss = losses.dice_loss(preds.to('cpu'), gt_samples)
            # print('loss for batch {} on epoch {} is {}'.format(i,epoch,loss))
        predicted_output = threshold(preds)
        predicted_output = predicted_output.type(torch.float32)
        try:
            ou = 0
            if len(np.unique(gt_samples[0][ou])) == 2:
                j += 1
                d = dice_per_organ(predicted_output, gt_samples, ou)
                organ_0 += d
                print i, d
            
            '''d_metric, dict= dice_score(predicted_output, gt_samples)
            if np.isnan(d_metric) == False:
                organ_0 += dict['organ_0']
                organ_1 += dict['organ_1']
                organ_2 += dict['organ_2']
                organ_3 += dict['organ_3']
                dice += d_metric 
                print i , dict['organ_0'], dict['organ_1'],dict['organ_2'],dict['organ_3']   '''
        except:
            pass
        
    print  np.float(organ_0)/j  
           
        
        
    # print np.float(dice)/2164.0,np.float(organ_0)/2164, np.float(organ_1)/2164, np.float(organ_2)/2164, np.float(organ_3)/2164
        
        
            
       
                
