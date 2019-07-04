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
from metrics import dice_score
from torch.utils.data import DataLoader
filter_type = 'bb'
out_type_selection = 'mask'
import matplotlib.pyplot as plt
import gc
gt_mode = 'norm'
size = 400
import numpy as np
model_type = 'BB_Unet'
initial_lr = 0.001
num_epochs = 550
opt_loss = 10000

def threshold(array):
    array = (array > 0.5) * 1.0
    return array
# print(torch.cuda.is_available())
# print(torch.cuda.current_device())
# dirr = '../Multi_Organ_Segmentation/DataSet'
dirr = '/home/eljurros/spare-workplace/Multi_Organ_Segmentation/DataSet'

if model_type == 'Unet':
    model = models.Unet(drop_rate=0.4, bn_momentum=0.1)
    # model = torch.load('../Multi_Organ_Segmentation/Ancillary_MO/checkpnt_BEST')

elif model_type == 'BB_Unet':
    model = models.BB_Unet(drop_rate=0.4, bn_momentum=0.1)


optimizer = optim.Adam(model.parameters(), lr=initial_lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
train_transform = None
txt_file_path ='../Multi_Organ_Segmentation/results_MO_BB_Unet.txt'       

d_prev = d_metric = 0
for epoch in range(0, num_epochs+1):
    start_time = time.time()
    
    for phase in ['train_ancillary', 'val']:
        dataset = DataSEt_Classes.SegTHor_2D_TrainDS(dirr, transform=train_transforms,out_type=out_type_selection,model_type = 'BB_Unet', ds_mode=phase, gt_mode=gt_mode, RGB=False, filter_type=filter_type, size=400)
        
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
    
            loader = DataLoader(dataset, batch_size=3, shuffle=True,
                                      pin_memory=False, num_workers=0)
    
            if phase in ['train', 'train_ancillary']:
                
                model.train()
                scheduler.step()
                lr = scheduler.get_lr()[0]
                train_loss_total = 0.0
                tr_dice_total = 0
                tr_len = len(loader)
            elif phase == 'val':
                model.eval()
                val_loss_total = 0
                val_dice_total = 0
                val_len = len(loader)
  
            for i, batch in enumerate(loader):
                try:
                    # print('before loading: {}'.format(torch.cuda.max_memory_allocated()))
    
                    # model = model.to('cuda')
                    print (phase + ' on batch : ' + str(i) + ' from ' + str(len(loader)))
                    # input_samples, gt_samples = batch["input"].to('cuda'), batch["gt"]
                    input_samples, gt_samples = batch["input"], batch["gt"]
    
                    # print('after loading: {}'.format(torch.cuda.max_memory_allocated()))
    
                    var_input = input_samples.reshape(-1, 1, size, size)
                    var_input = torch.tensor(var_input, dtype=torch.float)
                    var_input = Variable(var_input, requires_grad=True)
    
    
                    with torch.set_grad_enabled(phase in ['train', 'train_ancillary']):
                        if model_type == 'Unet':
                            preds = model(var_input)
                              
                        elif model_type == 'BB_Unet':
                            var_bbox = batch["bboxes"].reshape(-1, 1, size, size)
                            var_bbox = torch.tensor(var_bbox, dtype=torch.float)
                            var_bbox = Variable(var_bbox, requires_grad=True)
                            preds = model(var_input, var_bbox)
             
                        loss = losses.dice_loss(preds.to('cpu'), gt_samples)
                        print('loss for batch {} on epoch {} is {}'.format(i,epoch,loss))
    
                     
                    predicted_output = threshold(preds.to('cpu'))
                    predicted_output = predicted_output.type(torch.float32)
                    d_metric, dict= dice_score(predicted_output.to('cpu'), gt_samples.to('cpu'))
           
                    
                    if phase in ['train_ancillary','train']: 
                        train_loss_total += loss
                        tr_dice_total += d_metric
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        print ('training on batch' + str(i) + 'with accuracy' + str(d_metric))
                    else:
                        val_loss_total += loss
                        val_dice_total += d_metric
                        print('validating on batch : ' + str(i) + 'with accuracy' + str(d_metric))
                    
                except:
                    print('missed batch')
                    pass     
                                                        
    average_val_loss = np.float(val_loss_total)/val_len
    average_tr_loss = np.float(train_loss_total)/tr_len
    
    with open(txt_file_path, 'a') as the_file:
        the_file.write('{}###{}###{}'.format(epoch, average_tr_loss,  average_val_loss))
        the_file.write('\n') 
    
    print('{}###{}###{}'.format(epoch, average_tr_loss,  average_val_loss))
    
    if opt_loss > average_val_loss:
        opt_loss = average_val_loss
        name = 'checkpnt_BEST_BB_Unet.ckpt'
        chkpt_path = '../Multi_Organ_Segmentation/Ancillary_MO'

        if os.path.exists(chkpt_path) is False:
            os.mkdir(chkpt_path)

        save_path = os.path.join(chkpt_path, name)
        torch.save(model.to('cpu'), save_path)
        print ('model saved-{}-{}'.format(epoch, opt_loss))
        


                
                