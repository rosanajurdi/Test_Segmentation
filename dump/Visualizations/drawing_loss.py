def read_results(v):
    '''
    returns the list of infor,ation present within the line read 
    in the for, of a dictionary
    '''
    
    eval_list = v.split(',,,')
    for string_slice in eval_list:
        if 'tr_loss' in string_slice:
            tr_loss = string_slice.split('tr_loss:')[1]
        if 'val_loss' in string_slice:
            val_loss = string_slice.split('val_loss:')[1]
        if 'tr_dice' in string_slice:
            tr_dice = string_slice.split('tr_dice:')[1]
        if 'val_dice' in string_slice:
            val_dice = string_slice.split('dice:')[1]
        if 'epoch' in string_slice:
            epoch = string_slice.split('epoch:')[1]

    dictt = {'tr_loss': tr_loss, 'val_loss': val_loss,
             'tr_dice': tr_dice, 'val_dice': val_dice,
             'epoch': epoch}
    return dictt
import matplotlib.pyplot as plt
import numpy as np
def get_results(txt_file_path):
    i = 0
    logs = []
    v_logs = []
    error_tr = 0
    with open(txt_file_path, 'r') as the_file:
        lines = the_file.read()
        for l in lines.split('\n'):
            try:
           
                dict = l.split('###')
                epoch = np.int(dict[0])
                logs.append(np.float(dict[1]))
                v_logs.append(np.float(dict[2]))
                """if i == np.int(epoch):
                    error_tr += np.float(dict[1])
                else: 
                    i += 1    
                    logs.append(error_tr)
                    error_tr = 0"""
            except:
                pass
        
        f = plt.figure()
        epoch = range(0,len(logs))
        plt.plot(epoch,logs, 'b')
        plt.plot(epoch, v_logs, 'r')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['train loss', 'validation loss'])
        plt.show()
        print('hi')
                
        
file = '/home/eljurros/spare-workplace/Multi_Organ_Segmentation/DataSet/results_MO_i.txt'


d = get_results(file)
