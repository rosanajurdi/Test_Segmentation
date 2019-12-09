import os
from os import walk
from segment import circular_label_estimate,elliptical_label_estimate
i = 0
for (dirpath, dirnames, filenames) in walk('../Segthor/val/img'):
    for img in filenames:
        i = i + 1
        img_path = os.path.join(dirpath, img)
        circular_label_estimate(img_path, mode='conv')
        print ('generating clabels {}/{} done'.format(i, len(filenames)))