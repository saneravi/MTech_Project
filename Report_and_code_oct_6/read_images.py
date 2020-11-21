# Ravi Sane
# 12 Sep 2020

import numpy as np
import os
import glob
import cv2

Default_path = '/.'

# Call the function that reads the picture to get the dataset of pictures and labels
def read_img(path=Default_path, w=1200, h=400, c=3):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            # print('reading the images:%s' % (im))
            img = cv2.imread(im)
            img = cv2.resize(img, (w, h))
            imgs.append(img)
            labels.append(idx)
    
    print("ECG Images Read!!!")
    return np.array(imgs), np.asarray(labels, np.int32)
