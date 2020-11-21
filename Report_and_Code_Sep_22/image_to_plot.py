# Ravi Sane
# 12 Sep 2020

import numpy as np
import cv2

# function to convert 1 page ecg chart to individual 12 lead ECG signals
def get_12_ecgs(img, thresh = 125, crop=2, rows=3, cols=4):    
    
    # convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    maxval = 255
    
    # filter out pixels above threshold values to get binary image
    (T, bin_img) = cv2.threshold(gray_img, thresh, maxval, cv2.THRESH_BINARY_INV)
    
    # crop to remove boundaries
    bin_img = bin_img[crop:-crop, crop:-crop]
    
    # divide into 12 parts
    L, B = np.shape(bin_img)
    l, b = int(L/rows), int(B/cols)
    
    num_leads = rows*cols

    ecg_img = np.zeros((num_leads, l, b), dtype=int)
    count = 0
    for i in range(rows):
        for j in range(cols):
            ecg_img[count] = bin_img[i*l:(i+1)*l, j*b:(j+1)*b]
            count += 1
    return ecg_img
    
# function to convert 12 individual ecgs images to data points (can be stored in csv format)
def graph_to_plot(img):    
    l, y, x = np.shape(img)
    plot = np.zeros((l, x))
        
    for i in range(l):
        for k in range(x):
            for j in range(y):
                if img[i, j, k] == 255:
                    plot[i, k] = (y-j)
                    break
            if 255 not in img[i, :, k]:
                plot[i, k] = plot[i, k-1]
        
        # normalize the plot
        plot[i] = (plot[i] - np.mean(plot[i]))/np.std(plot[i])
        
        # delete boundary values pixels
    return plot[:,3:-3]    
    
# function to filter the data by moving avg filter
def moving_avg_filt(data_in, M=3):
    leads, length = np.shape(data_in)
    data_out = np.zeros((leads, length-M+1))
    
    #convolution with ones to do MA filtering
    for i in range(leads):
        data_out[i] = np.convolve(data_in[i], np.ones((M,))/M, mode='valid')
    return data_out
