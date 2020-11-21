# Ravi Sane
# Sep 12 2020
from my_ecg_plot import plot, save_as_jpg
import numpy as np
import matplotlib.pyplot as plt
import os

counter1 = 1
counter2 = 1
default_path = './'

def create(signals, fields, images_required = 30, path = default_path, display_factor = 1.5, sampling_freq=1000, line_color = (0,0,0), line_width = 0.5):
    """
    signals: input signal array having 12 lead ecg data
    fields: it has patient details as per ptbdb
    images_required: total images of each class
    path: where to store images
    display_factor: resize images
    sampling_freq: sampling frequencies
    """
    records = len(signals) 
    counter1 = 1
    counter2 = 1       
    
    newpath = path+'ptbdb'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        
    dir_path1 = newpath+'/'+'Myocardial infarction/'
    if not os.path.exists(dir_path1):
        os.makedirs(dir_path1)
        
    dir_path2 = newpath+'/'+'Healthy control/'
    if not os.path.exists(dir_path2):
        os.makedirs(dir_path2)
    

    for patient in range(records):
        if(fields[patient]['comments'][4] == 'Reason for admission: Myocardial infarction' and counter1 <= images_required):        
            ecg = np.transpose(np.array(signals[patient]))
            plot(ecg, 
               sample_rate = sampling_freq, 
               # title = 'Myocardial infarction', 
               columns = 4, 
               display_factor = display_factor,
               plot_line_color = line_color,
               line_width =line_width)                        
                
            save_as_jpg(f'MI_ecg_{counter1}', dir_path1)
            counter1 += 1                        
            
            
        elif(fields[patient]['comments'][4] == 'Reason for admission: Healthy control'and counter2 <= images_required):
            ecg = np.transpose(np.array(signals[patient]))
            plot(ecg, 
               sample_rate = sampling_freq, 
               # title = 'Healthy control', 
               columns = 4, 
               display_factor = display_factor,
               plot_line_color = line_color,
               line_width =line_width)                        
            save_as_jpg(f'HC_ecg_{counter2}', dir_path2)
            counter2 += 1
    
    print("ECG Graph Created")
