# Ravi Sane
# 12 Sep 2020

# to read ptb db
import wfdb
import numpy as np
from scipy.signal import lfilter, iirnotch
import os

# input the data from the ptb dataset
def read_data(data, dataset_path, M=25, each_lead_time=2.5, num_samples=10000, num_leads = 12, 
            images_per_record=10, power_line_freq=50, bandwidth=10, scale=1):
    '''
    num_samples = 10000     # no of samples to read from each  record
    num_leads = 12         # 12 leads out of given 15 to be choosen
    sampling_freq = 1000   # 1000 Hz
    each_lead_time = 2.5   # 2.5 seconds
    M = 25                 # mov avg filter
    images_per_record = 10
    power_line_freq = 50
    scale = 1              # for downsampling
    
    '''
    new_path = './data_folder'
    if not os.path.exists(new_path):
        os.makedirs(new_path)    
    
    fields = []
    signals = np.zeros((1, 2500//scale, 12))
    labels = np.zeros((1))
    Q = power_line_freq/bandwidth
    counting = 0
    
    for i in data:
        counting += 1
        print(counting, end=', ')        
            
        sig, f = wfdb.rdsamp(dataset_path + i[:-1], channels=[x for x in range(num_leads)])###########, sampfrom=0, sampto=num_samples)
        sampling_freq = f['fs']
        
        b, a = iirnotch(power_line_freq, Q, sampling_freq)        

        s = np.zeros((len(sig)-M+1, num_leads))
        for leads in range(num_leads):
            sig[:,leads] = lfilter(b, a, sig[:,leads])
            s[:,leads] = np.convolve(sig[:,leads], np.ones((M,))/M, mode='valid')

        if(f['comments'][4] == 'Reason for admission: Myocardial infarction'): 
            for copies in range(images_per_record):
                if (each_lead_time*sampling_freq*(copies+1.75)) < len(s):
                    s_temp = s[int(each_lead_time*sampling_freq*copies):int(each_lead_time*sampling_freq*(copies+1)):scale]   # 2.5 sec interval is selected
                    s_temp = s_temp - s_temp.mean(axis=0)              # remove baseline drift
                    signals = np.append(signals, s_temp.reshape(1,2500//scale,12), axis=0)
#                     fields += [f]
                    labels = np.append(labels, [1])
                else:
                    break
                    
        
        elif (f['comments'][4] == 'Reason for admission: Healthy control'):
            for copies in range(images_per_record):
                if (each_lead_time*sampling_freq*(copies+1.75)) < len(s):
                    s_temp = s[int(each_lead_time*sampling_freq*copies):int(each_lead_time*sampling_freq*(copies+1)):scale]   # 2.5 sec interval is selected
                    s_temp = s_temp - s_temp.mean(axis=0)              # remove baseline drift
                    signals = np.append(signals, s_temp.reshape(1,2500//scale,12), axis=0)
#                     fields += [f]
                    labels = np.append(labels, [0])
		
                    # additional samples in between
                    s_temp = s[int(each_lead_time*sampling_freq*(copies+0.25)):int(each_lead_time*sampling_freq*(copies+1.25)):scale]   # 2.5 sec interval is selected
                    s_temp = s_temp - s_temp.mean(axis=0)              # remove baseline drift
                    signals = np.append(signals, s_temp.reshape(1,2500//scale,12), axis=0)
#                     fields += [f]
                    labels = np.append(labels, [0])

                    # additional samples in between
                    s_temp = s[int(each_lead_time*sampling_freq*(copies+0.5)):int(each_lead_time*sampling_freq*(copies+1.5)):scale]   # 2.5 sec interval is selected
                    s_temp = s_temp - s_temp.mean(axis=0)              # remove baseline drift
                    signals = np.append(signals, s_temp.reshape(1,2500//scale,12), axis=0)
#                     fields += [f]
                    labels = np.append(labels, [0])

                    # additional samples in between
                    s_temp = s[int(each_lead_time*sampling_freq*(copies+0.75)):int(each_lead_time*sampling_freq*(copies+1.75)):scale]   # 2.5 sec interval is selected
                    s_temp = s_temp - s_temp.mean(axis=0)              # remove baseline drift
                    signals = np.append(signals, s_temp.reshape(1,2500//scale,12), axis=0)
                    fields += [f]
                    labels = np.append(labels, [0])
                else:
                    break
        if not(counting%10):
            np.save(new_path+'/'+f'data{counting//10}', signals[1:])
            np.save(new_path+'/'+f'label{counting//10}', labels[1:])
            del(signals)
            del(labels)
            signals = np.zeros((1, 2500//scale, 12))
            labels = np.zeros((1))
            
    print("Dataset Read!!!")
#     signals = np.array(signals)
#     labels = np.array(labels)
    np.save(new_path+'/'+f'data{counting//10}', signals[1:])
    np.save(new_path+'/'+f'label{counting//10}', labels[1:])
    del(signals)
    del(labels)
    
    return fields
