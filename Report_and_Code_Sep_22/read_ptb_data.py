# Ravi Sane
# 12 Sep 2020

# to read ptb db
import wfdb
import numpy as np
from scipy.signal import lfilter, iirnotch

# input the data from the ptb dataset
def read_data(data, dataset_path, M=25, each_lead_time=2.5, num_samples=10000, num_leads = 12, 
            images_per_record=10, power_line_freq=50, bandwidth=10):
    '''
    num_samples = 10000     # no of samples to read from each  record
    num_leads = 12         # 12 leads out of given 15 to be choosen
    sampling_freq = 1000   # 1000 Hz
    each_lead_time = 2.5   # 2.5 seconds
    M = 25                 # mov avg filter
    images_per_record = 10
    power_line_freq = 50
    
    '''
    
    fields = []
    signals = []
    Q = power_line_freq/bandwidth
    
    for i in data:
        sig, f = wfdb.rdsamp(dataset_path + i[:-1], channels=[x for x in range(num_leads)], sampfrom=0, sampto=num_samples)
        sampling_freq = f['fs']
        
        b, a = iirnotch(power_line_freq, Q, sampling_freq)        

        s = np.zeros((num_samples-M+1, num_leads))
        for leads in range(num_leads):
            sig[:,leads] = lfilter(b, a, sig[:,leads])
            s[:,leads] = np.convolve(sig[:,leads], np.ones((M,))/M, mode='valid')

        if(f['comments'][4] == 'Reason for admission: Myocardial infarction' or f['comments'][4] == 'Reason for admission: Healthy control'):    
            for copies in range(images_per_record):
                if (each_lead_time*sampling_freq*(copies+1)) < len(s):
                    s_temp = s[int(each_lead_time*sampling_freq*copies):int(each_lead_time*sampling_freq*(copies+1))]   # 2.5 sec interval is selected
                    s_temp = s_temp - s_temp.mean(axis=0)              # remove baseline drift
                    signals += [s_temp]
                    fields += [f]
    
    print("Dataset Read!!!")
    return signals, fields
