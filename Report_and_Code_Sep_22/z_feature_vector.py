# Ravi Sane
# 12 Sep 2020
import numpy as np



def find_Z_feature(coeff, max_dec_levels=4, max_eig_values=6, num_leads=12):
    '''
    dec_levels = 6         # decomposition levels
    max_eig_values = 6     # top 6 eigen values to be taken of cov matrix of 4 subbands coeff
    max_dec_levels = 4     # 4 subbands choosen cA6 cD6 cD5 cD4
    num_leads = 12
    '''    
    
    records=len(coeff)
    
    # Calculate energy vectors for each record, we choose 4 subbands * 12 ecg leads => 48 long vector
    # Normalize this energy vector
    
    energy = np.zeros((records, max_dec_levels, num_leads))
    energy_vector = np.zeros((records, max_dec_levels*num_leads))
    normalized_energy_vector = np.zeros((records, max_dec_levels*num_leads))

    for patient in range(records):
        for levels in range(max_dec_levels):               # 4 levels to keep that is cA6 cD6 cD5 cD4
            for leads in range(num_leads):
                temp_sum = 0
                for xx in coeff[patient][levels][:][leads]:
                    temp_sum += xx
                energy[patient,levels,leads] = temp_sum/len(coeff[patient][levels][:][leads])
        
        energy_vector[patient,:] = np.reshape(energy[patient,:,:], (4*num_leads)) # 48 long vector
        normalized_energy_vector[patient] = energy_vector[patient]/max(energy_vector[patient])
    
    
    # Get eigen values of covariance matrix of 4 subband coefficient and find principal eigen values (we choose 6 values) => 24 long eigen vector
    
    eig_value_vector = np.zeros((records, max_dec_levels*max_eig_values))
    # normalized_eig_value_vector = np.zeros((records, max_dec_levels*max_eig_values))

    for patient in range(records):
        princ_eig_value = []
        for levels in range(max_dec_levels):
            eig_value, eig_vector = np.linalg.eig(np.cov(np.array(coeff[patient][levels])))
            princ_eig_value = princ_eig_value + [np.sort(np.real(eig_value))[-max_eig_values:]]                  
        eig_value_vector[patient] = np.reshape(np.array(princ_eig_value),(max_dec_levels*max_eig_values))
        # normalized_eig_value_vector[patient] = eig_value_vector[patient]#/max(eig_value_vector[patient])
        
    # Get feature matrix by 48 energy values and 24 eigen values
    Z = np.concatenate((normalized_energy_vector, eig_value_vector), axis = 1)
    
    return Z
    

