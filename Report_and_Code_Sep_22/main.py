#!/usr/bin/env python
# coding: utf-8

# Import Libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2


# Define path

# In[2]:


dataset_path = '/home/ravisane/Downloads/MTP/ptbdb/'
data = open('/home/ravisane/Downloads/MTP/ptbdb/RECORDS.txt', 'r')


# Define init values

# In[3]:


num_samples = 10000     # no of samples to read from each  record
records = 549          # total records
num_leads = 12         # 12 leads out of given 15 to be choosen
sampling_freq = 1000   # 1000 Hz
images_required = 25 #################################changed from 250
each_lead_time = 2.5   # 2.5 seconds
mov_avg_filt = 25      # mov avg filter
display_factor = 1.5  # to scale the image
line_width = 1.5
images_per_record = 10


# Storing the resultant into signals and fields.

# In[4]:


# input the data from the ptb dataset
from read_ptb_data import read_data

signals, fields = read_data(data=data, dataset_path=dataset_path, 
                            M=mov_avg_filt, each_lead_time=each_lead_time,
                            num_samples=num_samples, num_leads=num_leads, 
                            images_per_record=images_per_record)# call fn to read PTB db
sampling_freq = fields[0]['fs']


# In[5]:


# to create ecg records from ptb db
from create_images import create

create(signals, fields, 
       images_required=images_required, path='./', 
       display_factor=display_factor, 
       sampling_freq=sampling_freq, 
       line_color = (0,0,0),
        line_width=1.5) # call fn to create ECG graphs


# In[6]:


# to read created ecg images
from read_images import read_img

# Resize all pics
w = 1800
h = 600
c = 3
path = './ptbdb/'


# In[7]:


data, label = read_img(path=path, w=w, h=h, c=c) # call the function to read the data


# In[8]:


plt.imshow(data[1])


# In[9]:


# to convert read images to plots ie having 1D data
from image_to_plot import get_12_ecgs, graph_to_plot, moving_avg_filt

# parameters
image_intensity_threshold = 125
crop_pixels_out = 2
moving_avg_value = 3


# In[10]:


# call above three functions
data_recovered = []
for i in range(len(data)):
    ecg_images_extracted = get_12_ecgs(data[i], 
                                       thresh=image_intensity_threshold, 
                                       crop=crop_pixels_out)
    
    raw_data = graph_to_plot(ecg_images_extracted)
    
    data_recovered += [moving_avg_filt(raw_data, M=moving_avg_value)]
    
data_recovered = np.array(data_recovered)
print("ECG images converted to plot data...Done")


# In[11]:


del(data) # to free up memory
# save recovered data & labels in files
np.save('data', data_recovered)
np.save('label', label)


# In[12]:


# Plot graph of recovered signals of jth record
j = 1
plt.figure()
if label[j]:
    plt.suptitle("Myocardial infarction")
else:
    plt.suptitle("Healthy Control")
for leads in range(num_leads):         
    plt.subplot(3, 4, leads+1)
    plt.plot(range(len(data_recovered[j, leads])), data_recovered[j, leads])
plt.show()    


# In[13]:


# read recovered data and convert it to proper format for signals
signals = np.transpose(data_recovered, axes=(0, 2, 1))
new_sampling_freq = int(np.shape(data_recovered)[2]/(each_lead_time))
sampling_freq = new_sampling_freq


# Show plot of one record

# In[14]:


#%% Plot the graph of all 12 ECG signals of jth patient record number
j = 0
t = np.arange(0, (len(signals[j])/sampling_freq), (1/ sampling_freq))
plt.plot(t, signals[j])
plt.title("ECG signal(Recovered from Graph)")
plt.xlabel("Time in s")
plt.ylabel("Amplitude")
plt.show()  


# Performing 6 level wavelet decomposition using db9 wavelet

# In[16]:


import pywt
from biorth_wavelet import custom_wavelet
dec_levels = 6         # decomposition levels
wave = custom_wavelet()

records = len(signals)
coeff = []

for patient in range(records):
    coeff += [pywt.wavedec(signals[patient], 
                           wavelet = wave, 
                           mode = 'zero', 
                           level = dec_levels, 
                           axis = 0)]


# Plotting the subband coeff for jth record

# In[17]:


j = 0
c = coeff[j]
plt.figure(1)

for i in range(7):
    plt.subplot(4,2,i+1)
    t = np.arange(0, len(c[i]), 1)
    plt.plot(t, c[i])
    if i==0:
        plt.title("cA6 Signal")
    else:
        plt.title(f"cD{7-i} Signal")    
plt.show()


# Calculate energy vectors for each record, we choose 4 subbands * 12 ecg leads => 48 long vector
# Normalize this energy vector

# Get eigen values of covariance matrix of 4 subband coefficient and find principal eigen values (we choose 6 values) => 24 long eigen vector

# Concatenate the two vectors to get 72 length vector

# In[20]:


from z_feature_vectore import find_Z_feature
max_eig_values = 6     # top 6 eigen values to be taken of cov matrix of 4 subbands coeff
max_dec_levels = 4     # 4 subbands choosen cA6 cD6 cD5 cD4

# Get feature matrix by 48 energy values and 24 eigen values
Z = find_Z_feature(coeff, 
                   max_dec_levels=max_dec_levels, 
                   max_eig_values=max_eig_values,
                   num_leads=num_leads)


# Apply 5 fold CV and clasifiers KNN, linear SVM, rbf SVM

# In[42]:


#%% Apply 5 fold CV and clasifiers KNN, linear SVM, rbf SVM
from sklearn.model_selection import KFold
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

scores_knn = []
scores_linear = []
scores_rbf = []

neigh = KNeighborsClassifier(n_neighbors = 5)
svc_linear = SVC(kernel = 'linear', C=30.0)
#svc_rbf = SVC(kernel = 'rbf')

#svc_linear = make_pipeline(StandardScaler(), SVC(kernel = 'linear'))
svc_rbf = make_pipeline(StandardScaler(), SVC(kernel = 'rbf', C=30.0))

cv = KFold(n_splits = 5, shuffle = True)

for train_index, test_index in cv.split(Z):
    X_train, X_test, y_train, y_test = Z[train_index], Z[test_index], label[train_index], label[test_index]
    
    # train classifiers    
    neigh.fit(X_train, y_train)
    scores_knn.append(neigh.score(X_test, y_test))
    
    svc_linear.fit(X_train, y_train)
    scores_linear.append(svc_linear.score(X_test, y_test))
    
    svc_rbf.fit(X_train, y_train)
    scores_rbf.append(svc_rbf.score(X_test, y_test))

# print avg scores  
print(np.mean(scores_knn))
print(np.mean(scores_linear))
print(np.mean(scores_rbf))


# Print classification report

# In[43]:


from sklearn.metrics import classification_report
target_names = ['HC','MI']
y_pred_KNN = neigh.predict(X_test)
y_pred_lsvc = svc_linear.predict(X_test)
y_pred_rsvc = svc_rbf.predict(X_test)

print('KNN classification_report')
print(classification_report(y_test, y_pred_KNN, target_names=target_names))
print('Linear SVM classification_report')
print(classification_report(y_test, y_pred_lsvc, target_names=target_names))
print('RBF SVM classification_report')
print(classification_report(y_test, y_pred_rsvc, target_names=target_names))


# Print Confusion Matrix

# In[44]:


import pandas as pd
import seaborn as sn

print("KNN Confusion Matrix")
data = {'y_Actual':    y_pred_KNN,
        'y_Predicted': y_test
        }

df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True)
plt.show()

print("Linear SVM Confusion Matrix")
data = {'y_Actual':    y_pred_lsvc,
        'y_Predicted': y_test
        }

df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True)
plt.show()

print("RBF SVM Confusion Matrix")
data = {'y_Actual':    y_pred_rsvc,
        'y_Predicted': y_test
        }

df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True)
plt.show()


# In[45]:


np.shape(data_recovered)


# In[27]:


from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(data_recovered, label, test_size = 0.2) 


# In[46]:


from DNN import dnn_model
import tensorflow as tf

final_model = dnn_model(data=data_recovered, 
                        rb1=True, rb2=False, rb3=False)

tf.keras.utils.plot_model(final_model, "dnn.jpg", show_shapes=True)


# In[31]:


opt = 'adam'# tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, name='SGD')
loss = 'binary_crossentropy' # tf.keras.losses.BinaryCrossentropy(from_logits=True)
final_model.compile(optimizer=opt,loss=loss, metrics=["accuracy"])


# In[32]:


final_model.fit(
    {"ecg1": train_X[:,0,:].reshape(-1,1,c),"ecg2": train_X[:,1,:].reshape(-1,1,c),"ecg3": train_X[:,2,:].reshape(-1,1,c),
     "ecg4": train_X[:,3,:].reshape(-1,1,c),"ecg5": train_X[:,4,:].reshape(-1,1,c),"ecg6": train_X[:,5,:].reshape(-1,1,c),
     "ecg7": train_X[:,6,:].reshape(-1,1,c),"ecg8": train_X[:,7,:].reshape(-1,1,c),"ecg9": train_X[:,8,:].reshape(-1,1,c),
     "ecg10": train_X[:,9,:].reshape(-1,1,c),"ecg11": train_X[:,10,:].reshape(-1,1,c),"ecg12": train_X[:,11,:].reshape(-1,1,c)},
    {"label": train_y},
    validation_split=0.2,
    epochs=60,
    batch_size=128,
    callbacks=tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=10)
)


# In[33]:


final_model.evaluate(
    {"ecg1": test_X[:,0,:].reshape(-1,1,c),"ecg2": test_X[:,1,:].reshape(-1,1,c),"ecg3": test_X[:,2,:].reshape(-1,1,c),
     "ecg4": test_X[:,3,:].reshape(-1,1,c),"ecg5": test_X[:,4,:].reshape(-1,1,c),"ecg6": test_X[:,5,:].reshape(-1,1,c),
     "ecg7": test_X[:,6,:].reshape(-1,1,c),"ecg8": test_X[:,7,:].reshape(-1,1,c),"ecg9": test_X[:,8,:].reshape(-1,1,c),
     "ecg10": test_X[:,9,:].reshape(-1,1,c),"ecg11": test_X[:,10,:].reshape(-1,1,c),"ecg12": test_X[:,11,:].reshape(-1,1,c)},
    {"label": test_y})


# In[34]:


from tensorflow.keras import layers
model = tf.keras.Sequential([
    layers.Input((b,c)),
    layers.Conv1D(32, 11, 1, 'same', activation= 'relu'),
    layers.BatchNormalization(),
    layers.Conv1D(32, 5, 1, 'same', activation= 'relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    
    layers.Conv1D(32, 11, 1, 'same', activation= 'relu'),
    layers.BatchNormalization(),
    layers.Conv1D(32, 5, 1, 'same', activation= 'relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    
#     layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='softmax')
    ])
#%%
# model.summary()
# #%%
# from tensorflow.keras.losses import CategoricalCrossentropy
# model.compile(optimizer='adam', 
#               loss=CategoricalCrossentropy(from_logits=True), 
#               metrics=['accuracy'])
# history = model.fit(train_ds, validation_data=val_ds, epochs=3)



model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_X.reshape(-1,b,c), train_y, validation_split=0.2, epochs=20, 
          callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=3))

#%%
test_loss, test_acc = model.evaluate(test_X.reshape(-1,b,c),  test_y, verbose=2)

print('\nTest accuracy:', test_acc)


# In[36]:




