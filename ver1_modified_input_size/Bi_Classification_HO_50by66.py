#!/usr/bin/env python
# coding: utf-8

# In[11]:


#!#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 17:27:38 2020

@author: syook
"""

from __future__ import print_function
import numpy as np
#from keras.utils import np_utils
#from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
#import numpy as np
import inspect, os, sys
# print(os.getcwd())
import cys_loading_50by66
from tqdm import tqdm

from keras.models import Sequential
import tensorflow
from keras.layers import TimeDistributed
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Activation, Dropout, Reshape
from keras.layers.wrappers import TimeDistributed
from keras.models import model_from_json
from keras.layers import LSTM, Dense, Activation, ThresholdedReLU, MaxPooling2D, Embedding, Dropout,Conv2D, Flatten
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd # for using pandas daraframe
import numpy as np # for som math operations
from sklearn.preprocessing import StandardScaler # for standardizing the Data
import math
import h5py as h5
import glob

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.utils import plot_model
import resnet2D
# import resnet3D
# import resnet1D

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#%% input arguments
model_id = 3 # 3=resnet
model_dim = 2
no_layer= 3#1=18, 2=34, 3=50, 4=101, 5=152
num_class = 2  # if 1, regression
input_shape = (1,50,66) # modified input size (original=(1,5,660))

#%%Resnet3D

if model_id == 3:     # Resnet
   if model_dim == 1:  # 1D
       from resnet1D import ResnetBuilder
       if no_layer==1:
           model = ResnetBuilder.build_resnet_18((20, 1), num_class)
       elif no_layer==2:
           model = ResnetBuilder.build_resnet_34((20, 1), num_class)
       elif no_layer==3:
           model = ResnetBuilder.build_resnet_50((20, 1), num_class)
       elif no_layer==4:
           model = ResnetBuilder.build_resnet_101((20, 1), num_class)
       elif no_layer==5:
           model = ResnetBuilder.build_resnet_152((20, 1), num_class)
        
        
   elif model_dim == 2:  # 2D
       from resnet2D import ResnetBuilder
       if no_layer==1:
           model = ResnetBuilder.build_resnet_18(input_shape, num_class)
       elif no_layer==2:
           model = ResnetBuilder.build_resnet_34(input_shape, num_class)
       elif no_layer==3:
           model = ResnetBuilder.build_resnet_50(input_shape, num_class)
       elif no_layer==4:
           model = ResnetBuilder.build_resnet_101(input_shape, num_class)
       elif no_layer==5:
           model = ResnetBuilder.build_resnet_152(input_shape, num_class)

   elif model_dim == 3:  # 3D
       from resnet3D import Resnet3DBuilder
       if no_layer==1:
           model = Resnet3DBuilder.build_resnet_18((68, 95, 79, 1), num_class)
       elif no_layer==2:
           model = Resnet3DBuilder.build_resnet_34((20, 20, 20, 1), num_class)
       elif no_layer==3:
           model = Resnet3DBuilder.build_resnet_50((20, 20, 20, 1), num_class)
       elif no_layer==4:
           model = Resnet3DBuilder.build_resnet_101((20, 20, 20, 1), num_class)
       elif no_layer==5:
           model = Resnet3DBuilder.build_resnet_152((20, 20, 20, 1), num_class)

model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# load train_data, train_label
binary_h_o = cys_loading_50by66.binary_h_o()
train_hypnos = binary_h_o.load_input()
train_labels = binary_h_o.load_label() # 1: healthy, 0: OSA

print("train data size: {}".format(train_hypnos.shape))
print("train label size: {}".format(train_labels.shape))

# train_test_split
print("train_test_splie --> split rate=0.1")
x_train, x_test, y_train, y_test = train_test_split(train_hypnos, train_labels, test_size=0.1, random_state=42)

print("train set size: {}".format(x_train.shape))
print("test set size: {}".format(x_test.shape))


# Class weight setting

# load 'Final_sub_ID_grouped.csv'
df_grouped_class = pd.read_csv('./sub_info/Final_sub_ID_grouped.csv')

num_healthy = df_grouped_class['Healthy'].dropna().size
num_osa = df_grouped_class['OSA'].dropna().size
total=num_healthy+num_osa

# Get class weights
weight_for_0 = (1 / num_osa) * (total / 2.0) # OSA
weight_for_1 = (1 / num_healthy) * (total / 2.0) # Healthy

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))


from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from livelossplot.tf_keras import PlotLossesCallback

checkpoint = ModelCheckpoint("model_weights_model_osa.h5", monitor='val_loss',
                             save_weights_only=True, mode='min', verbose=0)
callbacks = [PlotLossesCallback(), checkpoint]#, reduce_lr]

model.fit(
    x_train,
    y_train,
    batch_size=32,
    callbacks=callbacks,
    validation_split=0.1,
    epochs=50,
    # class_weight=class_weight
    )


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns

y_true = y_test
y_pred = np.argmax(model.predict(x_test), 1)
print(metrics.classification_report(y_true, y_pred))
print("Classification accuracy: %0.6f" % metrics.accuracy_score(y_true, y_pred))


# In[ ]:


import matplotlib.pyplot as plt
labels = ['OSA', 'Healthy']

ax= plt.subplot()
sns.heatmap(metrics.confusion_matrix(y_true, y_pred, normalize='true'), annot=True, ax = ax, cmap=plt.cm.Blues); #annot=True to annotate cells

# labels, title and ticks
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);


# In[ ]:




