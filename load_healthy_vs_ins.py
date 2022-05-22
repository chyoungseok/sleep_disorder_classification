#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 17:27:38 2020

@author: syook
"""
############## import modules ##############
from __future__ import print_function
from cgi import test

import os

from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.callbacks import ModelCheckpoint
# from livelossplot.tf_keras import PlotLossesCallback
# from livelossplot import PlotLossesKeras

import pandas as pd # for using pandas daraframe
import numpy as np # for som math operations

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import cys_loading
import cys_utils
#############################################


############# GPU selection #################
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#############################################

# load train_data, train_label
binary_h_i = cys_loading.binary_h_i()
train_hypnos = binary_h_i.load_input()
train_labels = binary_h_i.load_label() # 1: healthy, 0: Insomnia

# train_test_split
print("train_test_splie --> split rate=0.1")
x_train, x_test, y_train, y_test = train_test_split(train_hypnos, train_labels, test_size=0.1, random_state=42)


#############################################
########### kernel size experiment ##########
list_kernel_size = [(2,2)]
str_kernel_size = ['(2,2)']
for j in range(3,10):
    ker_nel_len = j
    list_kernel_size.append((2,ker_nel_len))
    str_kernel_size.append(str((2,ker_nel_len)))
#############################################

test_acc = []
# In[ ]:


# from livelossplot import PlotLossesKerasTF
# from livelossplot.outputs import MatplotlibPlot
############# Train model ###################
for i in tqdm(range(len(list_kernel_size))):
    model = cys_utils.get_model(kernel_size=list_kernel_size[i])

    path_weights = './weights/ins_%s_wights' % str(list_kernel_size[i])
    # checkpoint = ModelCheckpoint(path_weights, monitor='val_loss', save_best_only=True,
    #                             save_weights_only=True, mode='min', verbose=0)
    checkpoint = ModelCheckpoint(path_weights, mode='min', save_best_only=True, verbose=0)
    
    # callbacks = [PlotLossesCallback(), checkpoint]#, reduce_lr]
    # save the train_loss plot
    path_figure = './figure/ins_%s.png' % str(list_kernel_size[i])
    # callbacks = [PlotLossesKerasTF(outputs=[MatplotlibPlot(figpath=path_figure)]), checkpoint]
    callbacks = [checkpoint]
    
    batch_size=32
    epochs=20

    H = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        callbacks=callbacks,
        validation_split=0.1,
        epochs=epochs
        # class_weight=class_weight
        )
    
    cys_utils.plot_acc_loss(H, path_figure, epochs, str_kernel_size)
    
    y_true = y_test
    y_pred = np.argmax(model.predict(x_test), 1)
    
    print(metrics.classification_report(y_true, y_pred))
    # df_metrics = pd.DataFrame(metrics.classification_report(y_true, y_pred))
    # df_metrics.to_csv('./metric/osa_%s.csv' % str(list_kernel_size))

    plt.figure()
    labels = ['Insomnia', 'Healthy']
    ax= plt.subplot()
    sns.heatmap(metrics.confusion_matrix(y_true, y_pred, normalize='true'), annot=True, ax = ax, cmap=plt.cm.Blues); #annot=True to annotate cells
    # labels, title and ticks
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);
    plt.savefig('./confusion_matrix/ins_%s.png' % str(list_kernel_size[i]))

    test_acc.append(metrics.accuracy_score(y_true, y_pred))

df_test_acc = pd.DataFrame(test_acc, index=str_kernel_size, columns=['test_acc'])
df_test_acc.to_csv('test_acc_ins.csv')
# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




