## Module for loadind train data

## import modules ------------------------------

import os
from tqdm import tqdm
import pandas as pd
import numpy as np

## global namespace ------------------------------
# path_cwd = os.getcwd()
df_grouped_sub_ID = pd.read_csv('./sub_info/Final_sub_ID_grouped.csv')
df_flattened_sub_ID = pd.read_csv('./sub_info/sub_ID_class.csv')
path_input = 'input'

def load_input(path_parent, list_hypnos):
    data = []
    for sub_ID in tqdm(list_hypnos, desc='Load data ... '):
        temp_data = pd.read_csv(os.path.join(path_parent, sub_ID, 'probabilistic_hypnogram.csv'), index_col=0).transpose()
        temp_data = temp_data.values[:,:,np.newaxis]
        data.append(temp_data)
    data = np.array(data)
    return data
        
def load_label(df_flattened_sub_ID, list_hypnos):
    list_labels = sorted(set(df_flattened_sub_ID['class'].values))
    label_to_int = {k:v for v,k in enumerate(list_labels)}
    int_to_label = {v:k for k,v in label_to_int.items()}
    
    labels = []
    for sub_ID in tqdm(list_hypnos, desc='Get labels ... '):
        temp_label = df_flattened_sub_ID.loc[df_flattened_sub_ID['sub_ID']==sub_ID, 'class'].values[0]
        labels.append(label_to_int[temp_label])
    output = np.array(labels)
    # output = output[:,np.newaxis]
    return output

def zero_or_one_labeling(np_label):
    list_labels = sorted(set(df_flattened_sub_ID['class'].values))
    label_to_int = {k:v for v,k in enumerate(list_labels)}
    int_to_label = {v:k for k,v in label_to_int.items()}
    # {'Healthy':1, 'OSA':3, 'Insomnia':2, 'COMISA':0}

    con1 = np_label==label_to_int['Healthy'] # healthy is labeled as '1'
    con2 = np_label!=label_to_int['Healthy'] # non_healthy is labeled as {0:COMISA, 2:Insomnia, 3:OSA}

    np_label[con2] = 0 # non-healthy to 0
    return np_label 


## class definition ----------------------------------------------
class four_classes:
    def __init__(self):
        self.list_subID = os.listdir(path_input)
        
    def load_input(self):
        self.np_input_data = load_input(path_input, self.list_subID)
        return self.np_input_data

    def load_label(self):
        self.np_label = load_label(df_flattened_sub_ID, self.list_subID)
        
        return self.np_label
    

class binary_h_o:
    def __init__(self):
        self.list_subID_h = list(df_grouped_sub_ID['Healthy'].dropna().values)
        self.list_subID_o = list(df_grouped_sub_ID['OSA'].dropna().values)
        self.list_subID = self.list_subID_h + self.list_subID_o

    def load_input(self):
        self.np_input_data = load_input(path_input, self.list_subID)
        return self.np_input_data

    def load_label(self):
        self.np_label = zero_or_one_labeling(load_label(df_flattened_sub_ID, self.list_subID))
        return self.np_label

    
class binary_h_i:
    def __init__(self):
        self.list_subID_h = list(df_grouped_sub_ID['Healthy'].dropna().values)
        self.list_subID_i = list(df_grouped_sub_ID['Insomnia'].dropna().values)
        self.list_subID = self.list_subID_h + self.list_subID_i

    def load_input(self):
        self.np_input_data = load_input(path_input, self.list_subID)
        return self.np_input_data

    def load_label(self):
        self.np_label = zero_or_one_labeling(load_label(df_flattened_sub_ID, self.list_subID))
        return self.np_label


class binary_h_c:
    def __init__(self):
        self.list_subID_h = list(df_grouped_sub_ID['Healthy'].dropna().values)
        self.list_subID_c = list(df_grouped_sub_ID['COMISA'].dropna().values)
        self.list_subID = self.list_subID_h + self.list_subID_c

    def load_input(self):
        self.np_input_data = load_input(path_input, self.list_subID)
        return self.np_input_data

    def load_label(self):
        self.np_label = zero_or_one_labeling(load_label(df_flattened_sub_ID, self.list_subID))
        return self.np_label
