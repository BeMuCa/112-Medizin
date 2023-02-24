#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Script for testing the submission of our models.

"""
__author__ = "Berk Calabakan"

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from wettbewerb import load_references
from features_112 import features
from numpy import genfromtxt;

### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors

import xgboost as xgb
from xgboost import Booster
import pickle
from typing import List, Tuple
from train import RandomForrest_112
import features_112 as features_112
from sklearn import metrics

### Load Trainings data
ecg_leads,ecg_labels,fs,ecg_names = load_references()

### Array initiation
labels = np.array([], dtype=object)      
fail_label = np.array([], dtype=object)
    

### Calculate the features
features = genfromtxt('learningfeatures_14features.csv', delimiter=',')
#features = features(ecg_leads,fs)
print("FEATURES DONE")

### Change labels to 1 and 0 
### Delete labels with values != 0 or 1 and corresponding features
for nr,y in enumerate(ecg_labels):
    if ecg_labels[nr] == 'N':                   
        labels = np.append(labels,'N')
        continue

    if ecg_labels[nr] == 'A':
        labels = np.append(labels,'A')
        continue

    else:
        fail_label= np.append(fail_label, nr)

### delete feature for the labels ~ and O     
features = np.delete(features, fail_label.astype(int), axis=0)

### Test and training split
X_train_boost, X_test_boost, y_train_boost, y_test_boost = train_test_split(features, labels, test_size=0.4, random_state=7)

#### test method
def test(model_name : str='GBoosting_model.json'):
    '''
    Parameters
    ----------
    model_name : str
        Dateiname des Models. In Code-Pfad
    -------
    '''

    ### RF  
    if(model_name == 'RF_Model.pickle'):
        print("----- Testing RF ... ------")
        loaded_model = pickle.load(open(model_name, "rb"))

        Predictions_array = loaded_model.predict(features)
    

    ### XGB 
    if(model_name == 'GB_model.json'):
        print("----- Testing XGB ... ------")
        bst = xgb.Booster()
        bst.load_model(fname = model_name)

        dtest = xgb.DMatrix(features)
        
        Predictions_array = bst.predict(dtest)
        
    ### SVM
    if(model_name == 'SVM_model.pickle'):
        print("----- Testing RF ... ------")
        loaded_model = pickle.load(open(model_name, "rb"))

        Predictions_array = loaded_model.predict(features)
        Predictions_array = Predictions_array.astype(float)


    ### NN
    if(model_name == 'NN_model.pickle'):
        print("----- Testing RF ... ------")
        loaded_model = pickle.load(open(model_name, "rb"))

        Predictions_array = loaded_model.predict(features)

    ### transforming the predicts to 'A' and 'N'
    pred = np.array([], dtype=object)
    
    for nr,y in enumerate(Predictions_array):                           
        if y == 0. or y =='0':                   
            pred = np.append(pred,'N')  # normal = 0,N           

        if y == 1. or y =='1':
            pred = np.append(pred,'A')  # flimmern = 1,A

    print("DAS SIND DIE PREDICITONS", pred)
    print("DAS SIND DIE WAHREN WERTE", labels)


### Performance calculation
    print("-------------")
    print("Accuracy: %.3f " % metrics.accuracy_score(labels, pred))

    print("F1:" , metrics.f1_score(labels, pred, average='micro'))
    print("-------------")
    
#test()
test('RF_Model.pickle')