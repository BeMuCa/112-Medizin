# Test der Decision Trees

#import csv
#import scipy.io as sio

import matplotlib.pyplot as plt

#import pandas as pd

# evaluate random forest algorithm for classification
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


ecg_leads,ecg_labels,fs,ecg_names = load_references()

labels = np.array([], dtype=object)                    # Array für labels mit 1(A) und 0(N)
        
fail_label = np.array([], dtype=object)                # Array für labels mit ~ und O
    

########################### Calculate the features ######################################################
features = genfromtxt('learningfeatures.csv', delimiter=',')
#features = features(ecg_leads,fs)
print("FEATURES DONE")
########################### Delete labels with values != 0 or 1 and corresponding features  ###############

for nr,y in enumerate(ecg_labels):
    if ecg_labels[nr] == 'N':                   
        labels = np.append(labels,'N')  # normal = 0,N           
        continue                                            # ohne continue geht er aus unerklärlichen gründen immer ins else

    if ecg_labels[nr] == 'A':                               # ""
        labels = np.append(labels,'A')  # flimmern = 1,A
        continue

    else:
        fail_label= np.append(fail_label, nr)

    
########################### delete feature for the labels ~ and O    #########################################
    
features = np.delete(features, fail_label.astype(int), axis=0)


########################### Test and training Split    #########################################

X_train_boost, X_test_boost, y_train_boost, y_test_boost = train_test_split(features, labels, test_size=0.4, random_state=7)


####
def test(model_name : str='GBoosting_model.json'):

    ##################           RF             #########################    
    if(model_name == 'RF_Model.pickle'):
        print("----- Testing RF ... ------")
        loaded_model = pickle.load(open(model_name, "rb"))            # load model

        Predictions_array = loaded_model.predict(features)          # predict
    

    ##################          XGB             #########################
    if(model_name == 'GBoosting_model.json'):
        print("----- Testing XGB ... ------")
        bst = xgb.Booster()
        bst.load_model(fname = model_name)              ## load model

        dtest = xgb.DMatrix(features)                   ## DMatrix format is needed -- bei abgabe hier features rein
        
        Predictions_array = bst.predict(dtest)                     ## predict based on the features
        
 

    ######################################################################## UMWANDELN DER PREDICTS ZU 'A' und 'N'
    pred = np.array([], dtype=object)
    
    for nr,y in enumerate(Predictions_array):                           
        if y == 0. :                   
            pred = np.append(pred,'N')  # normal = 0,N           

        if y == 1. :
            pred = np.append(pred,'A')  # flimmern = 1,A
            
    
       ######################################################################
    
    print("DAS SIND DIE PREDICITONS", pred)
    print("DAS SIND DIE WAHREN WERTE", labels)

##################################################################  Performance berechnung 

    print("-------------")
    print("Accuracy: %.3f " % metrics.accuracy_score(labels, pred))

    print("F1:" , metrics.f1_score(labels, pred, average='micro'))

    
test()
#test('RF_Model.pickle')