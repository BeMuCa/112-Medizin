# -*- coding: utf-8 -*-

# Test of Gradien Boosting algorithm
#   Plotten der Feature Gains
#   Plotten der Anzahl an Feature Nutzungs 
#   Ganz unten kann das Modell des GBoostings gespeichert werden
#
#

#import csv
#import scipy.io as sio

import matplotlib.pyplot as plt
import numpy as np

from wettbewerb import load_references
from features_112 import features
#import features_112 as features_112

import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split            # 
from sklearn import metrics                                     # for F1 score


########################### Load Trainings data ########################################################

ecg_leads,ecg_labels,fs,ecg_names = load_references()

########################### Array init  #################################################################
    
labels = np.array([], dtype=object)                    # Array für labels mit 1(A) und 0(N)
        
fail_label = np.array([], dtype=object)                # Array für labels mit ~ und O
    
Predictions_boost = np.array([], dtype=object)          # Array für Prediction


########################### Calculate the features ######################################################

features = features(ecg_leads,fs)          #   --> das will er nicht checken auch mit methoden import

########################### Delete labels with values != 0 or 1 and corresponding features  ###############

for nr,y in enumerate(ecg_labels):
    if ecg_labels[nr] == 'N':                   
        labels = np.append(labels,'0')  # normal = 0,N           
        continue                                            # ohne continue geht er aus unerklärlichen gründen immer ins else

    if ecg_labels[nr] == 'A':                               # ""
        labels = np.append(labels,'1')  # flimmern = 1,A
        continue

    else:
        fail_label= np.append(fail_label, nr)

    
########################### delete feature for the labels ~ and O    #########################################
    
features = np.delete(features, fail_label.astype(int), axis=0)

###########################  Trainings und Test Satz Split ###################################################
  
X_train_boost, X_test_boost, y_train_boost, y_test_boost = train_test_split(features, labels, test_size=0.4, random_state=7)

############################### Modell und Training 

dtrain = xgb.DMatrix(X_train_boost, label=y_train_boost) # features und labels in Dmatrix gepackt(nötig bei xgb)    # train
dtest = xgb.DMatrix(X_test_boost, label=y_test_boost) # features und labels in Dmatrix gepackt(nötig bei xgb)       # test

#### parameters:

evallist = [(dtrain, 'train'), (dtest, 'eval')]             # evaluieren des Trainings
num_round = 30                                              # ab 21,22,23 .. alle gleich
param = {'max_depth': 10, 'eta': 0.1111111111, 'objective': 'binary:hinge', 'gamma': 7.0}       # param für das Modell      (max depth 5)

######### eigentliche training:

bst = xgb.train( param, dtrain, num_round, evals=evallist, early_stopping_rounds = 4)        # xgb.train returns booster model

############################### FEATURE TESTING 
#featureScore_weight = bst.get_score( importance_type='weight')    #the number of times a feature is used to split the data across all trees. 
#featureScore_gain = bst.get_score( importance_type='gain')        #the average gain across all splits the feature is used in.
#
### plot the weight
#keys_weight = featureScore_weight.keys()
#values_weight = featureScore_weight.values()
### plot the gain
#keys_gain = featureScore_gain.keys()
#values_gain = featureScore_gain.values()
### choose 1
#
#fig, ax = plt.subplots(figsize=(10,10))
#ax.bar(keys_gain, values_gain, width=1, edgecolor="purple", linewidth=0.7)
##ax.set_title("GAIN per feature")
##ax.bar(keys_weight, values_weight, width=1, edgecolor="purple", linewidth=0.7)
##ax.set_title("Number of times used")
#plt.show()
#
##################################################################  Prediction

y_pred = bst.predict(dtest)             # [ 1.  0. ....] -> List voller Floats

y_prediction = [str(round(value)) for value in y_pred]             # ['1', '0', ..]

##################################################################  Performance berechnung 

print("-------------")
print("Accuracy: %.3f " % metrics.accuracy_score(y_test_boost, y_prediction))

print("F1:" , metrics.f1_score(y_test_boost, y_prediction, average='micro'))       # weil wir alles in float haben kein binary mgl

##################################################################  Save Trained Modell
#
#bst.save_model('GBoosting_model.json')
#
print("-----DONE-----")
