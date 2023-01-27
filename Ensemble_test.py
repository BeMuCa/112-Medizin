# -*- coding: utf-8 -*-
"""
Skript zum kreieren eines Ensembles
"""
from wettbewerb import load_references
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
import pickle
import features_112 as features_112
import xgboost as xgb
import numpy as np
from sklearn import metrics

ecg_leads,ecg_labels,fs,ecg_names = load_references()

labels = np.array([], dtype=object)                    # Array für labels mit 1(A) und 0(N)
        
fail_label = np.array([], dtype=object)                # Array für labels mit ~ und O
    

########################### Calculate the features ######################################################

features = features_112.features(ecg_leads,fs,1)

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

######################## Load models    
#
#   Braucht das Modell die selben Features beim Predicten oder ist es okay wenn wir nur 2 Features geben statt unseren 16
#
#

## RF

RF = pickle.load(open('RF_Model_ensemble.pickle', "rb"))            # load model
prediction_RF = RF.predict(X_test_boost)

## xgb
bst = xgb.Booster()
bst.load_model(fname = 'GBoosting_model_ensemble.json')              ## load model
dtest = xgb.DMatrix(X_test_boost) 
prediction_xgb = bst.predict(dtest) 

## kNN

kNN = pickle.load(open('kNN_model_ensemble.pickle', "rb"))            # load model
prediction_kNN = kNN.predict(X_test_boost)


############### Ensemble:
"""Wir kriegen in der Prediction_xx eine Liste voll mit den """
prediction_Ensemble = np.array([])
for nr,y in enumerate(prediction_RF):
    if (prediction_xgb[nr] + y + prediction_kNN[nr]) == 2 or (prediction_xgb[nr] + y + prediction_kNN[nr]) == 3:
        prediction_Ensemble = np.append(prediction_Ensemble,1)
    else:
        prediction_Ensemble = np.append(prediction_Ensemble,0)
    
    ######################################################################## UMWANDELN DER PREDICTS ZU 'A' und 'N'
    labels = np.array([], dtype=object)
    
    for nr,y in enumerate(prediction_Ensemble):                           
        if prediction_Ensemble[nr] == 0. :                   
            labels = np.append(labels,'N')  # normal = 0,N           

        if prediction_Ensemble[nr] == 1. :
            labels = np.append(labels,'A')  # flimmern = 1,A
            
    print("DAS SIND DIE LABELS MIT A; N EINGESETZT:", labels)
    
##################################################################  Performance berechnung 

print("das sind die y_test_boost größen", y_test_boost)

print("ENSEMBLE:")
print("Accuracy: %.3f " % metrics.accuracy_score(y_test_boost, labels))
print("F1:" , metrics.f1_score(y_test_boost, labels, average='micro'))
## xgb
print("xgb:")
print("Accuracy: %.3f " % metrics.accuracy_score(y_test_boost, prediction_xgb))
print("F1:" , metrics.f1_score(y_test_boost, prediction_xgb, average='micro'))
## RF
print("RF:")
print("Accuracy: %.3f " % metrics.accuracy_score(y_test_boost, prediction_RF))
print("F1:" , metrics.f1_score(y_test_boost, prediction_RF, average='micro'))