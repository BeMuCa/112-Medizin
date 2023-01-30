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
from numpy import genfromtxt;

##
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RepeatedStratifiedKFold
##


ecg_leads,ecg_labels,fs,ecg_names = load_references()

labels = np.array([], dtype=object)                    # Array für labels mit 1(A) und 0(N)
        
fail_label = np.array([], dtype=object)                # Array für labels mit ~ und O
    

########################### Calculate the features ######################################################

#features = features_112.features(ecg_leads,fs,2)

features = genfromtxt('learningfeatures_16.csv', delimiter=',')
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

## RF

RF = pickle.load(open('RF_Model.pickle', "rb"))            # load model
prediction_RF = RF.predict(X_test_boost)

## xgb
bst = xgb.Booster()
bst.load_model(fname = 'GBoosting_model.json')              ## load model
dtest = xgb.DMatrix(X_test_boost) 
prediction_xgb = bst.predict(dtest) 

## kNN

kNN = pickle.load(open('kNN_model.pickle', "rb"))            # load model
prediction_kNN = kNN.predict(X_test_boost)

## SVM

SVM = pickle.load(open('SVM_model.pickle', "rb"))            # load model
prediction_SVM = SVM.predict(X_test_boost)

############### Ensemble:                                                               ## HIER ENTSCHEIDEN OB SVM ODER KNN REINKOMMT
"""Wir kriegen in der Prediction_xx eine Liste voll mit den """
prediction_SVM = prediction_SVM.astype(float)
prediction_Ensemble = np.array([])
for nr,y in enumerate(prediction_RF):
    if (prediction_xgb[nr] + y + prediction_kNN[nr] + prediction_SVM[nr]) == 2 or (prediction_xgb[nr] + y + prediction_kNN[nr] + prediction_SVM[nr]) == 3 or (prediction_xgb[nr] + y + prediction_kNN[nr] + prediction_SVM[nr]) == 4:
        prediction_Ensemble = np.append(prediction_Ensemble,1)
    else:
        prediction_Ensemble = np.append(prediction_Ensemble,0)

######################################################################## UMWANDELN DER PREDICTS ZU 'A' und 'N'
prediction_Ensemble_end = np.array([], dtype=object)
prediction_xgb_end = np.array([], dtype=object)
prediction_kNN_end = np.array([], dtype=object)
prediction_RF_end = np.array([], dtype=object)
prediction_SVM_end = np.array([], dtype=object)


for nr,y in enumerate(prediction_Ensemble):                           
    if prediction_Ensemble[nr] == 0. :                   
        prediction_Ensemble_end = np.append(prediction_Ensemble_end,'N')  # normal = 0,N           
    if prediction_Ensemble[nr] == 1. :
        prediction_Ensemble_end = np.append(prediction_Ensemble_end,'A')  # flimmern = 1,A
        
for nr,y in enumerate(prediction_xgb):                           
    if prediction_xgb[nr] == 0. :                   
        prediction_xgb_end = np.append(prediction_xgb_end,'N')  # normal = 0,N           
    if prediction_xgb[nr] == 1. :
        prediction_xgb_end = np.append(prediction_xgb_end,'A')  # flimmern = 1,A
        
for nr,y in enumerate(prediction_kNN):                           
    if prediction_kNN[nr] == 0. :                   
        prediction_kNN_end = np.append(prediction_kNN_end,'N')  # normal = 0,N           
    if prediction_kNN[nr] == 1. :
        prediction_kNN_end = np.append(prediction_kNN_end,'A')  # flimmern = 1,A
        
for nr,y in enumerate(prediction_RF):                           
    if prediction_RF[nr] == 0. :                   
        prediction_RF_end = np.append(prediction_RF_end,'N')  # normal = 0,N           
    if prediction_RF[nr] == 1. :
        prediction_RF_end = np.append(prediction_RF_end,'A')  # flimmern = 1,A
        
for nr,y in enumerate(prediction_SVM):                           
    if prediction_SVM[nr] == 0. :                   
        prediction_SVM_end = np.append(prediction_SVM_end,'N')  # normal = 0,N           
    if prediction_SVM[nr] == 1. :
        prediction_SVM_end = np.append(prediction_SVM_end,'A')  # flimmern = 1,A
        

##################################################################  Performance berechnung 
print(" ")

## xgb
print("xgb:")
print("Accuracy: %.3f " % metrics.accuracy_score(y_test_boost, prediction_xgb_end))
print("F1:" , metrics.f1_score(y_test_boost, prediction_xgb_end, average='micro'))
print(" ")
## RF
print("RF:")
print("Accuracy: %.3f " % metrics.accuracy_score(y_test_boost, prediction_RF_end))
print("F1:" , metrics.f1_score(y_test_boost, prediction_RF_end, average='micro'))
print(" ")
## kNN
print("kNN:")
print("Accuracy: %.3f " % metrics.accuracy_score(y_test_boost, prediction_kNN_end))
print("F1:" , metrics.f1_score(y_test_boost, prediction_kNN_end, average='micro'))
print(" ")
## SVM
print("SVM:")
print("Accuracy: %.3f " % metrics.accuracy_score(y_test_boost, prediction_SVM_end))
print("F1:" , metrics.f1_score(y_test_boost, prediction_SVM_end, average='micro'))
print(" ")

## Ensemble
print("-------------------------")
print(" ")
print("ENSEMBLE:")
print("Accuracy: %.3f " % metrics.accuracy_score(y_test_boost, prediction_Ensemble_end))
print("F1:" , metrics.f1_score(y_test_boost, prediction_Ensemble_end, average='micro'))
print(" ")
print("-------------------------")
print(" ")
##### 

#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)        # Teil in 10 gruppen,            
#
#n_f1 = cross_val_score(model, X_test, y_test, scoring='f1', cv=cv, n_jobs=-1, error_score='raise')       # f1 fürs scoring
#
#n_accuracy = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')       # für u

#print('Accuracy: %.3f (%.3f)' % (np.mean(n_accuracy), np.std(n_accuracy)))                # Mittelwert und Standartdeviation
#print('Der F1 score: \n')
#print(n_f1)
