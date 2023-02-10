# -*- coding: utf-8 -*-

"""
Test of Gradien Boosting algorithm
   Plotten der Feature Gains
   Plotten der Anzahl an Feature Nutzungs 
   Das Modell des GBoostings kann gespeichert werden
   F1:0.95978(0.96)
"""


#import csv
#import scipy.io as sio
import graphviz

import matplotlib.pyplot as plt
import numpy as np

from wettbewerb import load_references
from features_112 import features
#import features_112 as features_112
from numpy import genfromtxt
from sklearn.model_selection import cross_val_score;
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_tree

from sklearn.model_selection import train_test_split            # 
from sklearn import metrics                                     # for F1 score


########################### Load Trainings data ########################################################

ecg_leads,ecg_labels,fs,ecg_names = load_references()

########################### Array init  #################################################################
    
labels = np.array([], dtype=object)                    # Array für labels mit 1(A) und 0(N)
        
fail_label = np.array([], dtype=object)                # Array für labels mit ~ und O
    
Predictions_boost = np.array([], dtype=object)          # Array für Prediction


########################### Calculate the features ######################################################

#features = features(ecg_leads,fs)          #   --> das will er nicht checken auch mit methoden import
#features = genfromtxt('learningfeatures_16_scaled.csv', delimiter=',')
features = genfromtxt('learningfeatures_ALLESINDHIER.csv', delimiter=',')
#features = features.reshape(-1,1)
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
  
X_train_boost, X_test_boost, y_train_boost, y_test_boost = train_test_split(features, labels, test_size=0.2, random_state=7)

############################### Modell und Training 

dtrain = xgb.DMatrix(X_train_boost, label=y_train_boost) # features und labels in Dmatrix gepackt(nötig bei xgb)    # train
dtest = xgb.DMatrix(X_test_boost, label=y_test_boost) # features und labels in Dmatrix gepackt(nötig bei xgb)       # test

#### parameters:

evallist = [(dtrain, 'train'), (dtest, 'eval')]             # evaluieren des Trainings
num_round = 55                                              # ab 21,22,23 .. alle gleich
param = {'max_depth': 16, 'eta': 0.3, 'objective': 'binary:hinge', 'gamma': 5.0, 'subsample':0.75,'lambda':5.0,'alpha':0.3}       # param für das Modell      (max depth 5)
# 16,0.11,7.0 = 0.9597 -- altt:10;0.1111;7.0 = 0.9597(0.96)

######### eigentliche training:

bst = xgb.train( param, dtrain, num_round, evals=evallist, early_stopping_rounds = 3)        # xgb.train returns booster model

############################### FEATURE TESTING 
## choose 1:

#featureScore_weight = bst.get_score( importance_type='weight')    #the number of times a feature is used to split the data across all trees. 
#featureScore_gain = bst.get_score( importance_type='gain')        #the average gain across all splits the feature is used in.
# IMPORTANCE TYPE: https://towardsdatascience.com/be-careful-when-interpreting-your-features-importance-in-xgboost-6e16132588e7

## plot the weight
#keys_weight = featureScore_weight.keys()
#values_weight = featureScore_weight.values()

## plot the gain
#keys_gain = featureScore_gain.keys()
#values_gain = featureScore_gain.values()

#a = []
#weight = list(values_weight)
#gain = list(keys_gain)

#namen = [ "rel_lowPass", "rel_highPass", "rel_bandPass", "max_ampl.", "sdnn", "peak_diff_median", "peaks_per_measure", "peaks_per_lowPass", "peak_diff_mean", "rmssd", "rmssd_neu", "sdnn_neu", "nn20", "nn50", "pNN20", "pNN50"]



#for nr,y in enumerate(gain):
#    if(y == "f0"):
#        a.append(namen[0])
#    if(y == "f1"):
#        a.append(namen[1])
#    if(y == "f2"):
#        a.append(namen[2])
#    if(y == "f3"):
#        a.append(namen[3])
#    if(y == "f4"):
#        a.append(namen[4])
#    if(y == "f5"):
#        a.append(namen[5])
#    if(y == "f6"):
#        a.append(namen[6])
#    if(y == "f7"):
#        a.append(namen[7])
#    if(y == "f8"):
#        a.append(namen[8])
#    if(y == "f9"):
#        a.append(namen[9])
#    if(y == "f10"):
#        a.append(namen[10])
#    if(y == "f11"):
#        a.append(namen[11])
#    if(y == "f12"):
#        a.append(namen[12])
#    if(y == "f13"):
#        a.append(namen[13])
#    if(y == "f14"):
#        a.append(namen[14])
#    if(y == "f15"):
#        a.append(namen[15])
    

#fig, (ax1) = plt.subplots(1,1 ,figsize=(10,10))
#ax1.bar(a, values_gain, width=1, edgecolor="purple", linewidth=0.7)
#plt.xticks(rotation=90)
#ax1.set_title("GAIN per feature")
#plt.subplots_adjust(left=0.10, bottom=0.20, right=0.85, top=0.85)
#fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(10,10))
#ax1.bar(keys_gain, values_gain, width=1, edgecolor="purple", linewidth=0.7)
#ax1.set_title("GAIN per feature")
#ax2.bar(keys_weight, values_weight, width=1, edgecolor="purple", linewidth=0.7)
#ax2.set_title("Number of times used")
#ax3.bar(keys_weight, a, width=1, edgecolor="purple", linewidth=0.7)
#ax3.set_title("Gain/Num")
#ax1.axhline(y=5, color='red', linestyle='--')
#ax1.axhline(y=10, color='red', linestyle='--')
#plt.show()
#plot_tree(bst)
#plt.show()

##################################################################  Prediction

y_pred = bst.predict(dtest)             # [ 1.  0. ....] -> List voller Floats

y_prediction = [str(round(value)) for value in y_pred]             # ['1', '0', ..]

##################################################################  Performance berechnung 
print("################")
#print(y_prediction)
print("######### XGB #######")

print("Accuracy: %.3f " % metrics.accuracy_score(y_test_boost, y_prediction))

print("F1:" , metrics.f1_score(y_test_boost, y_prediction, average='micro'))       # weil wir alles in float haben kein binary mgl
print("################")
##################################################################  Save Trained Modell

#print("Saving...")

bst.save_model('GB_ENSEMBLE_16.json')

#print("-----DONE-----")
