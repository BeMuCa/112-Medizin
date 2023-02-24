#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Script for testing the gradient boosting algorithm using the xgboost library and saving it into a json file if needed.
Also offers the function to plot the gain and weight of the features used in the algorithm.

"""
__author__ = "Berk Calabakan"



import matplotlib.pyplot as plt
import numpy as np
from wettbewerb import load_references
from features_112 import features
from numpy import genfromtxt
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn import metrics
import time

### Load Trainings data
ecg_leads,ecg_labels,fs,ecg_names = load_references()

### Array initiation
labels = np.array([], dtype=object)
fail_label = np.array([], dtype=object)
Predictions_boost = np.array([], dtype=object)

### Calculate the features
#features = features(ecg_leads,fs)          
#features = genfromtxt('learningfeatures_14features.csv', delimiter=',')
#features = genfromtxt('learningfeatures_5_wichtigsten.csv', delimiter=',')
#features = genfromtxt('learningfeatures_2_stärksten.csv', delimiter=',')
features = genfromtxt('learningfeatures_2_features.csv', delimiter=',')

### Change labels to 1 and 0 
### Delete labels with values != 0 or 1 and corresponding features
for nr,y in enumerate(ecg_labels):
    if ecg_labels[nr] == 'N':
        labels = np.append(labels,'0')
        continue

    if ecg_labels[nr] == 'A':
        labels = np.append(labels,'1')
        continue

    else:
        fail_label= np.append(fail_label, nr)


### Delete features for the labels ~ and O
features = np.delete(features, fail_label.astype(int), axis=0)

### Training and Test split
X_train_boost, X_test_boost, y_train_boost, y_test_boost = train_test_split(features, labels, test_size=0.3, random_state=7)

### Model und Training 
dtrain = xgb.DMatrix(X_train_boost, label=y_train_boost) 
dtest = xgb.DMatrix(X_test_boost, label=y_test_boost)

#### parameters:
evallist = [(dtrain, 'train'), (dtest, 'eval')]
num_round = 55
param = {'max_depth': 16, 'eta': 0.3, 'objective': 'binary:hinge', 'gamma': 5.0, 'subsample':0.75,'lambda':5.0,'alpha':0.3}  # param of the model      

# start the timer 
start_time = time.time()

### training:
bst = xgb.train( param, dtrain, num_round, evals=evallist, early_stopping_rounds = 3)        # xgb.train returns booster model



# Plotting the features weights and gains (comment out when timing the algorithm) 
# Information about the importance types: https://towardsdatascience.com/be-careful-when-interpreting-your-features-importance-in-xgboost-6e16132588e7

#featureScore_weight = bst.get_score( importance_type='weight')    #the number of times a feature is used to split the data across all trees. 
#featureScore_gain = bst.get_score( importance_type='gain')        #the average gain across all splits the feature is used in.
## IMPORTANCE TYPE: https://towardsdatascience.com/be-careful-when-interpreting-your-features-importance-in-xgboost-6e16132588e7
#
### plot the weight
#keys_weight = featureScore_weight.keys()
#values_weight = featureScore_weight.values()
#
### plot the gain
#keys_gain = featureScore_gain.keys()
#values_gain = featureScore_gain.values()
#
#a = []
#namen = [ "rel_lowPass", "rel_highPass", "rel_bandPass", "max_ampl.", "peak_diff_median", "peaks_per_measure", "peaks_per_lowPass", "peak_diff_mean", "rmssd_neu", "sdnn_neu", "nn20", "nn50", "pNN20", "pNN50"]

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


#fig, (ax1) = plt.subplots(1,1 ,figsize=(10,10))
#ax1.bar(a, values_gain, width=1, edgecolor="purple", linewidth=0.7)
#plt.xticks(rotation=90)
#ax1.set_title("GAIN per feature")
#plt.subplots_adjust(left=0.10, bottom=0.20, right=0.85, top=0.85)
#fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(10,10))
#
#ax1.bar(keys_gain, values_gain, width=1, edgecolor="purple", linewidth=0.7)
#ax1.set_title("GAIN per feature")#
#ax2.bar(keys_weight, values_weight, width=1, edgecolor="purple", linewidth=0.7)
#ax2.set_title("Number of times used")#
#ax1.axhline(y=5, color='red', linestyle='--')
#ax1.axhline(y=10, color='red', linestyle='--')
#plt.show()


### Prediction
y_pred = bst.predict(dtest)
y_prediction = [str(round(value)) for value in y_pred]

# End time of the timer
end_time = time.time()

### Performance berechnung 
print("######### XGB Performance #######")

print("Accuracy: %.3f " % metrics.accuracy_score(y_test_boost, y_prediction))

print("F1:" , metrics.f1_score(y_test_boost, y_prediction, average='micro'))

print("Runtime: {:.6f} seconds".format(end_time - start_time))
print("################")

### Save trained Model
print("Saving...")
#bst.save_model('GB_final_2weakest.json')
print("-----DONE-----")
