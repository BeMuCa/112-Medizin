#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Script for testing the random forest algorithm using the sklearn library and saving it into a pickle file if needed.

"""
__author__ = "Berk Calabakan"



import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.utils import compute_class_weight

from wettbewerb import load_references
from features_112 import features
from numpy import genfromtxt
import time


### Load Trainings data
ecg_leads,ecg_labels,fs,ecg_names = load_references()

### Array initiation
labels = np.array([])
fail_label = np.array([])


### Calculate the features
#features = features(ecg_leads,fs);                 
#features = genfromtxt('learningfeatures_14features.csv', delimiter=',')
#features = genfromtxt('learningfeatures_5_wichtigsten.csv', delimiter=',')
features = genfromtxt('learningfeatures_2_features.csv', delimiter=',')
#features = genfromtxt('learningfeatures_2_stärksten.csv', delimiter=',')

### Change labels to 1 and 0 
### Delete labels with values != 0 or 1 and corresponding features
for nr,y in enumerate(ecg_labels):

    if ecg_labels[nr] == 'N':
        labels = np.append(labels,0)
        continue

    if ecg_labels[nr] == 'A':
        labels = np.append(labels,1)
        continue

    else:
        fail_label= np.append(fail_label, nr)

### Delete features for the labels ~ and O
features = np.delete(features, fail_label.astype(int), axis=0)

### Training and test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=7)

### Model and Training 
#class_weights = compute_class_weight('balanced',), class_weight='balanced'

# start the timer 
start_time = time.time()

model = RandomForestClassifier(n_estimators= 100, max_features=14, criterion = "entropy")
model.fit(X_train,y_train)

### Prediction
Predictions = model.predict(X_test)         
end_time = time.time()

### Performance berechnung      

# nutzbar mit 'N' -> 0 (statt '0'); 
#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)        # Teil in 10 gruppen,            
#
#n_f1 = cross_val_score(model, X_test, y_test, scoring='f1', cv=cv, n_jobs=-1, error_score='raise')       # f1 fürs scoring
#
#n_accuracy = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')       # für uns 


### Performance berechnung 
print("######### Random Forrest Performance #######")

print("Accuracy: %.3f " % metrics.accuracy_score(y_test, Predictions))
print("F1:" , metrics.f1_score(y_test, Predictions, average='micro'))
print("Runtime: {:.6f} seconds".format(end_time - start_time))
print('#####################')

scores = cross_val_score(model, features, labels, cv = 10)
print(scores)


### Save model
print("Saving...")
#filename = "RF_final_2_weakmodel.pickle"
#pickle.dump(model, open(filename, "wb"))
print("----done------")