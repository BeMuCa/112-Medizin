#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Script for testing the SVM algorithm using the sklearn library and saving it into a pickle file if needed.

"""


import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from wettbewerb import load_references
from sklearn.svm import LinearSVC;
from sklearn.svm import SVR;
from sklearn.svm import SVC;
from sklearn.svm import LinearSVR;
from sklearn.model_selection import train_test_split;
from sklearn.model_selection import cross_val_predict;
from sklearn import metrics;  
from numpy import genfromtxt;
from sklearn.pipeline import Pipeline;
from sklearn.preprocessing import StandardScaler;
from sklearn.model_selection import cross_val_score;
import pickle;

### Load Trainings data
ecg_leads,ecg_labels,fs,ecg_names = load_references()

### Array initiation   
labels = np.array([], dtype=object) 
fail_label = np.array([], dtype=object)
Predictions = np.array([], dtype=object)

### Calculate the features
#features = features_112.features(ecg_leads,fs)

### loading calculated features
features = genfromtxt('learningfeatures.csv', delimiter=',')


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

### Training and test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=7)

### Model and Training 
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = "poly", degree=3,C=50)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
# model = SVR(kernel = "poly", degree=2,C=100,epsilon=0.1);
# model = LinearSVR(epsilon=1.5);
model.fit(X_train,y_train)

### Prediction
Predictions = np.array([], dtype=object)
Predictions = model.predict(X_test)         
                                        

print("################")

print("Accuracy: %.3f " % metrics.accuracy_score(y_test, Predictions))
print("F1:" , metrics.f1_score(y_test, Predictions, average='micro'))

print('#####################')

#print('Accuracy: %.3f (%.3f)' % (np.mean(n_accuracy), np.std(n_accuracy)))                # Mittelwert und Standartdeviation

#print('Der F1 score: \n')

#print(n_f1)

print("Saving...")


########################## save model
filename = "SVM_Modelle.pickle"
#
pickle.dump(model, open(filename, "wb"))
#
print("----done------")
print('#####################')
print('Crossvalidation:')
model_cross = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = "poly", degree=3,C=50)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
scores = cross_val_score(model_cross, features, labels, cv = 10)
print(scores)