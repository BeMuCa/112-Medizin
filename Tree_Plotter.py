#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Script to test the decision tree algorithm and to plot a decision tree.

"""
__author__ = "Berk Calabakan"



import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics                                     # for F1 score

from wettbewerb import load_references
from features_112 import features


### Load Trainings data
ecg_leads,ecg_labels,fs,ecg_names = load_references()

### Array initiation
labels = np.array([])
fail_label = np.array([])


### Calculate the features
features = features(ecg_leads,fs);                 


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


### Trainings und test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=7)

### Modell und training 
model = tree.DecisionTreeClassifier(max_features=5, criterion = "entropy") 
model.fit(X_train,y_train)

### Prediction

Predictions = model.predict(X_test)         

### Crossvalidation

# nutzbar mit 'N' -> 0 (statt '0'); 
#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)        # Teil in 10 gruppen,            
#
#n_f1 = cross_val_score(model, X_test, y_test, scoring='f1', cv=cv, n_jobs=-1, error_score='raise')       # f1 fürs scoring
#
#n_accuracy = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')       # für uns 


### Performance 
print("######## Decision Tree Performance ########")

print("Accuracy: %.3f " % metrics.accuracy_score(y_test, Predictions))
print("F1:" , metrics.f1_score(y_test, Predictions, average='micro'))

print('#####################')


### Plotting:
tree.plot_tree(model,filled = True)
print(tree.plot_tree(model,filled = True))     #feature_names = True,class_names=True
plt.show()
