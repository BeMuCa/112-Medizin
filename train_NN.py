import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os
from scipy.fft import fft, fftfreq
from wettbewerb import load_references
import math
import features_112
from sklearn.svm import LinearSVC;
from sklearn.svm import SVR;
from sklearn.svm import SVC;
from sklearn.svm import LinearSVR;
from sklearn.model_selection import train_test_split;
from sklearn.model_selection import cross_val_predict;
from sklearn import metrics
from numpy import genfromtxt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.neural_network import MLPClassifier
from statistics import mean

ecg_leads,ecg_labels,fs,ecg_names = load_references();

########################### Array init  #################################################################
    
labels = np.array([], dtype=object)                    # Array für labels mit 1(A) und 0(N)
        
fail_label = np.array([], dtype=object)                # Array für labels mit ~ und O
    
Predictions = np.array([], dtype=object)          # Array für Prediction


########################### Calculate the features ######################################################

#features = features_112.features_kalman(ecg_leads,fs)
#np.savetxt("learningfeatures_kalman.csv", features, delimiter=",")
### loading calculated features
#features = genfromtxt('learningfeatures_ALLESINDHIER.csv', delimiter=',')
features = genfromtxt('learningfeatures_14features.csv', delimiter=',')
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
#features_kalman = np.delete(features_kalman, fail_label.astype(int), axis=0)

###################################################################  Trainings und Test Satz Split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=7)

##################################################################  Modell und Training 

#clf = MLPClassifier(solver='lbfgs', alpha=1, hidden_layer_sizes=(5,4,2), random_state=1) #alpha=1e-5
F1_score = np.array([]) 
model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(solver='lbfgs', max_iter = 1000)) # 50 fittet am besten, eventuell overfittung? -> senken hidden_layer_sizes=(5,4,2), random_state=1)
    ])
model.fit(X_train,y_train)

Predictions = model.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
metrics.f1_score(y_test, Predictions, average='micro')

print("################")

print("Accuracy: %.3f " % metrics.accuracy_score(y_test, Predictions))
print("F1:" , metrics.f1_score(y_test, Predictions, average='micro'))

print('#####################')
print("Saving...")
filename = "NN_final_default_param.pickle"
pickle.dump(model, open(filename, "wb"))
#
print("----done------")
print('#####################')

#clf.fit(X_train,y_train)
#clf_k.fit(X_train_k,y_train_k)

#Predictions = np.array([], dtype=object)
#Predictions = clf.predict(X_test)   

print("################")
print('labels:')
#print(y_test)
print('predicitons:')
#print(Predictions)
print("################")
print("Saving...")


######################### save model


#print("################")
#print('labels:')
#print(y_test)
#print('predicitons:')
#print(Predictions)
#print(Predictions_k)
#print("################")

#print("Accuracy: %.3f " % metrics.accuracy_score(y_test, Predictions))
#print("F1:" , metrics.f1_score(y_test, Predictions, average='micro'))
#print("Accuracy: %.3f " % metrics.accuracy_score(y_test_k, Predictions_k))
#print("F1:" , metrics.f1_score(y_test_k, Predictions_k, average='micro'))
#print('#####################')
#print('####################alpha=1e-5')
#clf = Pipeline([
#    ("scaler", StandardScaler()),
#    ("mlp", MLPClassifier(solver='lbfgs', max_iter = 100000, alpha=1e-5, hidden_layer_sizes=(5,4,2), random_state=1)) # 50 fittet am besten, eventuell overfittung? -> senken
#    ])
#scores = cross_val_score(clf, features, labels, cv = 10)
#print(scores)
#print(mean(scores))
#print('####################alpha=1')
#clf = Pipeline([
#    ("scaler", StandardScaler()),
#    ("mlp", MLPClassifier(solver='lbfgs', max_iter = 100000, alpha=1, hidden_layer_sizes=(5,4,2), random_state=1)) # 50 fittet am besten, eventuell overfittung? -> senken
#    ])
#scores = cross_val_score(clf, features, labels, cv = 10)
#print(scores)
#print(mean(scores))
#print('####################alpha=20')
#clf = Pipeline([
#    ("scaler", StandardScaler()),
#    ("mlp", MLPClassifier(solver='lbfgs', max_iter = 100000, alpha=20, hidden_layer_sizes=(5,4,2), random_state=1)) # 50 fittet am besten, eventuell overfittung? -> senken
#    ])
#scores = cross_val_score(clf, features, labels, cv = 10)
#print(scores)
#print(mean(scores))
#print('####################alpha=2000')
#clf = Pipeline([
#    ("scaler", StandardScaler()),
#    ("mlp", MLPClassifier(solver='lbfgs', max_iter = 100000, alpha=2000, hidden_layer_sizes=(5,4,2), random_state=1)) # 50 fittet am besten, eventuell overfittung? -> senken
#    ])
#scores = cross_val_score(clf, features, labels, cv = 10)
#print(scores)
#print(mean(scores))
#print('####################alpha=5000')
#clf = Pipeline([
#    ("scaler", StandardScaler()),
#    ("mlp", MLPClassifier(solver='lbfgs', max_iter = 100000, alpha=2000, hidden_layer_sizes=(5,4,2), random_state=1)) # 50 fittet am besten, eventuell overfittung? -> senken
#    ])
#scores = cross_val_score(clf, features, labels, cv = 10)
#print(scores)
#print(mean(scores))
#print('####################alpha=50000')
#clf = Pipeline([
#    ("scaler", StandardScaler()),
#    ("mlp", MLPClassifier(solver='lbfgs', max_iter = 100000, alpha=2000, hidden_layer_sizes=(5,4,2), random_state=1)) # 50 fittet am besten, eventuell overfittung? -> senken
#    ])
#scores = cross_val_score(clf, features, labels, cv = 10)
#print(scores)
#print(mean(scores))