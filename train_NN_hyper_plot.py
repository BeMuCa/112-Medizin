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
features = genfromtxt('learningfeatures_16.csv', delimiter=',')
features_kalman = genfromtxt('learningfeatures_16.csv', delimiter=',')


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
features_kalman = np.delete(features_kalman, fail_label.astype(int), axis=0)

###################################################################  Trainings und Test Satz Split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=7)
X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(features_kalman, labels, test_size=0.4, random_state=7)

##################################################################  Modell und Training 

#clf = MLPClassifier(solver='lbfgs', alpha=1, hidden_layer_sizes=(5,4,2), random_state=1) #alpha=1e-5
print("Asuwertung beginnt")
F1_score = np.array([]) 

print("Asuwertung 1 abgeschlossen")
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(solver='lbfgs', max_iter = 1000, alpha=0.0001, hidden_layer_sizes=(5,4,2), random_state=1)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
clf.fit(X_train,y_train)
Predictions = np.array([], dtype=object)
Predictions = clf.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print(metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 2 abgeschlossen")
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(solver='lbfgs', max_iter = 1000, alpha=0.1, hidden_layer_sizes=(5,4,2), random_state=1)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
clf.fit(X_train,y_train)
Predictions = np.array([], dtype=object)
Predictions = clf.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 3 abgeschlossen")
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(solver='lbfgs', max_iter = 1000, alpha=1, hidden_layer_sizes=(5,4,2), random_state=1)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
clf.fit(X_train,y_train)
Predictions = np.array([], dtype=object)
Predictions = clf.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 4 abgeschlossen")
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(solver='lbfgs', max_iter = 1000, alpha=10, hidden_layer_sizes=(5,4,2), random_state=1)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
clf.fit(X_train,y_train)
Predictions = np.array([], dtype=object)
Predictions = clf.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 5 abgeschlossen")
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(solver='lbfgs', max_iter = 1000, alpha=50, hidden_layer_sizes=(5,4,2), random_state=1)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
clf.fit(X_train,y_train)
Predictions = np.array([], dtype=object)
Predictions = clf.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(solver='lbfgs', max_iter = 1000, alpha=100, hidden_layer_sizes=(5,4,2), random_state=1)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
clf.fit(X_train,y_train)
Predictions = np.array([], dtype=object)
Predictions = clf.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 6 abgeschlossen")
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(solver='lbfgs', max_iter = 1000, alpha=1000, hidden_layer_sizes=(5,4,2), random_state=1)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
clf.fit(X_train,y_train)
Predictions = np.array([], dtype=object)
Predictions = clf.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 7 abgeschlossen")
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(solver='lbfgs', max_iter = 1000, alpha=2000, hidden_layer_sizes=(5,4,2), random_state=1)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
clf.fit(X_train,y_train)
Predictions = np.array([], dtype=object)
Predictions = clf.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 8 abgeschlossen")
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(solver='lbfgs', max_iter = 1000, alpha=5000, hidden_layer_sizes=(5,4,2), random_state=1)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
clf.fit(X_train,y_train)
Predictions = np.array([], dtype=object)
Predictions = clf.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 9 abgeschlossen")
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(solver='lbfgs', max_iter = 1000, alpha=10000, hidden_layer_sizes=(5,4,2), random_state=1)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
clf.fit(X_train,y_train)
Predictions = np.array([], dtype=object)
Predictions = clf.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 10 abgeschlossen")
print("Asuwertung 11 abgeschlossen")
print("plot wird erstellt")
points = np.array([])
F1 = np.array([])
F1 = F1_score
points = [0.001,0.1,1,10,50,100,1000,2000,5000,10000]
plt.plot(points, F1)
plt.xlabel("alpha")
plt.ylabel("F1 score")
plt.title("Performance der Hyperparameter")
plt.show()
#clf.fit(X_train,y_train)
#clf_k.fit(X_train_k,y_train_k)

#Predictions = np.array([], dtype=object)
#Predictions = clf.predict(X_test)   

#Predictions_k = np.array([], dtype=object)
#Predictions_k = clf.predict(X_test_k)

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
