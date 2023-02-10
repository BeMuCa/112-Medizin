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
from sklearn import metrics;  
from numpy import genfromtxt;
from sklearn.pipeline import Pipeline;
from sklearn.preprocessing import StandardScaler;
from sklearn.model_selection import cross_val_score;
import pickle;


ecg_leads,ecg_labels,fs,ecg_names = load_references();

########################### Array init  #################################################################
    
labels = np.array([], dtype=object)                    # Array für labels mit 1(A) und 0(N)
        
fail_label = np.array([], dtype=object)                # Array für labels mit ~ und O
    
Predictions = np.array([], dtype=object)          # Array für Prediction


########################### Calculate the features ######################################################

#features = features_112.features(ecg_leads,fs)

### loading calculated features
features = genfromtxt('learningfeatures_16.csv', delimiter=',')


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

###################################################################  Trainings und Test Satz Split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=7)

##################################################################  Modell und Training 
#kernel = "poly", degree=2,C=100,epsilon=0.1);
# model = LinearSVR(epsilon=1.5);
print("Asuwertung beginnt")
F1_score = np.array([]) 

print("Asuwertung 1 abgeschlossen")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = "poly", degree=3,C=1)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
model.fit(X_train,y_train);
Predictions = np.array([], dtype=object)
Predictions = model.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print(metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 2 abgeschlossen")
filename = "SVM_model_final.pickle"
##
pickle.dump(model, open(filename, "wb"))
print("saved")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = "poly", degree=3,C=0.1)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=1)
model.fit(X_train,y_train);
Predictions = np.array([], dtype=object)
Predictions = model.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 2 abgeschlossen")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = "poly", degree=3,C=1)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=2)
model.fit(X_train,y_train);
Predictions = np.array([], dtype=object)
Predictions = model.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 2 abgeschlossen")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = "poly", degree=3,C=10)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=3)
model.fit(X_train,y_train);
Predictions = np.array([], dtype=object)
Predictions = model.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 2 abgeschlossen")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = "poly", degree=3,C=50)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=4)
model.fit(X_train,y_train);
Predictions = np.array([], dtype=object)
Predictions = model.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 2 abgeschlossen")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = "poly", degree=3,C=100)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=5)
model.fit(X_train,y_train);
Predictions = np.array([], dtype=object)
Predictions = model.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 2 abgeschlossen")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = "poly", degree=3,C=1000)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=6)
model.fit(X_train,y_train);
Predictions = np.array([], dtype=object)
Predictions = model.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 2 abgeschlossen")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = "poly", degree=3,C=2000)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=7)
model.fit(X_train,y_train);
Predictions = np.array([], dtype=object)
Predictions = model.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 2 abgeschlossen")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = "poly", degree=3,C=5000)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=8)
model.fit(X_train,y_train);
Predictions = np.array([], dtype=object)
Predictions = model.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 2 abgeschlossen")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = "poly", degree=3,C=10000)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=9)
model.fit(X_train,y_train);
Predictions = np.array([], dtype=object)
Predictions = model.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 2 abgeschlossen")

print("plot wird erstellt")
points = np.array([])
F1 = np.array([])
F1 = F1_score
points = [0.001,0.1,1,10,50,100,1000,2000,5000,10000]
plt.plot(points, F1)
plt.xlabel("C")
plt.ylabel("F1_score")
plt.title("Performance der Hyperparameter")
plt.show()





#################################################################  Prediction
print("Asuwertung beginnt")
F1_score = np.array([]) 

print("Asuwertung 1 abgeschlossen")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = "poly", degree=3,coef0=0.001)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
model.fit(X_train,y_train);
Predictions = np.array([], dtype=object)
Predictions = model.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))

print("Asuwertung 2 abgeschlossen")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = "poly", degree=3,coef0=0.1)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=1)
model.fit(X_train,y_train);
Predictions = np.array([], dtype=object)
Predictions = model.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 2 abgeschlossen")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = "poly", degree=3,coef0=1)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=2)
model.fit(X_train,y_train);
Predictions = np.array([], dtype=object)
Predictions = model.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 2 abgeschlossen")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = "poly", degree=3,coef0=10)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=3)
model.fit(X_train,y_train);
Predictions = np.array([], dtype=object)
Predictions = model.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 2 abgeschlossen")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = "poly", degree=3,coef0=50)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=4)
model.fit(X_train,y_train);
Predictions = np.array([], dtype=object)
Predictions = model.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 2 abgeschlossen")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = "poly", degree=3,coef0=100)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=5)
model.fit(X_train,y_train);
Predictions = np.array([], dtype=object)
Predictions = model.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 2 abgeschlossen")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = "poly", degree=3,coef0=1000)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=6)
model.fit(X_train,y_train);
Predictions = np.array([], dtype=object)
Predictions = model.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 2 abgeschlossen")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = "poly", degree=3,coef0=2000)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=7)
model.fit(X_train,y_train);
Predictions = np.array([], dtype=object)
Predictions = model.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 2 abgeschlossen")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = "poly", degree=3,coef0=5000)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=8)
model.fit(X_train,y_train);
Predictions = np.array([], dtype=object)
Predictions = model.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 2 abgeschlossen")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel = "poly", degree=3,coef0=10000)) # 50 fittet am besten, eventuell overfittung? -> senken
    ])
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=9)
model.fit(X_train,y_train);
Predictions = np.array([], dtype=object)
Predictions = model.predict(X_test)
F1_score = np.append(F1_score , metrics.f1_score(y_test, Predictions, average='micro'))
print("Asuwertung 2 abgeschlossen")

print("plot wird erstellt")
points = np.array([])
F1 = np.array([])
F1 = F1_score
points = [0.001,0.1,1,10,50,100,1000,2000,5000,10000]
plt.plot(points, F1)
plt.xlabel("coef0")
plt.ylabel("F1_score")
plt.title("Performance der Hyperparameter")
plt.show()
#Predictions = np.array([], dtype=object)
#Predictions = model.predict(X_test)         
#np.savetxt("predictions.csv", Predictions, delimiter=",")
#np.savetxt("y_test.csv", y_test, delimiter=",")
#Predictions = Predictions.astype(int)
#Predictions = Predictions.astype(string)
#Predictions = np.rint(Predictions)
# Predictions = Predictions.astype(int)
# print(y_test.dtype)
# print(Predictions.dtype)
# for i in range(0,Predictions.size):
#     if Predictions[i] >= 1 or Predictions[i] <= -1:
#         Predictions[i] = '1';
#     else:
#         Predictions[i] = '0';
# np.savetxt("predictions.csv", Predictions, delimiter=",")
# Predictions = Predictions.astype(object)
# for i in range(0,Predictions.size):
#     if Predictions[i] >= 1 or Predictions[i] <= -1:
#         Predictions[i] = '1';
#     else:
#         Predictions[i] = '0';
# Predictions = Predictions.astype(object)
# Printen für uns                                                    
#print("################")
#print('labels:')
#print(y_test)
#print('predicitons:')
#print(Predictions)
#print("################")

#print("Accuracy: %.3f " % metrics.accuracy_score(y_test, Predictions))
#print("F1:" , metrics.f1_score(y_test, Predictions, average='micro'))

#print('#####################')

#print('Accuracy: %.3f (%.3f)' % (np.mean(n_accuracy), np.std(n_accuracy)))                # Mittelwert und Standartdeviation

#print('Der F1 score: \n')

#print(n_f1)

#print("Saving...")


########################### save model
#filename = "SVM_model.pickle"
##
#pickle.dump(model, open(filename, "wb"))
##
#print("----done------")
#print('#####################')
#print('Crossvalidation:')
#model_cross = Pipeline([
#    ("scaler", StandardScaler()),
#    ("svc", SVC(kernel = "poly", degree=3,C=50)) # 50 fittet am besten, eventuell overfittung? -> senken
#    ])
#scores = cross_val_score(model_cross, features, labels, cv = 10)
#print(scores)