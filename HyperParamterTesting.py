# -*- coding: utf-8 -*-
# HYPER PARAMETER TESTER:
# Ziel ist es alle Hyperparameter durchzugehen und zu gucken welches am besten ist 

import matplotlib.pyplot as plt

#import pandas as pd

# evaluate random forest algorithm for classification
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import metrics                                     # for F1 score

from wettbewerb import load_references
from features_112 import features

###################### Base Structure ##################

ecg_leads,ecg_labels,fs,ecg_names = load_references() 

######## ARRAY INIT ########

labels = np.array([], dtype=object)                    # Array für labels mit 1(A) und 0(N)
        
fail_label = np.array([], dtype=object)                # Array für labels mit ~ und O
    
Predictions_boost = np.array([], dtype=object)          # Array für Prediction

######## Features #########

features = features(ecg_leads,fs) 

######## Label ############

for nr,y in enumerate(ecg_labels):
    if ecg_labels[nr] == 'N':                   
        labels = np.append(labels,'0')            # '0' für xgb         
        continue

    if ecg_labels[nr] == 'A':
        labels = np.append(labels,'1')            # '1' für xgb
        continue

    else:
        fail_label= np.append(fail_label, nr)


features = np.delete(features, fail_label.astype(int), axis=0)

########  Trainings und Test Split ###########
  
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=1)    # random state 1


#################################################################################################################################
#1 XGBoosting
#2 RandomForrest
#3 (SVM)


####################### XGBoosting #############################

def HyperTest_XGB():
    ##### setup #####
    dtrain = xgb.DMatrix(X_train, label=y_train)     # train
    dtest = xgb.DMatrix(X_test, label=y_test)        # test

    ## Hyperparameter Setup ##
    # num_round                                      # Anzahl Boosting iterationen : useless if we use early stoppage
    p1_range = np.linspace(3,11, num = 6)            # max depth     : Maximale tiefe der Bäume
    p2_range = np.linspace(0,1, num = 10)            # eta           : learning rate     
    p3_range = np.linspace(0,10, num = 11)           # gamma         : Minimum loss reduction required to make a further partition
    
    
    evallist = [(dtrain, 'train'), (dtest, 'eval')]            
    
   
    p1_res = np.array([])
    p2_res = np.array([])
    p3_res = np.array([])
    p4_res = np.array([])                            # best Iteration

    F1 = np.array([])
    acc = np.array([])

    for p1 in p1_range:
        for p2 in p2_range:
            for p3 in p3_range:
                param = {'max_depth': int(p1), 'eta': p2, 'objective': 'binary:hinge', 'gamma': p3}       # param für das Modell      (max depth 5)


                bst = xgb.train( param, dtrain, 20, evals=evallist, early_stopping_rounds = 4) 
                #https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.train

                #bst_best_iteration = bst.best_iteration()    

                y_pred = bst.predict(dtest)             
                y_prediction = [str(round(value)) for value in y_pred]

                p1_res = np.append(p1_res, p1)
                p2_res = np.append(p2_res, p2)
                p3_res = np.append(p3_res, p3)
                #p4_res = np.append(p4_res, bst_best_iteration)

                F1 = np.append(F1, metrics.f1_score(y_test, y_prediction, average='micro'))
                acc = np.append(acc, metrics.accuracy_score(y_test, y_prediction))

    optimal_indice = np.argmax(F1)
    F1_opt = F1[optimal_indice]
    p1_opt = p1_res[optimal_indice]
    p2_opt = p2_res[optimal_indice]
    p3_opt = p3_res[optimal_indice]
    #p4_opt = p4_res[optimal_indice]
    acc_opt = acc[np.argmax(acc)]

    print("P1_optimal   (max_depth):",p1_opt)
    print("P2_optimal   (eta):",p2_opt)
    print("P3_optimal   (gamma):",p3_opt)
    #print("P4_optimal   (iteration):",p4_opt)
    
    print("###### Optimal Training results: ")    
    print("Max Accuracy:",acc_opt)
    print("Max F1:",F1_opt)

#################################################################################################################  





###################### RANDOM FORREST ###################
def HyperTest_RF():
    
    ##### setup #####

    p1_range = np.arange(10,200,10)                   # n_estimators  : Anzahl Bäume
    p2_range = np.arange(1,10,1)            # max features  : Anzahl maximal zu nutzender Feats     
    p3_range = np.arange(0,2,1)                       # criterion     : 3 optionen für split Berechnung

    p1_res = np.array([])
    p2_res = np.array([])
    p3_res = np.array([])


    F1 = np.array([])
    acc = np.array([])

    for p1 in p1_range:
        for p2 in p2_range:
            for p3 in p3_range:
                if p3==0:
                    crit="gini"                 # gini schneller
                if p3==1:
                    crit="entropy"              # entropy complexer
                if p3==2:
                    crit="log_loss"
                print("p3 :", p3)
                print("crit :", crit)
                # max_depth: of trees; -- max features: None = alle
                model = RandomForestClassifier(n_estimators= p1, max_features=p2, criterion = crit)
                
                print(y_train)
                
                model.fit(X_train,y_train)

                y_pred = model.predict(X_test)
                print(y_pred)                
                print("test1")
                #y_prediction = [str(value) for value in y_pred]

                print("################################################################################################")
      
                p1_res = np.append(p1_res, p1)
                p2_res = np.append(p2_res, p2)
                p3_res = np.append(p3_res, p3)
                #p4_res = np.append(p4_res, bst_best_iteration)

                F1 = np.append(F1, metrics.f1_score(y_test, y_pred, average='micro'))
                acc = np.append(acc, metrics.accuracy_score(y_test, y_pred))

    optimal_indice = np.argmax(F1)
    F1_opt = F1[optimal_indice]
    p1_opt = p1_res[optimal_indice]
    p2_opt = p2_res[optimal_indice]
    p3_opt = p3_res[optimal_indice]
    #p4_opt = p4_res[optimal_indice]
    acc_opt = acc[np.argmax(acc)]

    print("P1_optimal   (n_estimators):",p1_opt)
    print("P2_optimal   (max features):",p2_opt)
    print("P3_optimal   (criterion):",p3_opt)
    #print("P4_optimal   (iteration):",p4_opt)
    
    print("###### Optimal Training results: ")    
    print("Max Accuracy:",acc_opt)
    print("Max F1:",F1_opt)




###################################################



HyperTest_RF()