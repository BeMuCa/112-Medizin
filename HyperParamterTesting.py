#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Script for Hyperparameter testing.

"""
__author__ = "Berk Calabakan"

import matplotlib.pyplot as plt


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from wettbewerb import load_references
from features_112 import features
from numpy import genfromtxt;

###################### Base Structure ##################

### Load Trainings data
ecg_leads,ecg_labels,fs,ecg_names = load_references() 

### Array initiation
labels = np.array([], dtype=object)
fail_label = np.array([], dtype=object)
Predictions_boost = np.array([], dtype=object)

### Calculate the features
#features = features(ecg_leads,fs)
features = genfromtxt('learningfeatures_16_scaled.csv', delimiter=',')

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
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=7)


#################################################################################################################################
#1 XGBoosting
#2 RandomForrest
#3 kNN


####################### XGBoosting #############################

def HyperTest_XGB():
    ##### setup #####
    dtrain = xgb.DMatrix(X_train, label=y_train)     # train
    dtest = xgb.DMatrix(X_test, label=y_test)        # test

    ## Hyperparameter Setup ##
    # num_round                                      # Anzahl Boosting iterationen : useless if we use early stoppage
    p1_range = np.array(range(1,20))                 # max depth     : Maximale tiefe der Bäume
    p2_range = np.linspace(0,1, num = 10)            # eta           : learning rate     
    p3_range = np.linspace(0,10, num = 11)           # gamma         : Minimum loss reduction required to make a further partition (just fyi)
    
    
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


####################### XGBoosting - focus OVERFITTING #############################
def HyperTest_XGB_testing():
    ##### setup #####
    dtrain = xgb.DMatrix(X_train, label=y_train)     # train
    dtest = xgb.DMatrix(X_test, label=y_test)        # test

    ## Hyperparameter Setup ##
    # num_round                                      # Anzahl Boosting iterationen : useless if we use early stoppage
    p1_range = np.array(range(16,17))                     # max depth     : Maximale tiefe der Bäume (größer besser)
    p2_range = np.linspace(0.1,0.3, num = 5)              # eta           : learning rate  (lower= langsamer = weniger overfitten)   
    p3_range = np.linspace(0,10, num = 3)               # gamma         : (je höher, je mehr regularisiert)(bei kleinem p1 gamma hoch) Minimum loss reduction required to make a further partition (just fyi)
    p4_range = np.linspace(10,100, num = 3)               # number of boosting rounds = anzahl bäume (zu hoch=overfitting)
    p5_range = np.linspace(0.5,1, num = 3)               # subsample (verhindert overfitting ) jeder iteration wird andere random feature genommen zum lernen
    p6_range = np.linspace(1,5, num = 2)               # Lambda (L2 regularisierungswert, je höher je stärker regularisiert) default =1
    p7_range = np.linspace(0,1, num = 2)                 # alpha (L1 reg. ) default 0
    evallist = [(dtrain, 'train'), (dtest, 'eval')]            
    
   
    p1_res = np.array([])
    p2_res = np.array([])
    p3_res = np.array([])
    p4_res = np.array([])                            # best Iteration
    p5_res = np.array([])
    p6_res = np.array([])
    p7_res = np.array([])
    F1 = np.array([])
    acc = np.array([])

    for p1 in p1_range:
        for p2 in p2_range:
            for p3 in p3_range:
                for p4 in p4_range:
                    for p5 in p5_range:
                        for p6 in p6_range:
                            for p7 in p7_range:
                                param = {'max_depth': int(p1), 'eta': p2, 'objective': 'binary:hinge', 'gamma': p3, 'subsample':p5, 'lambda':p6, 'alpha':p7}       # param für das Modell      (max depth 5)


                                bst = xgb.train( param, dtrain, num_boost_round = int(p4)) 
                                #https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.train

                                #bst_best_iteration = bst.best_iteration()    

                                y_pred = bst.predict(dtest)             
                                y_prediction = [str(round(value)) for value in y_pred]

                                p1_res = np.append(p1_res, p1)
                                p2_res = np.append(p2_res, p2)
                                p3_res = np.append(p3_res, p3)
                                p4_res = np.append(p4_res, p4)
                                p5_res = np.append(p5_res, p5)
                                p6_res = np.append(p6_res, p6)
                                p7_res = np.append(p7_res, p7)
                                
                                print("max_depth:",p1,"eta:",p2,"gamma:",p3,"num boosting rounds:",p4)
                                print("subsample size:", p5, " lambda:",p6,"alpha:", p7, "F1 score:",metrics.f1_score(y_test, y_prediction, average='micro') )
                                print(" ")
                                F1 = np.append(F1, metrics.f1_score(y_test, y_prediction, average='micro'))
                                acc = np.append(acc, metrics.accuracy_score(y_test, y_prediction))

    optimal_indice = np.argmax(F1)
    F1_opt = F1[optimal_indice]
    p1_opt = p1_res[optimal_indice]
    p2_opt = p2_res[optimal_indice]
    p3_opt = p3_res[optimal_indice]
    p4_opt = p4_res[optimal_indice]
    acc_opt = acc[np.argmax(acc)]

    print("P1_optimal   (max_depth):",p1_opt)
    print("P2_optimal   (eta):",p2_opt)
    print("P3_optimal   (gamma):",p3_opt)
    print("P4_optimal   (num rounds):",p4_opt)
    
    print("###### Optimal Training results: ")    
    print("Max Accuracy:",acc_opt)
    print("Max F1:",F1_opt)

#################################################################################################################  


###################### RANDOM FORREST ###################
def HyperTest_RF():
    
    ##### setup #####

    p1_range = np.arange(10,200,10)                   # n_estimators  : Anzahl Bäume
    p2_range = np.arange(1,16,1)            # max features  : Anzahl maximal zu nutzender Feats     
    p3_range = np.arange(0,3,1)                       # criterion     : 3 optionen für split Berechnung

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


###################### RANDOM FORREST - focus OVERFITTING ###################

def HyperTest_RF_testing():
    
    ##### setup #####

    p1_range = np.arange(200,300,10)                   # n_estimators  : Anzahl Bäume
    p2_range = np.arange(1,8,1)            # max features  : Anzahl maximal zu nutzender Feats     
    #p3_range = np.arange(0,3,1)                       # criterion     : 3 optionen für split Berechnung

    p1_res = np.array([])
    p2_res = np.array([])
    #p3_res = np.array([])

    F1 = np.array([0])
    acc = np.array([])

    for p1 in p1_range:
        for p2 in p2_range:
            # max_depth: of trees; -- max features: None = alle
            model = RandomForestClassifier(n_estimators= p1, max_features=p2, criterion = "entropy")
                      
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)

            p1_res = np.append(p1_res, p1)
            p2_res = np.append(p2_res, p2)
            
            #if (np.argmax(F1) < metrics.f1_score(y_test, y_pred, average='micro')): #falls f! score besser 
            print("Anzahl Bäume:",p1,"max features:",p2,"F1 score:",metrics.f1_score(y_test, y_pred, average='micro') )

            #p4_res = np.append(p4_res, bst_best_iteration)
            F1 = np.append(F1, metrics.f1_score(y_test, y_pred, average='micro'))
            acc = np.append(acc, metrics.accuracy_score(y_test, y_pred))
    print("################################################################################################")
    optimal_indice = np.argmax(F1)
    F1_opt = F1[optimal_indice]
    p1_opt = p1_res[optimal_indice]
    p2_opt = p2_res[optimal_indice]

    acc_opt = acc[np.argmax(acc)]

    print("P1_optimal   (n_estimators):",p1_opt)
    print("P2_optimal   (max features):",p2_opt)

    
    print("###### Optimal Training results: ")    
    print("Max Accuracy:",acc_opt)
    print("Max F1:",F1_opt)

###################################################


###################### k - nearest neighbours ###################
def HyperTest_kNN():
    """_summary_
    """
    ##### setup #####

    p1_range = np.arange(1,20,1)                   # n_neighbors  : Anzahl benachbarter Pkte ( der Faktor k bei kNN) 
    p2_range = np.array([1,2])            # max features  : Anzahl maximal zu nutzender Feats     
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
                    crit='uniform'                 # 
                if p3==1:
                    crit='distance'             # 

                print("p3 :", p3)
                print("crit :", crit)
                # max_depth: of trees; -- max features: None = alle
                model = KNeighborsClassifier(n_neighbors = p1, p=p2, weights = crit)
                
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

    print("P1_optimal   (n_neighbours):",p1_opt)
    print("P2_optimal   (p):",p2_opt)
    print("P3_optimal   (criterion):",p3_opt)
    #print("P4_optimal   (iteration):",p4_opt)
    
    print("###### Optimal Training results: ")    
    print("Max Accuracy:",acc_opt)
    print("Max F1:",F1_opt)


###################################################



#HyperTest_RF()

#HyperTest_XGB()

#HyperTest_kNN()

#HyperTest_RF_testing()

HyperTest_XGB_testing()