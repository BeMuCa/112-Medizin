# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""
import pickle
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os
from typing import List, Tuple
from train import RandomForrest_112
import features_112 as features_112
import xgboost as xgb
from xgboost import Booster

###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : str='RF_Model.pickle',is_binary_classifier : bool=True) -> List[Tuple[str,str]]:
    '''
    Parameters
    ----------
    model_name : str
        Dateiname des Models. In Code-Pfad
    ecg_leads : list of numpy-Arrays
        EKG-Signale.
    fs : float
        Sampling-Frequenz der Signale.
    ecg_names : list of str
        eindeutige Bezeichnung für jedes EKG-Signal.
    model_name : str
        Name des Models, kann verwendet werden um korrektes Model aus Ordner zu laden
    is_binary_classifier : bool
        Falls getrennte Modelle für F1 und Multi-Score trainiert werden, wird hier übergeben, 
        welches benutzt werden soll
    Returns
    -------
    predictions : list of tuples
        ecg_name und eure Diagnose
    '''

#------------------------------------------------------------------------------
# Euer Code ab hier  
    
    Predictions_array = np.array([])    

########################### Calculate and load feature out of the data ######################################################## 

    features = features_112.features(ecg_leads,fs)

########################### Use features and trained model to predict ########################################################

    ##################           ENSEMBLE             #########################

    if(model_name == 'Ensemble'):
        ## RF
        RF = pickle.load(open('RF_Model.pickle', "rb"))
        prediction_RF = RF.predict(features)

        ## xgb
        GB = xgb.Booster()
        GB.load_model(fname = 'GBoosting_model.json')
        dfeat = xgb.DMatrix(features) 
        prediction_xgb = GB.predict(dfeat) 

        ## kNN
        kNN = pickle.load(open('kNN_model.pickle', "rb"))            # load model
        prediction_kNN = kNN.predict(features)

        ## SVM
        SVM = pickle.load(open('SVM_Model.pickle', "rb"))
        prediction_SVM = SVM.predict(features)

        ## Ensemble calculation
        for nr,y in enumerate(prediction_RF):
            if (prediction_xgb[nr] + y + prediction_kNN[nr]) == 2 or (prediction_xgb[nr] + y + prediction_kNN[nr]) == 3:
                Predictions_array = np.append(Predictions_array,1)
            else:
                Predictions_array = np.append(Predictions_array,0)

    ##################           RF             #########################
    
    if(model_name == 'RF_Model.pickle'):
        loaded_model = pickle.load(open(model_name, "rb"))
        Predictions_array = loaded_model.predict(features)

    ##################          XGB             #########################
    if(model_name == 'GBoosting_model.json'):
        bst = xgb.Booster()
        bst.load_model(fname = model_name)
        dtest = xgb.DMatrix(features)
        Predictions_array = bst.predict(dtest)

    ##################           kNN             #########################
    
    if(model_name == 'kNN_Model.pickle'):
        kNN = pickle.load(open(model_name, "rb"))
        Predictions_array = kNN.predict(features)
        
    ##################           SVM             #########################
    
    if(model_name == 'SVM_Model.pickle'):
        SVM = pickle.load(open(model_name, "rb"))
        Predictions_array = SVM.predict(features)


############################       Change from 0,1 to N,A    ############################################################

    labels = np.array([], dtype=object)
    
    for nr,y in enumerate(Predictions_array):                           ## We will probably need A,N instead of 0,1
        if y == 0.:                   
            labels = np.append(labels,'N')  # normal = 0,N           

        if y == 1.:
            labels = np.append(labels,'A')  # flimmern = 1,A

########################### Form into the Output Prediction form ########################################################    

    predictions = []

    for nr,y in enumerate(labels):           # ecg_names = List ; Pred = Array
        
        predictions.append((ecg_names[nr],y))
    
#------------------------------------------------------------------------------    
    return predictions # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
                               
                               
        
