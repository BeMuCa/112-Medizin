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
from scipy.fft import fft

############# nur für tester:

from wettbewerb import load_references
import pyhrv
import pyhrv.time_domain as td
import math

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
    
    

########################### Calculate and load feature out of the data ######################################################## 

    features = features_112.features(ecg_leads,fs)

########################### Use features and trained model to predict ########################################################
    
    ##################           RF             #########################
    
    if(model_name == 'RF_Model.pickle'):
        loaded_model = pickle.load(open(model_name, "rb"))            # load model

        Predictions_array = loaded_model.predict(features)          # predict
    
    
    

    ##################          XGB             #########################
    if(model_name == 'GBoosting_model.json'):
        bst = xgb.Booster()
        bst.load_model(fname = model_name)              ## load model

        dtest = xgb.DMatrix(features)                   ## DMatrix format is needed
        
        Predictions_array = bst.predict(dtest)                     ## predict based on the features




############################       Change from 0,1 to N,A    ############################################################

    labels = np.array([], dtype=object)
    
    for nr,y in enumerate(Predictions_array):                           ## We will probably need A,N instead of 0,1
        if Predictions_array[nr] == 0.:                   
            labels = np.append(labels,'N')  # normal = 0,N           

        if Predictions_array[nr] == 1.:
            labels = np.append(labels,'A')  # flimmern = 1,A

########################### Form into the Output Prediction form ########################################################    

    predictions = []

    for nr,y in enumerate(labels):           # ecg_names = List ; Pred = Array
        
        predictions.append((ecg_names[nr],y))
    
#------------------------------------------------------------------------------    
    return predictions # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
                               
                               
        
ecg_leads,ecg_labels,fs,ecg_names = load_references()


#
##features = features_112.features(ecg_leads,fs,1)
#try:
#    for idx, ecg_lead in enumerate(ecg_leads):
#
#        if len(ecg_lead)<9000:
#            print("alte größe", ecg_lead.size)
#            teiler = 9000//len(ecg_lead)                       # 2999 ecgs zb -> Teiler= 3(weil gerundet) -> ecgs werden mit 2 weiteren ecgs erweitert
#            for i in range (0,teiler):                         # -> 2999*3= 8997; 
#              ecg_lead = np.append(ecg_lead, ecg_lead)
#            print("Neue größe", ecg_lead.size, " der Teiler war:", teiler, "der index:", idx)
#        #
#        if len(ecg_lead)>9000:
#            print("alte größe", ecg_lead.size, "der index:", idx)
#            index = []
#            index.extend(range(9000,ecg_lead.size))
#
#            ecg_lead= np.delete(ecg_lead, index)
#        #
#            print("Neue größe", ecg_lead.size)        
#        else:
#            print(ecg_lead.size, "der else fall ---------------")

sdnn = np.array([])
N = 9000  
for idx, ecg_lead in enumerate(ecg_leads):
    if idx ==3:
        print("------")
        r_peaks = [0,1,2]
        yf = fft(ecg_lead)                                    # Berechnung des komplexen Spektrums.
        r_yf = 2.0/N * np.abs(yf[0:N//2])                     # Umwandlung in ein reelles Spektrum.
        print('Die Transforation in den Frequenzbereich schlägt fehl!')
        normier_faktor = (np.sum(r_yf))                     # Inverses Integral über Frequenzbereich  
                                                            # Gesamt integ, weil unten direkt der gesamte freq. bereich normiert wird
        yf_lowPass = np.array([])                            # Tiefpassfilter von Frequenz (0-450)*fd, dass entspricht (0-15)Hz.
        result_NN50 = td.nn50([1,1])
        #nn50 = np.append(nn50, )
        print(result_NN50['pnn50'])