# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""

import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os
from typing import List, Tuple
from train import RandomForrest_112

###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : str='model.npy',is_binary_classifier : bool=True) -> List[Tuple[str,str]]:
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
    with open(model_name, 'rb') as f:  
        th_opt = np.load(f)         # Lade simples Model (1 Parameter)

    detectors = Detectors(fs)        # Initialisierung des QRS-Detektors

    predictions = list()
    
    for idx,ecg_lead in enumerate(ecg_leads):
        r_peaks = detectors.hamilton_detector(ecg_lead)     # Detektion der QRS-Komplexe
        sdnn = np.std(np.diff(r_peaks)/fs*1000) 
        if sdnn < th_opt:
            predictions.append((ecg_names[idx], 'N'))
        else:
            predictions.append((ecg_names[idx], 'A'))
        if ((idx+1) % 100)==0:
            print(str(idx+1) + "\t Dateien wurden verarbeitet.")
            
            
#------------------------------------------------------------------------------    
    return predictions # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
                               
                               
        
