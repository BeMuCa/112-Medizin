# -*- coding: utf-8 -*-
"""
Beispiel Code und  Spielwiese

"""

import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os
from scipy.fft import fft, fftfreq
from wettbewerb import load_references
import math
import pyhrv.time_domain as td



### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

#ecg_leads,ecg_labels,fs,ecg_names = load_references()     # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name (meist fs=300 Hz)


def features(ecg_leads,fs, set = 2):
    '''
    set = 1: only the most gain bringing features 
    set = 2: all features
    '''


    detectors = Detectors(fs)                                 # Initialisierung des QRS-Detektors

    # Label-List
    labels = np.array([])                                     # Initialisierung Array für die Labels.


    # Feature-List
    # alt
    sdnn_normal = np.array([])                                # Initialisierung normal ("N") SDNN.
    # alt
    sdnn_afib = np.array([])                                  # Initialisierung afib ("A") SDNN.
    # neu
    sdnn = np.array([])                                       # Initialisierung des SDNN Wertes
    peak_diff_mean = np.array([])                             # Initialisierung Mittelwert des R-Spitzen Abstand.
    peak_diff_median = np.array([])                           # Initialisierung Median des R-Spitzen Abstand.
    peaks_per_measure = np.array([])                          # Initialisierung Anzahl der R-Spitzen.
    peaks_per_lowPass = np.array([])                          # Initialisierung R-Spitzen im Nierderfrequenzbereich.
    max_amplitude = np.array([])                              # Initialisierung Maximaler Auschlage des Spannungspegels. 
    relativ_lowPass = np.array([])                            # Initialisierung Relativer Anteil des Niederfrequenzbandes an dem Gesamtspektrum.
    relativ_highPass = np.array([])                           # Initialisierung Relativer Anteil des Mittelfrequenzbandes an dem Gesamtspektrum.
    relativ_bandPass = np.array([])                           # Initialisierung Relativer Anteil des Hochfrequenzbandes an dem Gesamtspektrum.
    rmssd = np.array([])                                      # Initialisierung des RMSSD Wertes
    
  ## pyhrv:  
    rmssd_neu = np.array([])                                  # Initialisierung des RMSSD Wertes (pyhrv - Version)
    sdnn_neu = np.array([])                                   # Initialisierung des SDNN Wertes (pyhrv - Version)
    nn50 = np.array([])                                       # Initialisierung des NN50 Wertes (pyhrv - Version)
    nn20 = np.array([])                                       # Initialisierung des NN20 Wertes (pyhrv - Version)
    pNN50 = np.array([])                                      # Initialisierung des pNN50 Wertes (pyhrv - Version)
    pNN20 = np.array([])                                      # Initialisierung des pNN20 Wertes (pyhrv - Version)
    
    ### FFT Initialisierung
    N = 9000                                                  # Anzahl der Messungen (9000 in 30s, für jede Messung gleich, daher nur einemal berechnet).
    fs = 300                                                  # Gegebene Abtastfrequenz des Messung.
    T = 1.0/300.0                                             # Kalibrierung auf Sampel-Frequenz/Abtastungsrate (300Hz).
    fd = fs/N                                                 # Frequenzauflösung des Spektrumes der Messung. !Nyquistkriterium: Es können höchstens bis 150Hz aussagekräftige Informationen gewonnen werden!
    t = np.linspace(0.0, N*T, N, endpoint=False)             # Initialisierung des Zeitbereiches (für jede Messung gleich, daher nur einemal berechnet).
    xf = fftfreq(N, T)[:N//2]                                # Initialisierung des Frequenzbereiches (für jede Messung gleich, daher nur einemal berechnet).

    ## Erstelleung eines Feature enums
    feature_list = [sdnn_afib, sdnn, peak_diff_mean, peak_diff_median, peaks_per_measure, peaks_per_lowPass, max_amplitude, relativ_lowPass, relativ_highPass, relativ_bandPass, rmssd, rmssd_neu, sdnn_neu, nn50, nn20, pNN50, pNN20]
    ### Wenn Testlauf, dann können in range(102,6000) Messungen gelöscht werden, welche dann nicht mehr verarbietet werden.
    #ecg_leads = np.delete(ecg_leads, range(102,6000))


    ### Datenverarbeitung für jede Messung. Die Ergebnisse werden in den Arrays der Feature-List gespeichert.
    
    for idx, ecg_lead in enumerate(ecg_leads):
      try:
        ### Anzahl der ecg_leads anpassen:
        #y = ecg_lead                                          # Laden des Messung
        ## mit '0'
        #if len(y)<4499 :                                      # Bei weniger Messungen (<9000) werden "0" an den Array gehängt.
        #    d = 4500-len(y)                                   # müsste hier nicht <9000 hin?
        #    for i in range(0,d):
        #        y = np.append(y, 0)
        ## Werte verdoppelt
        try:
          if len(ecg_lead)<9000:
            teiler = 9000//len(ecg_lead)                       # 2999 ecgs zb -> Teiler= 3(weil alle kommastellen gecuttet) -> ecgs werden mit 2 weiteren ecgs erweitert
          for i in range (0,teiler):                         # -> 2999*3= 8997; 
            ecg_lead = np.append(ecg_lead, ecg_lead)         # potentiell erweitern sodass wir safe auf 9000 kommen (aber unnötig)
          if len(ecg_lead)<9000:
            for i in range(len(ecg_lead),9000):
              ecg_lead = np.append(ecg_lead, 0)
        except:
          print('Das Auffüllen der Daten auf 9000 Messpunkte schlägt fehl!')
        ## Wenn mehr Werte als 9000
        try:               
          if len(ecg_lead)>9000:

            index = []
            index.extend(range(9000,ecg_lead.size))

            ecg_lead= np.delete(ecg_lead, index)              # alle Werte Über 9000 gecutet
        except:
          print('Das Abschneiden der Messung auf 9000 Messpunkte schlägt fehl!')
        ### Zeitbereich
        try:
          r_peaks = detectors.swt_detector(ecg_lead)            # Detektion der QRS-Komplexe.(SWT>PT>HAMI>CHRIS)  
        except:
          print('Erkennung der QRS-Komplexe schlägt fehl!')
        try:
          if len(r_peaks)<3:                                    # Wenn zu wenige peaks detektiert
            r_peaks = [0,1,2]       # einfach gemacht 

          #ecg_lead=ecg_lead[1500::1]                          # Wir skippen die ersten 5 Sekunden ( weil manchmal am anfang das ecg fehlerhaft hohe werte annimmt; Ziel ist das skippen dieses Bereichs)
          #r_peaks = detectors.swt_detector(ecg_lead)          # Detektion auf gekürzten
          #
          #if len(r_peaks)<10:                                 # Wenns immernoch nicht klappt überschreiben wir peaks mit 0 
          #  r_peaks = [0,1]
          #  # continue                          -> zum skippen der Messung müssten wir auch beim training die label nr rauswerfen
        except:
          print('Zu wenige QRS-Komplexe, durch das Überspringen der ersten fünf Sekunden schlägt fehl!')
        try:
          peak_to_peak_diff = (np.diff(r_peaks))   #/fs*1000)                # Abstände der R-Spitzen.
        except:
          print('Die Bestimmung der Peak to Peak Abstände schlägt fehl!')
        try:
          sdnn = np.append(sdnn,np.std(np.diff(r_peaks)/fs*1000))         # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden.
        except:
          print('Berechung der Peak to Peak Standartabweichung schlägt fehl!')
        ### Frequenzbereich
        try:    
          yf = fft(ecg_lead)                                    # Berechnung des komplexen Spektrums.
          r_yf = 2.0/N * np.abs(yf[0:N//2])                     # Umwandlung in ein reelles Spektrum.
        except:
          print('Die Transforation in den Frequenzbereich schlägt fehl!')
        try:
          normier_faktor = (np.sum(r_yf))                     # Inverses Integral über Frequenzbereich  
                                                              # Gesamt integ, weil unten direkt der gesamte freq. bereich normiert wird
        except:
          print('Die Berechung des Normierfaktors des Frequenzbandes schlägt fehl!')
                                                           

        ### LowPass Filter
        try:
          yf_lowPass = np.array([])                            # Tiefpassfilter von Frequenz (0-450)*fd, dass entspricht (0-15)Hz.
          for i in range(0,450):
            yf_lowPass = np.append(yf_lowPass, r_yf[i])
          if math.isnan(r_yf[i]):
            print("error 1")
        except:
          print('Der Tiefpassfilter schlägt fehl!')     
        ### BandPass Filter
        try:
          yf_bandPass = np.array([])                           # Bandpassfilter von Frequenz (451-3500)*fd, dass entspricht (15-116)Hz.
          for i in range(451,3500):
            yf_bandPass = np.append(yf_bandPass, r_yf[i])
          if math.isnan(r_yf[i]):
            print("error 2")
        except:
          print('Der Bandpassfilter schlägt fehl!')    
        ### HighPass Filter                                   # Hochpassfilter von Frequenz (3501-3999)*fd, dass entspricht (116-133)Hz.
        try:
          yf_highPass = np.array([])
          for i in range(3501,3999):
            yf_highPass = np.append(yf_highPass, r_yf[i])
          if math.isnan(r_yf[i]):
            print("error 3")
        except:
          print('Der Hochpassfilter schlägt fehl!')
########### Features:       Relatives Gewicht der Unter-, Mittel- und Oberfreqeunzen.
        try:
          relativ_lowPass = np.append(relativ_lowPass, np.sum(yf_lowPass)/normier_faktor)
          if math.isnan(np.sum(yf_lowPass)/normier_faktor):
            print("error 4")
          
          relativ_bandPass = np.append(relativ_bandPass, np.sum(yf_bandPass)/normier_faktor)
          if math.isnan(np.sum(yf_bandPass)/normier_faktor):
            print("error 5")
          
          relativ_highPass = np.append(relativ_highPass, np.sum(yf_highPass)/normier_faktor)
          if math.isnan(np.sum(yf_highPass)/normier_faktor):
            print("error 6")
        except:
          print('Rechnung der relativen Banddominanz!')  
########### Feature:       Maximaler Ausschlag/Amplitude einer Messung.
        try:
          max_amplitude = np.append(max_amplitude, max(r_yf))
          if math.isnan(max(r_yf)):
            print("error 7")
        except:
          print('Berechung der mAximalen Amplitude schlägt fehl!')
########### Features:       R-Spitzen Abstand und Anzahl einer Messung.
        try:
          peaks_per_measure = np.append(peaks_per_measure, len(r_peaks))
          if math.isnan(len(r_peaks)):
            print("error 8")
        except:
          print('Berechung der Anzahl an Peaks schlägt dehl!')
        try:
          peak_diff_mean = np.append(peak_diff_mean, np.mean(peak_to_peak_diff))
          if math.isnan(np.mean(peak_to_peak_diff)):
            print("error 9")
        except:
          print('Die Berechung des Mittelwertes der Peak to Peak Intervalle schlägt fehl!')
        try:  
          peak_diff_median = np.append(peak_diff_median, np.median(peak_to_peak_diff))
          if math.isnan(np.median(peak_to_peak_diff)):
            print("error 10")
        except:
          print('Die Berechung des Medians der Peak to Peak Intervalle schlägt fehl!')  
########### Feature:        Anzahl an Spektrum-Spitzen im Niederfrequenzband.
        try:
          max_peak_sp = max(r_yf)                               # Ermittlung der höchsten Spitze.
          peaks_low = np.array([])                    
          for i in range(0, 4500):                   # Alle Spitzen übernehmen welche 80% der  höchsten Spitze erreichen.
            if r_yf[i] > 0.8*max_peak_sp:
                peaks_low = np.append(peaks_low, r_yf[i])
          peaks_per_lowPass = np.append(peaks_per_lowPass, peaks_low.size)  # Ermittlung der Anzahl der Spitzen mit mindesten 80% der maximal Spitze.
          if math.isnan(peaks_low.size):
            print("error 11")
        except:
          print('Die Berechung der Peaks im Niederfrerquenzband schlägt fehl!')
########### Feature:        RMSSD
        try:
          n = peak_to_peak_diff.size                 # Anzahl an R-Spitzen-Abständen
          sum = 0.0
          for i in range(0, n-2):                    # Berechnung des RMSSD-Wertes
            sum += (peak_to_peak_diff[i + 1] - peak_to_peak_diff[i])**2
          if math.isnan(sum):
            print("error sum")
          rmssd = np.append(rmssd, math.sqrt(1/((n-1))*sum))
        except:
          print('Die Berechung der RMSSD schlägt fehl!')
                                                                          #### pyhrv - funktionen
########### Feature:        RMSSD (pyhrv)
        try:
          result_rmssd = td.rmssd(peak_to_peak_diff)
          rmssd_neu = np.append(rmssd_neu ,result_rmssd['rmssd'])
        except:
          print('Die Berechung der RMSSD (pyhrv) schlägt fehl!')
########### Feature:        SDNN (pyhrv)
        try:
          result_sdnn = td.sdnn(peak_to_peak_diff)
          sdnn_neu = np.append(sdnn_neu, result_sdnn['sdnn'])
        except:
          print('Die Berechung der SDNN (pyhrv) schlägt fehl!')   
########### Feature:        NN50 (pyhrv)    +     pNN50 (pyhrv)
        try:
          result_NN50 = td.nn50(peak_to_peak_diff)
          nn50 = np.append(nn50, result_NN50['nn50'])
          pNN50 = np.append(pNN50,result_NN50['pnn50'])
        except:
          print('Die berechung der NN50 (pyhrv) + pNN50 (pyhrv) schlägt fehl!')
########### Feature:        NN20 (pyhrv)    +     pNN20 (pyhrv)
        try:
          result_NN20 = td.nn20(peak_to_peak_diff)
          nn20 = np.append(nn20,result_NN20['nn20'])
          pNN20 = np.append(pNN20,result_NN20['pnn20'])
        except:
          print('Die Berechung der NN20 (pyhrv) + pNN20 (pyhrv) schlägt fehl!')
    
###################
        if (idx % 100)==0:
          print("Features von: \t" + str(idx) + "\t EKG Signalen wurden verarbeitet.")

        if(set==2):
          ## Erstellen der Feature-Matrix inklusive der Labels.       # transpose weil für tree brauchen wir die Form
          features =np.transpose(np.array([ relativ_lowPass, relativ_highPass, relativ_bandPass, max_amplitude, sdnn, peak_diff_median, peaks_per_measure, peaks_per_lowPass, peak_diff_mean, rmssd, rmssd_neu, sdnn_neu, nn20, nn50, pNN20, pNN50]))
        if(set==1):
          ## Erstellen der Feature-Matrix inklusive der Labels.       # transpose weil für tree brauchen wir die Form
          features =np.transpose(np.array([ sdnn])) # nn20 ist stärkste

<<<<<<< HEAD
        # print("yf:",ecg_lead.size)
        # print("yf:",yf.size)
        # print("yf:",yf.shape)

        # print("yf:",r_yf.size)
        # print("yf:",r_yf.shape)
        print('Messung: ',idx)
      except:
        print('Messung', idx, 'wird ignoriert')
        for num in enumerate(feature_list):
          if(num.size == idx):
            np.delete(num, idx)
      else:
        continue
=======
>>>>>>> a012a54cdf9eb812c7658957b4c6657fbe8b4bf3
    return features

    #####################################################################################    Plots

    ## Erstellen eines Diagrammes.
    #fig, axs = plt.subplots(2,1, constrained_layout=True)
    #axs[0].hist(sdnn_normal,2000)
    #axs[0].set_xlim([0, 300])
    #axs[0].set_title("Normal")
    #axs[0].set_xlabel("SDNN (ms)")
    #axs[0].set_ylabel("Anzahl")
    #axs[1].hist(sdnn_afib,300)
    #axs[1].set_xlim([0, 300])
    #axs[1].set_title("Vorhofflimmern")
    #axs[1].set_xlabel("SDNN (ms)")
    #axs[1].set_ylabel("Anzahl")
    #plt.show()

    #sdnn_total = np.append(sdnn_normal,sdnn_afib) # Kombination der beiden SDNN-Listen
    #p05 = np.nanpercentile(sdnn_total,5)          # untere Schwelle
    #p95 = np.nanpercentile(sdnn_total,95)         # obere Schwelle
    #thresholds = np.linspace(p05, p95, num=20)    # Liste aller möglichen Schwellwerte

    ####################################################################################    F1 Stuff
    #F1 = np.array([])
    #for th in thresholds:
    #  TP = np.sum(sdnn_afib>=th)                  # Richtig Positiv
    #  TN = np.sum(sdnn_normal<th)                 # Richtig Negativ
    #  FP = np.sum(sdnn_normal>=th)                # Falsch Positiv
    #  FN = np.sum(sdnn_afib<th)                   # Falsch Negativ
    #  F1 = np.append(F1, TP / (TP + 1/2*(FP+FN))) # Berechnung des F1-Scores

    #th_opt=thresholds[np.argmax(F1)]              # Bestimmung des Schwellwertes mit dem höchsten F1-Score

    #if os.path.exists("model.npy"):
    #    os.remove("model.npy")
    #with open('model.npy', 'wb') as f:
    #    np.save(f, th_opt)






    ######################################################### PLOTS

    #fig, ax = plt.subplots()
    #ax.plot(thresholds,F1)
    #ax.plot(th_opt,F1[np.argmax(F1)],'xr')
    #ax.set_title("Schwellwert")
    #ax.set_xlabel("SDNN (ms)")
    #ax.set_ylabel("F1")
    #plt.show()

    #fig, axs = plt.subplots(2,1, constrained_layout=True)
    #axs[0].hist(sdnn_normal,2000)
    #axs[0].set_xlim([0, 300])
    #tmp = axs[0].get_ylim()
    #axs[0].plot([th_opt,th_opt],[0,10000])
    #axs[0].set_ylim(tmp)
    #axs[0].set_title("Normal")
    #axs[0].set_xlabel("SDNN (ms)")
    #axs[0].set_ylabel("Anzahl")
    #axs[1].hist(sdnn_afib,300)
    #axs[1].set_xlim([0, 300])
    #tmp = axs[1].get_ylim()
    #axs[1].plot([th_opt,th_opt],[0,10000])
    #axs[1].set_ylim(tmp)
    #axs[1].set_title("Vorhofflimmern")
    #axs[1].set_xlabel("SDNN (ms)")
    #axs[1].set_ylabel("Anzahl")
    #plt.show()