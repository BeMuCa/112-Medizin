import csv;
import scipy.io as sio;
import matplotlib.pyplot as plt;
import numpy as np;
from ecgdetectors import Detectors;
import os;
from scipy.fft import fft, fftfreq;
from wettbewerb import load_references;
import math;
import features_112;

ecg_leads,ecg_labels,fs,ecg_names = load_references();

########################### Calculate the features ######################################################

features = features_112.features(ecg_leads,fs);

########################### Save features in csv ######################################################

np.savetxt("learningfeatures_16.csv", features, delimiter=",")
