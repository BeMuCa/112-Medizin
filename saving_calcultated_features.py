#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Script for saving the features into a csv datum to make the algorithm testing process faster.

"""

import matplotlib.pyplot as plt;
import numpy as np;
from wettbewerb import load_references;
import features_112;

ecg_leads,ecg_labels,fs,ecg_names = load_references();

### Calculate the features

features = features_112.features(ecg_leads,fs,3)

### Save features in csv

np.savetxt("learningfeatures_2_stärksten.csv", features, delimiter=",")
