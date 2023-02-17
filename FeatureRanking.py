# -*- coding: utf-8 -*-
""" 
We have a lot of features, using the xgboost feature of showing the gain per feature gives us a idea which of the features 
are helpful and which seem useless. To be more sure of the ranking of the features we will use the PCA in this script to
get the features that contribute the most and then try to visualize the 3D of the features and samples.
- PCA for feature extraction
- PCA for data visualization
"""
from wettbewerb import load_references
from features_112 import features
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
import numpy as np
from numpy import genfromtxt


# load the features
#ecg_leads,ecg_labels,fs,ecg_names = load_references()

features = genfromtxt('learningfeatures_14features.csv', delimiter=',')


# create a PCA object and fit
pca = PCA(n_components=0.95)  # Minimize till we are left with 95% of the variance
pca.fit(features)

# get the coefficients of the linear combination that defines each principal component
coef = pca.components_

# find the original features that contribute the most to each principal component
best_features = np.abs(coef).argmax(axis=1)

# print the indices of the best features
print("Best Features:", best_features)
print("anzahl feats:", pca.n_features_)
print("anzahl samples:", pca.n_samples_)