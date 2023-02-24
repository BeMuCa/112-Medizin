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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# load the features and labels
ecg_leads,ecg_labels,fs,ecg_names = load_references()

features = genfromtxt('learningfeatures_14features.csv', delimiter=',')


# create a PCA object and fit
pca = PCA(n_components=0.90)  # Minimize till we are left with 90% of the variance
pca.fit(features)

# get the coefficients of the linear combination that defines each principal component
coefs = pca.components_

# find the original features that contribute the most to each principal component
best_features = np.abs(coefs).argmax(axis=1)

# print the indices of the best features
print("Best features:", best_features)
print("Nr of feats:", pca.n_features_)
print("Nr of samples:",pca.n_components_)


# Visualization of the top 3 features:

# get the best 3 features from the PCA object
best_3_features = pca.components_[:3, :]

# transform the original data to the reduced 3D space
features_reduced = pca.transform(features)[:, :3]

# color the data points
colors = np.array([])

for nr,y in enumerate(ecg_labels):

    if ecg_labels[nr] == 'N': 
        colors = np.append(colors,'green')
        continue  

    if ecg_labels[nr] == 'A': 
        colors = np.append(colors,'red')
        continue

    else:
        colors= np.append(colors, 'white')


# create a 3D scatter plot of the reduced data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
<<<<<<< HEAD
ax.scatter(features_reduced[:, 0], features_reduced[:, 1], features_reduced[:, 2], c=colors) 
ax.set_xlabel('NN50')
ax.set_ylabel('relativeHighPass')
ax.set_zlabel('peakdiffMedian')
=======
ax.scatter(features_reduced[:, 0], features_reduced[:, 1], features_reduced[:, 2], c=colors)
ax.set_xlabel('nn50')
ax.set_ylabel('relativ_highPass')
ax.set_zlabel('peak_diff_median')
>>>>>>> eb8bf64b2cf2aa40e7b8b5b089771295fd6ceae0
plt.show()




