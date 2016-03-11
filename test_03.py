# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 23:13:05 2016

@author: Rahma
"""

from functions import *
from Kernel_SVM_Multiclass import *
import pandas as pd
import numpy as np
import numpy.linalg as la

# Data
X_df = pd.read_csv('Xtr.csv',header=None)
y = pd.read_csv("Ytr.csv")["Prediction"].values.copy()
# Sobel operator for gradient and angle mat
G_mat, A_mat = sobel(X_df.values)

#%% HOG Transform

#Train data
bin_nb = 12
hog_mat = []
for n_l in range(np.shape(G_mat)[0]):
    # gradient and angle matrices
    G = np.reshape(G_mat[n_l,:],(28,28))
    A = np.reshape(A_mat[n_l,:],(28,28))
    A = 360*(A + pi)/(2*pi)
    hist_list1 = hog3(G,A,4,bin_nb)
    hist_list2 = hog3(G,A,7,bin_nb)
    hist_list3 = hog3(G,A,14,bin_nb)
    hist_list = np.concatenate([hist_list1,hist_list2,hist_list3],axis=1)
    hog_mat.append(hist_list)
    if(n_l%500==0):
        print n_l

hog_train = np.array(hog_mat)[:,0,:]
# Adding weights: 4 for hist_list1, 2 for hist_list2 and 1 for hist_list3
hog_train = np.concatenate([4.0*hog_train[:,:432],2.0*hog_train[:,432:432+108],hog_train[:,432+108:]],axis=1)

#%% Train/Test
import sklearn.metrics as me
from sklearn.cross_validation import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(hog_train,y,test_size=0.3)
#%% Classification

def gaussian_kernel(X,Y):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        if(i%500==0):
            print i
        for j, y in enumerate(Y):
            gram_matrix[i, j] = np.exp(-np.sqrt(la.norm(x-y) ** 2 / (2 * 0.01 ** 2)))
    return gram_matrix

# Computing Gram matrices
K_train = gaussian_kernel(X_tr,X_tr)
K_test = gaussian_kernel(X_te,X_tr)

y_tr = y_tr.astype('double')
y_te = y_te.astype('double')
# Train SVM
labels, labels_index, alphas, bs, y_oaa = one_vs_all_train(K_train,y_tr)

# Predict SVM
y_pred2 = one_vs_all_predict(K_test, labels, labels_index, alphas, bs, y_oaa)

#Test accuracyq
print "Precision à la main ", me.accuracy_score(y_te,y_pred2) 
