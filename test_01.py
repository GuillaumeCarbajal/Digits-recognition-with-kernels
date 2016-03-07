# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 10:11:39 2016

@author: ivaylo
"""

from functions import *
from Kernel_SVM_Multiclass import *

import pandas as pd
import numpy as np


# Data
X_df = pd.read_csv('Xtr.csv',header=None)
y = pd.read_csv("Ytr.csv")["Prediction"].values.copy()

# Sobel operator for gradient and angle mat
G_mat, A_mat = sobel(X_df.values)


#%% Visu
plt.close('all')

n_line = 1500
visu_sample(X_df.values,n_line,[28,28])
visu_sample(G_mat,n_line,[28,28])
visu_sample(A_mat,n_line,[28,28])


#%% test histogram of gradient

std = 1.0
bin_nb = 12
hog_mat = []
for n_l in range(np.shape(G_mat)[0]):
    # gradient and angle matrices
    G = np.reshape(G_mat[n_l,:],(28,28))
    A = np.reshape(A_mat[n_l,:],(28,28))
    A = 360*(A + pi)/(2*pi)
    hist_list1 = hog2(G,A,4,bin_nb,std)
    hist_list2 = hog2(G,A,7,bin_nb,std)
    hist_list3 = hog2(G,A,14,bin_nb,std)
    weigths = [1.,1.,1.]
    hist_list = np.concatenate([weigths[0]*hist_list1,weigths[1]*hist_list2,weigths[2]*hist_list3],axis=1)
    hog_mat.append(hist_list)
    if(n_l%500==0):
        print n_l
        
hog_mat2 = np.array(hog_mat)[:,0,:]

#%% Classification

import sklearn.metrics as me
from sklearn.svm import SVC, LinearSVC, libsvm
from sklearn.cross_validation import train_test_split

plt.close('all')

def inter_kernel(X,Y):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i, j] = np.sum(np.min(np.array([x,y]),axis=0))
    return gram_matrix

X_tr, X_te, y_tr, y_te = train_test_split(hog_mat2,y,test_size=0.3)

#clf = SVC(C=0.01,kernel='linear')
#clf = LinearSVC(C=0.0001)
clf = SVC(kernel=inter_kernel)

clf.fit(X_tr,y_tr)
pred = clf.predict(X_te)
print "Precision sklearn: ", me.accuracy_score(y_te,pred)

#y_false = y_te[np.where(y_te!=pred)[0]]
#pred_false = pred[np.where(y_te!=pred)[0]]

#plt.hist(y_false)

#%%
#SVM à la main

import sklearn.metrics as me
from sklearn.cross_validation import train_test_split

plt.close('all')

def inter_kernel(X,Y):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i, j] = np.sum(np.min(np.array([x,y]),axis=0))
    return gram_matrix

X_tr, X_te, y_tr, y_te = train_test_split(hog_mat2,y,test_size=0.3)

y_tr = y_tr.astype('double')
y_te = y_te.astype('double')

# Test on smaller dataset

K_train = inter_kernel(X_tr,X_tr)
K_test = inter_kernel(X_te,X_tr)

# Train SVM
labels, labels_index, alphas, bs, y_oaa = one_vs_all_train(K_train,y_tr)

# Predict SVM
y_pred2 = one_vs_all_predict(K_test, labels, labels_index, alphas, bs, y_oaa)

#Test accuracy
print "Precision à la main ", me.accuracy_score(y_te,y_pred2) 




