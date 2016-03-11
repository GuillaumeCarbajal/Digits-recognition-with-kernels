# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 08:24:44 2016

@author: ivaylo
"""

from functions import *
from Kernel_SVM_Multiclass import *
import pandas as pd
import numpy as np

# Data
X_df = pd.read_csv('Xtr.csv',header=None)
X_test = pd.read_csv('Xte.csv',header=None)

y = pd.read_csv("Ytr.csv")["Prediction"].values.copy()

# Sobel operator for gradient and angle mat
G_mat, A_mat = sobel(X_df.values)
G_te, A_te = sobel(X_test.values)

#%% Visu
plt.close('all')

n_line = 1500
visu_sample(X_df.values,n_line,[28,28])
visu_sample(G_mat,n_line,[28,28])
visu_sample(A_mat,n_line,[28,28])

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


#Test data
bin_nb = 12
hog_mat = []
for n_l in range(np.shape(G_te)[0]):
    # gradient and angle matrices
    G = np.reshape(G_te[n_l,:],(28,28))
    A = np.reshape(A_te[n_l,:],(28,28))
    A = 360*(A + pi)/(2*pi)
    hist_list1 = hog3(G,A,4,bin_nb)
    hist_list2 = hog3(G,A,7,bin_nb)
    hist_list3 = hog3(G,A,14,bin_nb)
    hist_list = np.concatenate([hist_list1,hist_list2,hist_list3],axis=1)
    hog_mat.append(hist_list)
    if(n_l%500==0):
        print n_l
    
hog_test = np.array(hog_mat)[:,0,:]
# Adding weights: 4 for hist_list1, 2 for hist_list2 and 1 for hist_list3
hog_test = np.concatenate([4.0*hog_test[:,:432],2.0*hog_test[:,432:432+108],hog_test[:,432+108:]],axis=1)


#%% Classification

def inter_kernel(X,Y):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        if(i%500==0):
            print i
        for j, y in enumerate(Y):
            gram_matrix[i, j] = np.sum(np.min(np.array([x,y]),axis=0))
    return gram_matrix

# Computing Gram matrices
K_train = inter_kernel(hog_train,hog_train)
K_test = inter_kernel(hog_test,hog_train)

y_fi = y.astype('double')

# Train SVM
labels, labels_index, alphas, bs, y_oaa = one_vs_all_train(K_train,y_fi)

# Predict SVM
y_pred = one_vs_all_predict(K_test, labels, labels_index, alphas, bs, y_oaa)


#%% Output

out = pd.DataFrame(data={"Id":X_test.index.values+1, "Prediction": y_pred})
out["Prediction"] = out["Prediction"].values.astype('int')
out.to_csv('sub3.csv', index = False, quoting = 3)















