# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 11:12:12 2016

@author: ivaylo
"""

import numpy as np
from math import pi
import sklearn.metrics as me
import matplotlib.pyplot as plt
from svm_functions import *

plt.close('all')
# New dataset
n = 200
# radius
r1 = 8.0
r2 = 6.0
angles = 2*pi*np.random.rand(n,1) - pi
X1 = np.concatenate([r1*np.cos(angles),r1*np.sin(angles)] , axis=1) + 0.5*np.random.randn(n,2)
X2 = np.concatenate([r2*np.cos(angles),r2*np.sin(angles)] , axis=1) + 0.5*np.random.randn(n,2)
X = np.concatenate([X1,X2],axis=0)
y = np.ones(2*n)
y[n:] = -1

# Visu
plt.figure()
plt.title('data')
plt.plot(X[:n,0],X[:n,1],'bo')
plt.plot(X[n:,0],X[n:,1],'ro')

# Prendre un example sur deux pour le test
X_train = X[::2]
X_test = X[1::2]
y_train = y[::2]
y_test = y[1::2]

gram_train = circular_kernel(X_train,X_train)
gram_test = circular_kernel(X_test,X_test)

# Parameter for svm
lambd = 100.0
iter_max = 200
# Train of SVM
alpha, b = svm_kernel_train(y_train, lambd, iter_max, gram_train)
# Test of SVM
pred = svm_kernel_predict(gram_test,alpha,b)

print "Accuracy score: ", me.accuracy_score(y_test,pred) 





