# -*- coding: utf-8 -*-
"""
Created on Sun Mar 06 12:36:45 2016

@author: Rahma
"""

# Exemple

import os
os.chdir( 'C:\\Users\\Rahma\\Documents\\MasterCourses\\Kernels\\Challenge')

import Kernel_SVM
from Kernel_SVM import svm_kernel_predict
from Kernel_SVM import svm
from Kernel_SVM import pol_kernel
 
import numpy as np
from math import pi
import sklearn.metrics as me
import matplotlib.pyplot as plt

plt.close('all')
# New dataset
n = 200
# radius
r1 = 8.0
r2 = 5.0
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
plt.show()
# Prendre un example sur deux pour le test
X_train = X[::2]
X_test = X[1::2]
y_train = y[::2]
y_test = y[1::2]

K_train = pol_kernel(X_train,X_train)
K_test = pol_kernel(X_train,X_test)

# Train of SVM
alpha, b = svm(K_train,y_train)
# Test of SVM
pred = svm_kernel_predict(K_test, alpha,b)

print "Accuracy score: ", me.accuracy_score(y_test,pred) 