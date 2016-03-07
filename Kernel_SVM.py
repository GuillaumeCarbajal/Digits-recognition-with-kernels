# -*- coding: utf-8 -*-
"""
Created on Sun Mar 06 12:11:56 2016

@author: Rahma
"""
import cvxopt
import numpy as np
import numpy.linalg as la


# Train SVM

def svm(K , y, C=1.0):
   
   n_samples = len(y)
   P = cvxopt.matrix((np.outer(y, y) * K).tolist())
   q = cvxopt.matrix(-1 * np.ones(n_samples))
   G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
   h_std = cvxopt.matrix(np.zeros(n_samples))

   # a_i \leq c
   G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
   h_slack = cvxopt.matrix(np.ones(n_samples) * C)
   G = cvxopt.matrix(np.vstack((G_std, G_slack)))
   h = cvxopt.matrix(np.vstack((h_std, h_slack)))
    
   A = cvxopt.matrix(y, (1, n_samples))
   b = cvxopt.matrix(0.0)
    
   sol = cvxopt.solvers.qp(P,q,G,h,A,b)
   alpha = np.ravel(sol['x'])
   #inter = -0.5*(np.mean(K.dot(alpha)[y==1]) + np.mean(K.dot(alpha)[y!=1])) 
   #One support vector machine
   
   result = 0
   for i in range(n_samples):
      result += (K[i].dot(alpha*y))

   inter = np.mean(y) - result / n_samples

   print 'je comprends le changement 4'
   return alpha, inter
   

# Predict

def svm_kernel_predict(K_test,y, alpha,b): 
   pred = np.sign(K_test.dot(alpha*y) + b) 
   return pred
   
   
# Define different kernels
   
# Different Kernels

def inter_kernel(X,Y): 
   gram_matrix = np.zeros((X.shape[0], Y.shape[0])) 
   for i, x in enumerate(X): 
      for j, y in enumerate(Y): 
         gram_matrix[i, j] = np.sum(np.min(np.array([x,y]),axis=0)) 
   return gram_matrix 


def linear_kernel(X,Y):
   return np.inner(X,Y)

def pol_kernel(X,Y):
   gram_matrix = np.zeros((X.shape[0], Y.shape[0])) 
   for i, x in enumerate(X): 
      for j, y in enumerate(Y): 
         gram_matrix[i, j] = x[0]*y[0] + x[1]*y[1] + (x[0]**2 + x[1]**2)*(y[0]**2 + y[1]**2) 
   return gram_matrix 


def gaussian(X, Y, sigma):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0])) 
    for i, x in enumerate(X): 
        for j, y in enumerate(Y): 
            exponent = -np.sqrt(la.norm(x-y) ** 2 / (2 * sigma ** 2))
            gram_matrix[i, j] = np.exp(exponent)
    return gram_matrix

def inter_kernel(X,Y):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i, j] = np.sum(np.min(np.array([x,y]),axis=0))
    return gram_matrix