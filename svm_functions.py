# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 11:03:24 2016

@author: ivaylo
"""

import numpy as np

################################ kernel #######################################

def circular_kernel(X,Y):
    """
    Gram matrix of 'circular' kernel:
    phi([x1,x2]) = [x1, x2, x1**2 + x2**2]
    """
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i, j] = x[0]*y[0] + x[1]*y[1] + (x[0]**2 + x[1]**2)*(y[0]**2 + y[1]**2)
    return gram_matrix
    
################################ SVM ##########################################
def svm_kernel_train(y,lambd, iter_max,gram):
    """ Kernelized Pegasos Algorithm:
        http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf (page 11)
        Return alpha of dual and bias. y and gram are the labels and 
        the Gram matrix of the train data.
    """
    m = gram.shape[0]
    alpha = np.zeros(m)
    t=1
    # Stochastic part
    rand_ind = np.random.randint(0,m,iter_max)
    for ite in xrange(iter_max):
        # choose one sample uniformly
        i = rand_ind[ite]
        condition = (y[i]/(lambd*t)) * ((y*alpha).dot(gram[i,:])) < 1 
        if(condition==True):
            alpha[i] = alpha[i] + 1
        t = t+1
    # bias
    b = 0.5*(np.mean(gram.dot(alpha)[y==1]) + np.mean(gram.dot(alpha)[y==-1]))

    return alpha, b
    
def svm_kernel_predict(gram_test,alpha,b):
    """
    Return predictions for test data represented under gram_test.
    alpha and b come from svm_kernel_train.
    """
    pred = np.sign(gram_test.dot(alpha) - b)
    return pred 

