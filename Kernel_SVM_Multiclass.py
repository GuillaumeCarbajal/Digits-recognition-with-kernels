from Kernel_SVM import *
import cvxopt
import numpy as np

def one_vs_all_train(K_train, y_train):
    
    # get the indices of each class k
    labels = np.unique(y_train)
    labels_index = []
    for k in labels:
        labels_index.append([i for i, j in enumerate(y_train) if j == k])
    
    # Encode a new y with 1 if class k, -1 otherwise
    y_new = y_train.copy()
    
    alphas=[]
    bs = []
    y_oaa = []
    for k, ktem in enumerate(labels_index):
        y_new = y_train.copy()
        y_new[labels_index[k]] = 1
        y_new[[i for i, j in enumerate(y_train) if not i in labels_index[k]]] = -1. / (len(labels) - 1)
        
        # Train of SVM
        alpha, b = svm(K_train, y_new)
        
        # Store the values of alpha and b for the class k
        alphas.append(alpha)
        bs.append(b)
        y_oaa.append(y_new)
    
    return labels, labels_index, alphas, bs, y_oaa


def one_vs_all_predict(K_test, labels, labels_index, alphas, bs, y_oaa):
    y_pred = []
    for i, item in enumerate(K_test):
        
        # prediction of multi class SVM :
        # on attribue la classe dont la fonction \sum alpha_i y_i K(x_i, x) + b est la + grande
        f_max = item.dot(alphas[0]*y_oaa[0]) + bs[0]
        pred = labels[0]
        for label, alpha, y, b in zip(labels, alphas, y_oaa, bs):
            if(f_max < (item.dot(alpha*y) + b)):
                f_max = item.dot(alpha*y) + b
                pred = label
        
        y_pred.append(pred)

    return y_pred