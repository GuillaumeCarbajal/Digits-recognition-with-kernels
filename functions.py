# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 12:21:05 2016

@author: ivaylo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sg
from math import pi

def atan2(x,y):
    angl = 2*np.arctan(y/(np.sqrt(x**2+y**2)+x))
    angl[angl!=angl]=0
    return angl
    
def sobel(X):
    n_samples, n_features = X.shape
    G = []
    Angl = []
    op1 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    op2 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])    
    for i in range(n_samples):
        A = np.reshape(X[i,:],(28,28))
        Gx = sg.convolve(op1,A)[:-2,:-2]
        Gy = sg.convolve(op2,A)[:-2,:-2]
        g = np.sqrt(Gx**2+Gy**2)
        angl = atan2(Gx,Gy) 
        G.append(list(np.reshape(g,(np.shape(g)[0]**2,1))))
        Angl.append(list(np.reshape(angl,(np.shape(angl)[0]**2,1))))
    return np.array(G)[:,:,0], np.array(Angl)[:,:,0]

def visu_sample(X,n_line,shape):
    plt.figure()
    plt.subplot(121)
    plt.plot(X[n_line,:])
    plt.subplot(122)
    plt.imshow(X[n_line,:][::-1].reshape(28,28).T[::-1,::-1], cmap=plt.cm.gray, interpolation='none')
    plt.xticks(())
    plt.yticks(())

def block_per_line(cell_size):
    count = 1
    temp = 2*cell_size
    while(temp<28):
        count += 1
        temp += cell_size       
    return count


def hog(G,A,cell_size,bin_nb=18):
    # Divide image into cells of cell_size**2 size
    cell_nb = int(np.shape(G)[0]/cell_size)
    cells = []
    for i in range(cell_nb):
        for j in range(cell_nb):
            cells.append(np.mean((G*A)[i*cell_size:(i+1)*cell_size,j*cell_size:(j+1)*cell_size]))  
    cells = np.reshape(np.array(cells),(cell_nb,cell_nb))
    hist_list = []
    hist_val = np.linspace(0,360,bin_nb)
    # Count the number of overlapping block per line/column
    blk_line = block_per_line(cell_size)
    # Compute hist for each block
    for k in range(blk_line):
        for l in range(blk_line):
            # For One block we get the 2x2 cells corresponding
            cell_list = list(np.reshape(cells[k:k+2,l:l+2],(4,)))
            hist = np.zeros(bin_nb) 
            for cell_val in cell_list:
                # For the hist we put a value in two closest bin with ratio
                indices_two_closest = np.argsort((hist_val - cell_val)**2)[:2]
                two_closest_val = hist_val[indices_two_closest]
                ceil_two_closest_diff = np.abs(two_closest_val - cell_val)
                ratio_two_closest_val = 1 -  ceil_two_closest_diff/(np.sum(ceil_two_closest_diff))
                hist[indices_two_closest] += ratio_two_closest_val*two_closest_val
            hist_list.append(hist)
    hist_list = np.array(hist_list)
    hist_list = np.reshape(hist_list,(1,np.size(hist_list)))
    return hist_list
    
def hog2(G,A,cell_size,bin_nb,std_gauss):
    from scipy.signal import gaussian
    h = gaussian(bin_nb,std_gauss)
    h = h/np.sum(h)
    # Divide image into cells of cell_size**2 size
    hist_list = []
    hist_val = np.linspace(0,360,bin_nb)
    # Count the number of overlapping block per line/column
    blk_line = block_per_line(cell_size)
    # Compute hist for each block
    for k in range(blk_line):
        for l in range(blk_line):
            A_blk = A[k*cell_size:(k+2)*cell_size,l*cell_size:(l+2)*cell_size]
            G_blk = G[k*cell_size:(k+2)*cell_size,l*cell_size:(l+2)*cell_size]
            hist = np.zeros(bin_nb) 
            for line in range(np.shape(A_blk)[0]):
                for col in range(np.shape(A_blk)[1]):          
                    angl = A_blk[line,col]
                    grad = G_blk[line,col]
                    indices_two_closest = np.argsort((hist_val - angl)**2)[:2]
                    two_closest_val = hist_val[indices_two_closest]
                    angl_two_closest_diff = np.abs(two_closest_val - angl)
                    ratio_two_closest_val = 1 -  angl_two_closest_diff/(np.sum(angl_two_closest_diff))
                    hist[indices_two_closest] += grad*ratio_two_closest_val*two_closest_val
                hist_list.append(np.convolve(h,hist,mode='same'))
    hist_list = np.array(hist_list)
    hist_list = np.reshape(hist_list,(1,np.size(hist_list)))
    return hist_list
    
def hog3(G,A,cell_size,bin_nb):
    # Divide image into cells of cell_size**2 size
    hist_list = []
    hist_val = np.linspace(0,360,bin_nb)
    # Count the number of overlapping block per line/column
    blk_line = block_per_line(cell_size)
    # Compute hist for each block
    for k in range(blk_line):
        for l in range(blk_line):
            A_blk = A[k*cell_size:(k+2)*cell_size,l*cell_size:(l+2)*cell_size]
            G_blk = G[k*cell_size:(k+2)*cell_size,l*cell_size:(l+2)*cell_size]
            hist = np.zeros(bin_nb) 
            for line in range(np.shape(A_blk)[0]):
                for col in range(np.shape(A_blk)[1]):          
                    angl = A_blk[line,col]
                    grad = G_blk[line,col]
                    indices_two_closest = np.argsort((hist_val - angl)**2)[:2]
                    two_closest_val = hist_val[indices_two_closest]
                    angl_two_closest_diff = np.abs(two_closest_val - angl)
                    ratio_two_closest_val = 1 -  angl_two_closest_diff/(np.sum(angl_two_closest_diff))
                    hist[indices_two_closest] += grad*ratio_two_closest_val*two_closest_val
                hist_list.append(hist)
    hist_list = np.array(hist_list)
    hist_list = np.reshape(hist_list,(1,np.size(hist_list)))
    return hist_list

def pre_pro(X_val,buff=30):
    X = X_val.copy()
    col_nb = np.shape(X)[1]
    X[X<0.0] = 0.0
    begin_points = np.argmin(X<0.85,axis=1) - buff
    begin_points[begin_points<=0.0] = 0
    end_points = col_nb - np.argmin(X[:,::-1]<0.85,axis=1) + buff
    end_points[end_points>=col_nb-1] = col_nb -1
    for i in range(np.shape(X)[0]):
        X[i,:begin_points[i]] = 0.0
        X[i,end_points[i]:] = 0.0
    X[X<0.60] = 0.0
    return X
    
def plot_digit(X_df, y, value, fig_nb, n_rows=4, n_cols=5, max_iter=200):
    temp=0
    print value
    for k in range(max_iter):
        if(temp>=fig_nb):
            break
        if(y[k] == value):
            plt.subplot(n_rows, n_cols, temp+1)
            plt.imshow(np.reshape(X_df[k,:],(28,28)), cmap=plt.cm.gray, interpolation='none')
            plt.xticks(())
            plt.yticks(())
            temp += 1

####################
# data augmentation (rotation of the dataset)
###################
from scipy import ndimage

def rotate_dataset(data, angle):
    n_samples, n_features = data.shape
    new_data = []
    for i, item in enumerate(data):
        X = item.reshape(28,28).copy()
        rotate_X = ndimage.rotate(X, angle, reshape = False)
        rotate_X = rotate_X.reshape(n_features, 1)
        new_data.append(rotate_X)
    new_data = np.array(new_data)
    return new_data
