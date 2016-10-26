# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 13:31:35 2016

@author: Issac
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(datafile):
    data = pd.read_excel(datafile,header=None)
    data = np.mat(data) # 将表格转换为矩阵
    np.random.shuffle(data)
    m,n = np.shape(data)
    data_X = data[:,:n-1]
    data_Y = data[:,-1]
    return data_X,data_Y
    
def figure_2D(data_X1,data_X2,data_Y,supportVecrot):
    plt.figure()
    positive = np.nonzero(data_Y>0)[0]
    negative = np.nonzero(data_Y<0)[0]
    plt.scatter(data_X1[positive],data_X2[positive],c='r',label='positive')
    plt.scatter(data_X1[negative],data_X2[negative],c='g',label='negative')
    plt.scatter(supportVecrot[:,0],supportVecrot[:,1],edgecolors='k',facecolor='None',s=50,label='supportVecrot')
    plt.title('Train data set')
    plt.legend()
    plt.show()
