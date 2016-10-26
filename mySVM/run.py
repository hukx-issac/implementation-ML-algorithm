# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:35:22 2016

@author: Issac
"""

from mySVM import MySVM
from preprocess import loadDataSet, figure_2D
import numpy as np

'''
每次运行的结果可能会有所不同，
其一，是数据集每次都打乱了，而算法每次都从第一个数据开始寻找 Lagrange 乘子；
其二，最大步长加速算法启动之前，有一个随机选取j的过程，即 mySVM.py 中的 selectJRadom() 函数
'''

datafile = r'F:\mySVM\mySVM\testSetRBF.xlsx'
data_X,data_Y = loadDataSet(datafile)


# 搭建svm模型
model = MySVM(data_X,data_Y,C=20,toler=0.001,maxIter=10,kTup=('rbf',1.3))
supportVector = model.fit()


# 检查训练误差并绘图
predicted_Y = model.predict(data_X)
print "there are %d support vecrots" % (np.shape(supportVector)[0])
print "train error: %.2f" % (np.sum(np.sign(predicted_Y)!=np.sign(data_Y))/float(np.shape(data_Y)[0]))

data_X,data_Y = loadDataSet(datafile)
figure_2D(data_X[:,0],data_X[:,1],data_Y,supportVector)

# 检查测试数据集误差
datafile = r'F:\mySVM\mySVM\testSetRBF2.xlsx'
data_X_test,data_Y_test = loadDataSet(datafile)
predicted_Y_test = model.predict(data_X_test)
print "test error: %.2f" % (np.sum(np.sign(predicted_Y_test)!=np.sign(data_Y_test))/float(np.shape(data_Y_test)[0]))