# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 19:19:08 2016

@author: Issac

"""
# 此处是 sklearn 中的 SVM 算法，可实现对比

from sklearn.svm import SVC
from preprocess import loadDataSet
import numpy as np

datafile = r'F:\mySVM\testSetRBF.xlsx'
data_X,data_Y = loadDataSet(datafile)

clf = SVC()
clf.fit(data_X,data_Y)

predicted_Y = clf.predict(data_X)
predicted_Y = np.mat(predicted_Y).T
print " train error: %d" % (np.sum(np.sign(np.mat(predicted_Y))!=np.sign(data_Y)))

# 检查测试数据集误差
datafile = r'F:\mySVM\testSetRBF2.xlsx'
data_X_test,data_Y_test = loadDataSet(datafile)
predicted_Y_test = clf.predict(data_X_test)
predicted_Y_test = np.mat(predicted_Y_test).T
print "test error: %d" % (np.sum(np.sign(predicted_Y_test)!=np.sign(data_Y_test)))
