# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 15:05:24 2016

@author: Issac
"""

import numpy as np

class MySVM:

    def __init__(self,data_X,data_Y,C=2000,toler=0.001,maxIter=10,kTup=('rbf',0.5)):
        self.X = data_X
        self.y = data_Y
        self.m,self.n = np.shape(data_X)
        self.b = 0
        self.C = C
        self.tol = toler
        self.maxIter = maxIter
        self.kTup = kTup
        self.alpha = np.mat(np.zeros((self.m,1))) 
        self.eCache = np.mat(np.zeros((self.m,2)))
        self.K = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = self.__calcKernel(self.X,self.X[i,:])
            

    def __calcKernel(self,X,A):
        m,n = np.shape(X)
        K = np.mat(np.zeros((m,1)))
        if self.kTup[0]=='lin':
            K = X*A.T
        elif self.kTup[0]=='rbf':
            for j in range(m):
                delta = X[j,:] - A
                K[j] = delta*delta.T
            K = np.exp(K/(-1*self.kTup[1]**2))
        else:
            raise NameError('The Kernel is not recognized.')
        return K
                
    def fit(self):
        iter = 0
        entireSet = True
        alphaPairsChanged = 0
        while (iter<self.maxIter) and ((alphaPairsChanged>0) or entireSet):
            alphaPairsChanged = 0
            if entireSet:                                                       # 更新全部的 alpha
                for i in range(self.m):
                    alphaPairsChanged += self.__update(i)
                print "fullSet iter: %d, alphaPairsChanged: %d" % \
                (iter,alphaPairsChanged)
                iter += 1
            else:                                                               #更新非边界值 alpha
                nonBoundAlpha = np.nonzero(np.multiply((self.alpha>0),(self.alpha<self.C)))[0]
                for i in nonBoundAlpha:
                    alphaPairsChanged += self.__update(i)
                print "nonBound iter: %d, alphaPairsChanged: %d" % \
                (iter,alphaPairsChanged)
                iter += 1
            if entireSet: 
                entireSet = False
            elif alphaPairsChanged == 0:
                entireSet = True
        print "the number of iters is %d / %d" % (iter,self.maxIter)
        sV = np.nonzero(self.alpha.A > 0)[0]
        supportVecrot = self.X[sV,:]
        return supportVecrot
                       
    def __update(self,i):
        Ei = self.__calcEk(i)
        # alpha误差比较大，并且可以优化
        if ((self.y[i]*Ei < -self.tol) and (self.alpha[i]<self.C)) or \
        ((self.y[i]*Ei>self.tol) and (self.alpha[i]>0)):                
            j,Ej = self.__selectJ(i)
            alphaIold = self.alpha[i].copy()
            alphaJold = self.alpha[j].copy()
            if (self.y[i] != self.y[j]):
                L = max(0,self.alpha[j]-self.alpha[i])
                H = min(self.C,(self.C+self.alpha[j]-self.alpha[i]))
            else:
                L = max(0,(self.alpha[j]+self.alpha[i]-self.C))
                H = min(self.C,(self.alpha[j]+self.alpha[i]))
            if L==H:
                return 0
            eta = self.K[i,i] + self.K[j,j] - 2.0 * self.K[i,j]
            if eta<=0:
#                print "eta <= 0"
                return 0
            self.alpha[j] += self.y[j]*(Ei-Ej)/eta
            if self.alpha[j] > H:
                self.alpha[j] = H
            elif self.alpha[j] < L:
                self.alpha[j] = L
            self.__updateEk(j)
            if (np.abs(self.alpha[j]-alphaJold)<0.00001):
#                print "j is not moving enough."
                return 0
            self.alpha[i] += (alphaJold - self.alpha[j])*self.y[j]*self.y[i]
            self.__updateEk(i)
            bi = Ei + self.y[i]*(self.alpha[i] - alphaIold)*self.K[i,i] + \
            self.y[j]*(self.alpha[j] - alphaJold)*self.K[i,j] + self.b
            bj = Ej + self.y[i]*(self.alpha[i] - alphaIold)*self.K[i,j] + \
            self.y[j]*(self.alpha[j] - alphaJold)*self.K[j,j] + self.b
            if (0<self.alpha[i]<self.C):
                self.b = bi
            elif(0<self.alpha[j]<self.C):
                self.b = bj
            else:
                self.b = (bi+bj)/2.0
            return 1
        else:
            return 0
        
    def __updateEk(self,k):
        Ek = self.__calcEk(k)
        self.eCache[k] = [1,Ek]
    
    def predict(self,predict_X):
        m,n = np.shape(predict_X)
        y_predicted = np.mat(np.zeros((m,1)))
        for i in range(m):
            y_predicted[i] = np.multiply(self.alpha,self.y).T * \
            self.__calcKernel(self.X,predict_X[i,:]) - self.b
        y_predicted = np.sign(y_predicted)
        return y_predicted
    
    def __selectJRadom(self,i):
        j=i
        while (j==i):    
            j = np.random.randint(self.m)
            return j
              
    def __selectJ(self,i):
        Ei = self.__calcEk(i)
        self.eCache[i] = [1,Ei]
        j = 0
        Ej = 0
        maxDeltaE = 0
        validEk = np.nonzero(self.eCache[:,0])[0]
        if validEk.size>1:
            for k in validEk:
                if k==i: 
                    continue
                else:
                    Ek = self.__calcEk(k)
                    deltaE = np.abs(Ei-Ek)
                    if deltaE>maxDeltaE:
                        maxDeltaE = deltaE
                        j = k
                        Ej = Ek
        else:
            j = self.__selectJRadom(i)
            Ej = self.__calcEk(j) 
        return j,Ej
                            
    def __calcEk(self,k):
        fXk = (np.multiply(self.alpha,self.y).T)*self.K[:,k] - self.b
        Ek = fXk - float(self.y[k])
        return Ek
