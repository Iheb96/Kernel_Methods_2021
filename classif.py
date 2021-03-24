# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:22:35 2021

@author: Iheb96
"""
import pandas as pd
import numpy as np
import warnings
from kernel import kernel
warnings.filterwarnings('ignore')
import scipy.optimize
class logistic_regression():
  def __init__(self,lamb,kernel_name="Linear",substring_size=6,substring_all=True,degree=2,gamma=0.1):

    self.gamma=gamma
    self.kernel_name=kernel_name
    self.substring_size=substring_size
    self.degree=degree
    self.substring_all=substring_all

    self.lamb=lamb
    

  def logistic_loss(self,yf):
    return np.log(1+np.exp(yf))

  def logistic_regression(self,alpha):
    
    return np.mean(self.logistic_loss(self.y * (self.K @ alpha)))  + (self.lamb/2) * alpha.T @ self.K @ alpha

  def train (self,X,y):
    DNA_chars=["A","T","C","G"]
    self.DNA_seq=[]
    self.X_train=X
    self.y=y
    self.alpha=np.zeros(y.shape)
    n=len(y)
    self.K=np.zeros((n,n))
    if (self.kernel_name=="Linear"):
      self.K=(X@X.T)
    else:

      for i in range(n):
        for j in range(i+1):
          self.K[i,j]=kernel(X[i],X[j],self.kernel_name)
          self.K[j,i]=self.K[i,j]

    self.alpha=scipy.optimize.minimize(self.logistic_regression, self.alpha)['x']

  def valid(self,X,Y):
    pred=self.predict(X,val=True)
    acc=0
    for i in range(len(Y)):
      if(pred[i]==Y[i]):
        acc=acc+1
    acc=acc/len(Y)
    return acc
  def predict(self,X,val=False):
    nb=X.shape[0]

    result=[]
    for i in range(nb):
      pred=0
      for j in range(len(self.alpha)):

        pred=pred+(self.kernel(self.X_train[j],X[i]) *self.alpha[j])

      if (pred>=0):
        result.append(1)
      else:
        if(val):
          result.append(-1)
        else:
          result.append(0)


        
    return result

      
      

from cvxopt import matrix, solvers
class SVM():
  def __init__(self,C,kernel_name="Linear",substring_size=6,substring_all=True,degree=2,gamma=0.1):
    self.gamma=gamma
    self.kernel_name=kernel_name
    self.substring_size=substring_size
    self.degree=degree
    self.substring_all=substring_all

    self.C=C
    
    

    


  def train (self,X,y):
    DNA_chars=["A","T","C","G"]
    self.DNA_seq=[]
    self.X_train=X
    self.y=y
    self.alpha=np.zeros(y.shape)
    n=len(y)
    self.K=np.zeros((n,n))
    if (self.kernel_name=="Linear"):
      self.K=(X@X.T+1)**2
    else:
      for i in range(n):
        for j in range(i+1):
          self.K[i,j]=kernel(X[i],X[j],kernel_name=self.kernel_name,substring_size=self.substring_size,substring_all=self.substring_all,degree=self.degree,gamma=self.gamma)
          self.K[j,i]=self.K[i,j]

    q=-self.y.reshape((-1,))
    n=len(self.y)
    G =np.vstack((np.diag(self.y), -np.diag(self.y)))
    h =np.vstack((np.ones((n, 1))*self.C, np.zeros((n, 1)))).reshape((-1,))
    sol=solvers.qp(matrix(self.K),matrix(q) , matrix(G),matrix(h))
    self.alpha=sol['x']
    

  def valid(self,X,Y):
    pred=self.predict(X,val=True)
    acc=0
    for i in range(len(Y)):
      if(pred[i]==Y[i]):
        acc=acc+1
    acc=acc/len(Y)
    return acc
        

      



      



  def predict(self,X,val=False):
    nb=X.shape[0]

    result=[]
    for i in range(nb):
      pred=0
      for j in range(len(self.alpha)):

        pred=pred+(kernel(self.X_train[j],X[i],kernel_name=self.kernel_name,degree=self.degree,gamma=self.gamma) *self.alpha[j])

      if (pred>=0):
        result.append(1)
      else:
        if(val):
          result.append(-1)
        else:
          result.append(0)


        
    return result
