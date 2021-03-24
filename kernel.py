# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:19:01 2021

@author: Iheb96
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
def kernel(x0,x1,kernel_name,degree=2,gamma=0.5,K=2,substring_size=9,substring_all=True,mistmatch_m=2,beta=0.5,s=100,e=0.2,d=0.2):  
    DNA_chars=["A","T","C","G"]  
    if(kernel_name=="Linear"):
      return (x0.T@x1.T)

    elif (kernel_name=="Polynomial"):
      return (x0.T@x1.T+1)**2

    elif (kernel_name == "rbf"):
        

        return np.exp(-(np.linalg.norm(x1-x0)**2)/(2*gamma**2))
    elif (kernel_name=="Spectrum"):

      substrings_0 = {}
      substrings_1 = {}
      DNA0=str(x0)
      DNA1=str(x1)

      res=0
      for i in range(len(DNA0)):
        for j in range(i+1,min(substring_size+1+i,len(DNA0))):
          key0=DNA0[i:j]
          key1=DNA1[i:j]

          if (key0 not in substrings_0.keys()): 
            substrings_0[key0]= 0
            substrings_1[key0]= 0
          if (key1 not in substrings_1.keys()): 
            substrings_0[key1]= 0
            substrings_1[key1]= 0
          substrings_0[key0]=substrings_0[key0]+1
          substrings_1[key1]=substrings_1[key1]+1
      for key in substrings_0:
          res=res+substrings_0[key]*substrings_1[key]

      return res
    elif (kernel_name=="CKN"):

      DNA0=str(x0)
      DNA1=str(x1)
      encoded0=np.zeros((4,len(DNA0)))
      encoded1=np.zeros((4,len(DNA1)))
      res=0
      for i in range(len(DNA0)):
        for j in range(len(DNA_chars)):
          if (DNA0[i]==DNA_chars[j]):
            encoded0[j,i]=1
      for i in range(len(DNA0)-K):
          res=res+kernel(encoded0[:,i:i+K],encoded1[:,i:i+K],"rbf",gamma=gamma)
      return res

    elif (kernel_name=="LA"):
      DNA0=str(x0)
      DNA1=str(x1)
      n1=len(DNA0)
      n2=len(DNA1)
      M=np.zeros((n1+1,n2+1))
      X=np.zeros((n1+1,n2+1))
      Y=np.zeros((n1+1,n2+1))
      X2=np.zeros((n1+1,n2+1))
      Y2=np.zeros((n1+1,n2+1))
      s=0
      for i in range(1,n1+1):
        for j in range(1,n2+1):
          M[i,j]=np.exp(beta*s)*(1+X[i-1,j-1]+Y[i-1,j-1]+M[i-1,j-1])
          X[i,j]=np.exp(beta*d)*M[i-1,j]+np.exp(beta*e)*X[i-1,j]
          Y[i,j]=np.exp(beta*d)*(M[i,j-1]+X[i,j-1])+np.exp(beta*e)*Y[i,j-1]
          X2[i,j]=M[i-1,j]+X2[i-1,j]
          Y2[i,j]=M[i,j-1]+X2[i,j-1]+Y2[i,j-1]
      res=1+X2[n1,n2]+Y2[n1,n2]+M[n1,n2]
      return res