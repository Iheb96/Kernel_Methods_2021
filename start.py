# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 23:22:14 2021

@author: Iheb96
"""

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("kernel")

args = parser.parse_args()
import csv

from classif import SVM,logistic_regression
import pandas as pd
import numpy as np
import warnings
from time import time
warnings.filterwarnings('ignore')


def read_mat(csvfile):
  df=pd.read_csv(csvfile,header=None)  
  n,m=df.shape
  X=[]
  for i in range(n):
    X.append(df.iloc[i,0].split())
  X=np.array(X, dtype='float64')
  return X
csvfile="./data/Xtr0_mat100.csv"
X0_mat=read_mat(csvfile)
csvfile="./data/Xtr1_mat100.csv"
X1_mat=read_mat(csvfile)
csvfile="./data/Xtr2_mat100.csv"
X2_mat=read_mat(csvfile)


def read_dna(csvfile):
  df=pd.read_csv(csvfile)  
  x=df.iloc[:,1]
  x=np.array(x)
  return x
csvfile="./data/Xtr0.csv"
X0=read_dna(csvfile)
csvfile="./data/Xtr1.csv"
X1=read_dna(csvfile)
csvfile="./data/Xtr2.csv"
X2=read_dna(csvfile)

def read_y(csvfile):
  df=pd.read_csv(csvfile)  
  y=df.iloc[:,1]*2-1
  y=np.array(y, dtype='float64')
  return y
csvfile="./data/Ytr0.csv"
y0=read_y(csvfile)
csvfile="./data/Ytr1.csv"
y1=read_y(csvfile)
csvfile="./data/Ytr2.csv"
y2=read_y(csvfile)

def split_data(X,val_perc=0.8):
  if (len(X.shape)==1):
    X_train=X[:int(len(X)*val_perc)]
    X_val=X[int(len(X)*val_perc):]
  else:
    X_train=X[:int(len(X)*val_perc),:]
    X_val=X[int(len(X)*val_perc):,:]
  return X_train,X_val
X0_mat_train,X0_mat_val=split_data(X0_mat)
X1_mat_train,X1_mat_val=split_data(X1_mat)
X2_mat_train,X2_mat_val=split_data(X2_mat)
X0_train,X0_val=split_data(X0)
X1_train,X1_val=split_data(X1)
X2_train,X2_val=split_data(X2)
y0_train,y0_val=split_data(y0)
y1_train,y1_val=split_data(y1)
y2_train,y2_val=split_data(y2)


t0=time()
if(args.kernel=="Linear"):
    svm0=SVM(40,args.kernel)
    svm0.train(X0_mat_train,y0_train)
    
    print("Dataset 0 validation accuracy: ",svm0.valid(X1_mat_val,y1_val))
    svm1=SVM(30,args.kernel)
    svm1.train(X1_mat_train,y1_train)
    print("Dataset 1 validation accuracy: ",svm1.valid(X1_mat_val,y1_val))
    svm2=SVM(20,args.kernel)
    svm2.train(X2_mat_train,y2_train)
    print("Dataset 2 validation accuracy: ",svm2.valid(X2_mat_val,y2_val))
    
elif (args.kernel=="Polynomial"):
    svm0=SVM(40,args.kernel)
    svm0.train(X0_mat_train,y0_train)
    
    print("Dataset 0 validation accuracy: ",svm0.valid(X1_mat_val,y1_val))
    svm1=SVM(30,args.kernel)
    svm1.train(X1_mat_train,y1_train)
    print("Dataset 1 validation accuracy: ",svm1.valid(X1_mat_val,y1_val))
    svm2=SVM(20,args.kernel)
    svm2.train(X2_mat_train,y2_train)
    print("Dataset 2 validation accuracy: ",svm2.valid(X2_mat_val,y2_val))
    
elif (args.kernel=="rbf"):
    svm0=SVM(20,"rbf",gamma=0.5)
    svm0.train(X0_mat_train,y0_train)
    
    print("Dataset 0 validation accuracy: ",svm0.valid(X1_mat_val,y1_val))
    svm1=SVM(20,"rbf",gamma=0.5)
    svm1.train(X1_mat_train,y1_train)
    print("Dataset 1 validation accuracy: ",svm1.valid(X1_mat_val,y1_val))
    svm2=SVM(20,"rbf",gamma=0.5)
    svm2.train(X2_mat_train,y2_train)
    print("Dataset 2 validation accuracy: ",svm2.valid(X2_mat_val,y2_val))
else:

    svm0=SVM(15,"Spectrum",substring_size=12,substring_all=True)
    svm0.train(X0_train,y0_train)
    print("Dataset 0 validation accuracy: ",svm0.valid(X0_val,y0_val))
    svm1=SVM(15,"Spectrum",substring_size=12,substring_all=True)
    svm1.train(X1_train,y1_train)
    print("Dataset 1 validation accuracy: ",svm1.valid(X1_val,y1_val))
    svm2=SVM(15,"Spectrum",substring_size=12,substring_all=True)
    svm2.train(X2_train,y2_train)
    print("Dataset 2 validation accuracy: ",svm2.valid(X2_val,y2_val))

print("Training & validation time :",time()-t0)   
if(args.kernel=="rbf" or args.kernel=="Polynomial" or args.kernel=="Linear"):
    csvfile="./data/Xte0_mat100.csv"
    X0_mat_test=read_mat(csvfile)
    res0=svm0.predict(X0_mat_test)
    csvfile="./data/Xte1_mat100.csv"
    X1_mat_test=read_mat(csvfile)
    res1=svm1.predict(X1_mat_test)
    csvfile="./data/Xte2_mat100.csv"
    X2_mat_test=read_mat(csvfile)
    res2=svm2.predict(X2_mat_test)
else:
    csvfile="./data/Xte0.csv"
    X0_test=read_dna(csvfile)
    res0=svm0.predict(X0_test)
    csvfile="./data/Xte1.csv"
    X1_test=read_dna(csvfile)
    res1=svm1.predict(X1_test)
    csvfile="./data/Xte2.csv"
    X2_test=read_dna(csvfile)
    res2=svm2.predict(X2_test)


file=open('prediction.csv', 'w')
myCsv = csv.writer(file)
myCsv.writerow(["Id", "Bound"]),
c=0
for i in res0:
    myCsv.writerow([c, i])
    c=c+1
for i in res1:
    myCsv.writerow([c, i])
    c=c+1
for i in res2:
    myCsv.writerow([c, i])
    c=c+1
file.close()