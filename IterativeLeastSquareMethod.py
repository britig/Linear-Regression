# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 19:05:37 2018

@author: Briti

Note: The method described in the pdf gives optimal result in a single iteration

"""

import pandas as pd
import numpy as np
import matplotlib as plt
from numpy.linalg import inv

def linearRegression(x1,x2,x3,x4,y,x1_test,x2_test,x3_test,x4_test,y_test):
    #initialise x0 and coefficients
    samples = len(y)
    x0=np.ones(samples)
    c0=np.zeros(samples)
    c1=np.zeros(samples)
    c2=np.zeros(samples)
    c3=np.zeros(samples)
    c4=np.zeros(samples)
    #Run iterative reweighted least square method
    X_train=np.array([x0,x1,x2,x3,x4])
    #Calculate Hessian Matrix
    Hinv=inv(X_train.dot(X_train.T))
    c0,c1,c2,c3,c4=iterative_reweighted_least_square(Hinv,x0,x1,x2,x3,x4,y,c0,c1,c2,c3,c4)
    
    y_pred=x0[0:4322]*(c0[0:4322])+x1_test*(c1[0:4322])+x2_test*(c2[0:4322])+x3_test*(c3[0:4322])+x4_test*(c4[0:4322])
    print('---pred--')
    print(y_pred)
    print('---act--')
    print(y_test)
    rm=rmse(y_test,y_pred)
    print(rm)
    
#function to calculate RMSE   
def rmse(y, y_pred):
    rmse = np.sqrt(sum((y - y_pred) ** 2) / len(y))
    return rmse

#function to calculate sum
def sumM(Y):
    sumy=0
    for i in range(len(Y)):
        sumy+=Y[i]
    return sumy
    
def iterative_reweighted_least_square(Hinv,x0,x1,x2,x3,x4,y,c0,c1,c2,c3,c4):
    
    h=(x0*c0)+(x1*c1)+(x2*c2)+(x3*c3)+(x4*c4)
    diff = h-y
    deltaGrad = np.array([sumM(x0*diff),sumM(x1*diff),sumM(x2*diff),sumM(x3*diff),sumM(x4*diff)])
    C=np.array([0,0,0,0,0])
    C=C-Hinv.dot(deltaGrad)
    print(C)
    #print(H)
    c0.fill(C[0])
    c1.fill(C[1])
    c2.fill(C[2])
    c3.fill(C[3])
    c4.fill(C[4])
    
    return c0,c1,c2,c3,c4

#function for minmax scaling
def minmax(X):
    X = (X-np.amin(X))/((np.amax(X)-np.amin(X)))
    return X
    
if __name__=="__main__":
    #Read the CSV file
    df = pd.read_csv('C:/Users/Briti/kc_house_data.csv')
    #80% split to train data and 20% test data
    train_sqft=(df.sqft[0:17291].values)
    test_sqft=(df.sqft[17291:].values)
    train_floors=(df.floors[0:17291].values)
    test_floors=(df.floors[17291:].values)
    train_bedroom=(df.bedrooms[0:17291].values)
    test_bedroom=(df.bedrooms[17291:].values)
    train_bathroom=(df.bathrooms[0:17291].values)
    test_bathroom=(df.bathrooms[17291:].values)
    y_train=df.price[0:17291].values
    y_test=df.price[17291:].values
    
    #Scale the data
    train_sqft = minmax(train_sqft)
    train_floors = minmax(train_floors)
    train_bedroom = minmax(train_bedroom)
    train_bathroom = minmax(train_bathroom)
    test_sqft = minmax(test_sqft)
    test_floors = minmax(test_floors)
    test_bedroom = minmax(test_bedroom)
    test_bathroom = minmax(test_bathroom)
    
    linearRegression(train_sqft,train_floors,train_bedroom,
                     train_bathroom,y_train,test_sqft,test_floors,test_bedroom,test_bathroom,y_test)