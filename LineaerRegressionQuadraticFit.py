# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 14:45:39 2018

@author: Briti
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def linearRegression(x1,x2,x3,x4,x12,x22,x32,x42,y_train,
                     tx1,tx2,tx3,tx4,tx12,tx22,tx32,tx42,y_test):
    #initialise x0 and coefficients
    samples = len(y_train)
    alpha=0.01
    c0=np.zeros(samples)
    c1=np.zeros(samples)
    c2=np.zeros(samples)
    c3=np.zeros(samples)
    c4=np.zeros(samples)
    c5=np.zeros(samples)
    c6=np.zeros(samples)
    c7=np.zeros(samples)
    c8=np.zeros(samples)
    #Run gradient descent without regularization on 500 iterations
    x0=np.ones(samples)  
    
    c0,c1,c2,c3,c4,c4,c6,c7,c8=gradient_descent_without_regularization(x0,x1,x2,x3,
                                              x4,x12,x22,x32,x42,
                                              c0,c1,c2,c3,c4,c5,c6,c7,c8,y_train,samples,alpha,500)
    y_pred=c0[0:4322]*x0[0:4322]+c1[0:4322]*tx1+c2[0:4322]*tx2+c3[0:4322]*tx3+c4[0:4322]*tx4+c5[0:4322]*tx12+c6[0:4322]*tx22+c7[0:4322]*tx32+c8[0:4322]*tx42
        
        
    print('C0'),
    print(c0[0]),
    print('C1'),
    print(c1[0]),
    print('C2'),
    print(c2[0]),
    print('C3'),
    print(c3[0]),
    print('c4'),
    print(c4[0])
    print('C5'),
    print(c5[0]),
    print('C6'),
    print(c6[0]),
    print('C7'),
    print(c7[0]),
    print('C8'),
    print(c8[0]),
    print('---pred--')
    print(y_pred)
    print('---act--')
    print(y_test)
    rm=rmse(y_test,y_pred)
    print(rm) 
    
#function to calculate rmse   
def rmse(y, y_pred):
    rmse = np.sqrt(sum((y - y_pred) ** 2) / len(y))
    return rmse

#function to calculate matrix sum    
def sum(Y):
    sumy=0
    for i in range(len(Y)):
        sumy+=Y[i]
    return sumy
    
#gradient descent function
def gradient_descent_without_regularization(x0,x1,x2,x3,x4,x12,x22,x32,x42,
                                            c0,c1,c2,c3,c4,c5,c6,c7,c8,y,samples,alpha,iterations): 
    for i in range(iterations):
        #Hypothesis
        h=(x0*c0)+(x1*c1)+(x2*c2)+(x3*c3)+(x4*c4)+(x12*c5)+(x22*c6)+(x32*c7)+(x42*c8)
        diff=h-y
        #print (h)
        #Update the coefficients
        c0=c0-(alpha/samples)*(sum(x0*diff))
        c1=c1-(alpha/samples)*(sum(x1*diff))
        c2=c2-(alpha/samples)*(sum(x2*diff))
        c3=c3-(alpha/samples)*(sum(x3*diff))
        c4=c4-(alpha/samples)*(sum(x4*diff))
        c5=c5-(alpha/samples)*(sum(x12*diff))
        c6=c6-(alpha/samples)*(sum(x22*diff))
        c7=c7-(alpha/samples)*(sum(x32*diff))
        c8=c8-(alpha/samples)*(sum(x42*diff))
    return c0,c1,c2,c3,c4,c5,c6,c7,c8

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
    train_sqft2 = minmax(train_sqft**2)
    train_floors2 = minmax(train_floors**2)
    train_bedroom2 = minmax(train_bedroom**2)
    train_bathroom2 = minmax(train_bathroom**2)
    test_sqft = minmax(test_sqft)
    test_floors = minmax(test_floors)
    test_bedroom = minmax(test_bedroom)
    test_bathroom = minmax(test_bathroom)
    test_sqft2 = minmax(test_sqft**2)
    test_floors2 = minmax(test_floors**2)
    test_bedroom2 = minmax(test_bedroom**2)
    test_bathroom2 = minmax(test_bathroom**2)
    linearRegression(train_sqft,train_floors,train_bedroom,train_bathroom,train_sqft2,
                     train_floors2,train_bedroom2,train_bathroom2,y_train,test_sqft,
                     test_floors,test_bedroom,test_bathroom,test_sqft2,test_floors2,
                     test_bedroom2,test_bathroom2,y_test)