# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 18:40:31 2018

@author: Briti
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def linearRegression(x1,x2,x3,x4,y,x1_test,x2_test,x3_test,x4_test,y_test):
    #initialise x0 and coefficients
    samples = len(x1)
    x0=np.ones(samples)
    alpha=0.05
    c0=np.zeros(samples)
    c1=np.zeros(samples)
    c2=np.zeros(samples)
    c3=np.zeros(samples)
    c4=np.zeros(samples)
    #Calculate initial cost fumction
    cost_function(x0,x1,x2,x3,x4,y,c0,c1,c2,c3,c4,samples)
    #Run gradient descent without regularization on 500 iterations
    c0,c1,c2,c3,c4=gradient_descent_without_regularization(x0,x1,x2,x3,x4,y,c0,c1,c2,c3,c4,samples,alpha,500)
    y_pred=x0[0:4322]*(c0[0:4322])+x1_test*(c1[0:4322])+x2_test*(c2[0:4322])+x3_test*(c3[0:4322])+x4_test*(c4[0:4322])
    
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
    print('---pred--')
    print(y_pred)
    print('---act--')
    print(y_test)
    rm=rmse(y_test,y_pred)
    print(rm)
    
def cost_function(x0,x1,x2,x3,x4,y,c0,c1,c2,c3,c4,samples):
    #Calculate the cost function
    h=x0*c0+x1*c1+x2*c2+x3*c3+x4*c4
    J=np.sum((h-y)**2)/(2*samples)
    print(J)
    
def rmse(y, y_pred):
    rmse = np.sqrt(sum((y - y_pred) ** 2) / len(y))
    return rmse
    
def gradient_descent_without_regularization(x0,x1,x2,x3,x4,y,c0,c1,c2,c3,c4,samples,alpha,iterations): 
    for i in range(iterations):
        #Hypothesis
        #print (h)
        #Update the coefficients
        c0=c0-(alpha/samples)*(np.sum(x0))
        c1=c1-(alpha/samples)*(np.sum(x1))
        c2=c2-(alpha/samples)*(np.sum(x2))
        c3=c3-(alpha/samples)*(np.sum(x3))
        c4=c4-(alpha/samples)*(np.sum(x4))
        #cost_function(x0,x1,x2,x3,x4,y,c0,c1,c2,c3,c4,samples)
    return c0,c1,c2,c3,c4

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
    
    linearRegression(train_sqft,train_floors,train_bedroom,train_bathroom,y_train,test_sqft,test_floors,test_bedroom,test_bathroom,y_test)