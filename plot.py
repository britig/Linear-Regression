# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 22:45:45 2018

@author: Briti
"""

import matplotlib.pyplot as plt
import numpy as np

#Lambda vs rmse plot
y=np.array([0.1,0.5,1,10,100,500,600,1000,5000,10000])
x=np.array([350131,350127,350123,350052,349436,348269,348262,348920,371022,404564])
plt.plot(x,y)
plt.ylabel('lambda')
plt.xlabel('rmse')
plt.show()


#Script for generating learning rate vs rmse
alpha=0.01
alpha_history=[0]*100
rmse_history=[0]*100
it=0
while(alpha<=1):
	c0,c1,c2,c3,c4,ch=gradient_descent_without_regularization(x0,x1,x2,x3,x4,y,
                                                           c0,c1,c2,c3,c4,samples,alpha,500,
                                                           x1_test,x2_test,x3_test,x4_test,y_test)
	y_pred=x0[0:4322]*(c0[0:4322])+x1_test*(c1[0:4322])+x2_test*(c2[0:4322])+x3_test*(c3[0:4322])+x4_test*(c4[0:4322])
	rm=rmse(y_test,y_pred)
	alpha_history[it]=alpha
	rmse_history[it]=rm
	alpha+=0.01
	it+=1
	
    plt.plot(rmse_history,alpha_history)
    plt.ylabel('alpha')
    plt.xlabel('rmse')
    plt.show()