# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 13:42:06 2018

@author: t-blu
"""
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
import math
#Are we doing adaptive!?
Adaptive = False
lr = 1 #Setting the Learning rate, should be set to 1 when doing dynamic shrinkage
#Data Prep
train = pd.read_csv(FILEPATH,  delimiter=',')
test = pd.read_csv(FILEPATH,  delimiter=',')
y = train['y']
x = train
x.drop('y', axis=1, inplace=True)
y_test = test['y']
x_test = test
x_test.drop('y', axis=1, inplace=True)
xi = x # initialization of input
yi = y # initialization of target
# x,y --> use where no need to change original y
#Creating a bunch of initial values that we need
ei = 0 # initialization of error
predf = 0 # initial prediction 0
count = 1
adap = 1 #This constant can be used to help with the convergence of the dynamic shrinkage (if needed).  It simply multiplies the calculated learning rate.
testme = 0
rmse_list_adap = [] #Creating the rmse data set
sumei = 0
learning_rates = [] #Creating the learning rates data set
for i in range(1000): # loop will make n trees (n_estimators). 
    tree = DecisionTreeRegressor(max_depth = 2) 
    tree.fit(xi,yi)
    predi =  tree.predict(xi)
    predicted =  tree.predict(x_test)
    #The first model is just the basic tree, subsequent models will be based on the residuals
    if count < 2:
        predf = predi
    else:
        predf = predf + adap*lr*predi  # final prediction will be previous prediction value + new prediction of residual
    if count < 2:
        testme = predicted
    else:
        testme = testme + adap*lr*predicted
    ei = y - predf  # needed originl y here as residual always from original y    
    yi = ei # update yi as residual to reloop

    if Adaptive is True:
        if count > 1: #The "dynamic" Part
            lr = math.sqrt(((sumei - np.sum(ei**2))/sumei)) #The square root of the percentage change in the training error seemed to work well.
            #It is bounded by 0 and 1 (the training error will never increase)
        
    sumei = np.sum(ei**2) # Calculate the sum of squared errors
    count = count + 1
    learning_rates.append(lr)
    rmse = math.sqrt(np.mean((y_test - testme)**2)) #Calculate RMSE
    rmse_list_adap.append(rmse)



#List of the RMSE's for each iteration
rmse_list_adap = pd.DataFrame(rmse_list_adap)
#List of the learning rates for each iteration
learning_rates = pd.DataFrame(learning_rates)







