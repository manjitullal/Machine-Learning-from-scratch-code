# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:48:37 2020

@author: Manjit
"""
#using the least square method
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data.csv')
#print(dataset.shape)
#print(dataset.head())

#x and y value of the input
#x = dataset['Head Size(cm^3)'].values
#y = dataset['Brain Weight(grams)'].values
x = dataset.iloc[:,0]
y = dataset.iloc[:,1]

#mean of the values
x_mean = np.mean(x)
y_mean = np.mean(y)

#print(y[0:5])
n =  len(x)
numerator = 0
denominator = 0

for i in range(n):
    numerator += (x[i] - x_mean)*(y[i] - y_mean)
    denominator += (x[i] - x_mean)**2
    
b1 = numerator/denominator
b0 = y_mean - (b1*x_mean)

print(b0,b1)

#plotting the values
x_max = np.max(x) 
x_min = np.min(x) 

x1 = np.linspace(x_min,x_max,1000) 
y1 = b0 + b1*x1

plt.plot(x1,y1,color='blue',label='Linear Regression')
plt.scatter(x,y,color='red',label='data point')

plt.xlabel('Head Size')
plt.ylabel('Brain weight')

plt.legend()
plt.show()

#the method we have used to find the co-efficients is called the least mean squared method
#we will find the accuracy of the model using the root mean square method

rmse = 0
for i in range(n):
    y_pred = b0 + b1*x[i]
    rmse += (y[i] - y_pred)**2

rmse = np.sqrt(rmse/n)
print(rmse)

sumofsquares = 0
sumofresiduals = 0

for i in range(n):
    y_pred = b0 + b1*x[i]
    sumofsquares += (y[i] - y_mean)**2
    sumofresiduals += (y[i] - y_pred)**2

score =  1 - (sumofresiduals/sumofsquares)

print(score)

