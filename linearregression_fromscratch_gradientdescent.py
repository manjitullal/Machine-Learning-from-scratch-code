# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 17:33:25 2020

@author: Manjit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12.0,9.0)

#preprocessing the data 
dataset = pd.read_csv('dataset.csv')
#print(dataset.head(5))

x = dataset.iloc[:,2]
y = dataset.iloc[:,3]

#print(y)


#plt.show()

#building the model
#y = mx + c
#here we take the starting values as 0 and 0 for m and c

m = 0
c = 0

L = 0.0001
epochs = 1000

#Here L is learning rate and epochs is the number of iterations

n = len(x)

for i in range(epochs):
    y_pred = m*x + c
    d_m = (-2/n)*sum(x*(y - y_pred))
    d_c = (-2/n)*sum(y - y_pred)
    print(d_m,d_c)
    m = m - L*d_m
    c = c - L*d_c
    
#print(m,c)

y_pred = m*x + c
plt.scatter(x,y)
plt.plot([min(x),max(x)],[min(y_pred),max(y_pred)],color='red')
plt.show
