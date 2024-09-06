# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: 
RegisterNumber:  
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(x1,y,learning_rate=0.01,num_iters=1000):
    
    # add a column of ones to x for the intercept term
    x = np.c_[np.ones(len(x1)),x1]
    
    # initialize theta with zeros
    theta = np.zeros(x.shape[1]).reshape(-1,1)
    
    # perform gradient descent
    for _ in range(num_iters):
        # calculate predictions
        prediction = (x).dot(theta).reshape(-1,1)
        
        #calculate errors
        errors = (prediction - y).reshape(-1,1)
        
        # update the theta using gradient descent
        theta -= learning_rate * (1/ len(x1)) * x.T.dot(errors)
        
        return theta
```

## Output:
![linear regression using gradient descent](sam.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
