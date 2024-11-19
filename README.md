# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a company using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use numpy, pandas, and StandardScaler for computations and preprocessing.
2. Read the dataset and extract features (x) and target (y) variables.
3. Normalize features and target using StandardScaler.
4. Add a bias column to x and initialize parameters (theta) as zeros.
5. Update theta iteratively using gradient descent.
6. Scale new data, predict using theta, and inverse-transform the result.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: VINOTH M P
RegisterNumber:  212223240182
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

df=pd.read_csv('50_startups.csv')

df.head()

# assuming the last column is your targer variable 'y' and the preceding colums are your feat
x = (df.iloc[1:,:-2].values)
print("the value of x is\n",x)

x1=x.astype(float)
scaler = StandardScaler()
y = (df.iloc[1:, -1].values).reshape(-1,1)
print("\nthe value of y is\n",y)

x1_scaled=scaler.fit_transform(x1)
y1_scaled=scaler.fit_transform(y)
print("\nthe value of x1_scaled\n",x1_scaled,"\nand the value of y1_scaled is\n",y1_scaled)


#  learn model parameters
theta= linear_regression(x1_scaled,y1_scaled)

# predict target value for the new data point
nd=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
print("new data value\n",nd)


ns=scaler.fit_transform(nd)
print("\nnew scaled value\n",ns)

prediction = np.dot(np.append(1,ns),theta)
prediction = prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"\npredicted value: \n{pre}")

```

## Output:

![Screenshot 2024-09-05 142550](https://github.com/user-attachments/assets/94e1bd71-cc0a-4fd7-98e7-c5de7793753d)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
