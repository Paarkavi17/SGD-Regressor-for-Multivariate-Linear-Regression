# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Load California housing data, select features and targets, and split into training and testing sets.
2. Scale both X (features) and Y (targets) using StandardScaler. 
3. Use SGDRegressor wrapped in MultiOutputRegressor to train on the scaled training data. 
4. Predict on test data, inverse transform the results, and calculate the mean squared error.
## Program:
```
Developed by:PAARKAVI A
Register Number:25012275

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

X = np.array([
    [1000, 2],
    [1200, 3],
    [1500, 3],
    [1800, 4],
    [2000, 4],
    [2200, 5]
])

y = np.array([
    [30, 3],
    [40, 4],
    [50, 4],
    [65, 5],
    [75, 5],
    [90, 6]
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sgd = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='invscaling', eta0=0.01)
model = MultiOutputRegressor(sgd)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Actual Values:\n", y_test)
print("\nPredicted Values:\n", predictions)
```
## Output:

<img width="436" height="247" alt="Screenshot 2026-02-12 191241" src="https://github.com/user-attachments/assets/aac6ca32-a373-47d0-9817-e55cdfa9cb96" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
