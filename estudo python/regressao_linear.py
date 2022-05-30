# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:33:11 2022

@author: escneto

From: https://realpython.com/linear-regression-in-python/?utm_source=notification_summary&utm_medium=email&utm_campaign=2022-05-16
"""

"""
Simple Linear Regression With scikit-learn
You’ll start with the simplest case, which is simple linear regression. 
There are five basic steps when you’re implementing linear regression:

1. Import the packages and classes that you need.
2. Provide data to work with, and eventually do appropriate transformations.
3. Create a regression model and fit it with existing data.
4. Check the results of model fitting to know whether the model is satisfactory.
5. Apply the model for predictions.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([5,15,25,35,45,55]).reshape((-1,1))
y = np.array([5,20,14,32,22,38])


model = LinearRegression().fit(x,y)

r_sq = model.score(x,y)

print("--------------------------------------------")
print(f"R² (coeficiente de determinação): {r_sq}")
print(f"intercept (interceptador com eixo vertical): {model.intercept_}")
print(f"slope (coeficiente angular): {model.coef_}")
print("--------------------------------------------")

# Note: In scikit-learn, by convention, a trailing underscore indicates that an attribute is estimated. 
# In this example, .intercept_ and .coef_ are estimated values.

y_pred = model.predict(x)
print("--------------------------------------------")
print(f"predicted response:\n{y_pred}")
print("--------------------------------------------")

y_pred_func = model.intercept_ + model.coef_ * x
print("--------------------------------------------")
print(f"Using function: {model.intercept_} + ({model.coef_} * x)")
print(f"predicted response:\n{y_pred_func.flatten()}")
print("--------------------------------------------")

# In practice, regression models are often applied for forecasts. 
# This means that you can use fitted models to calculate the outputs based on new inputs:

x_new = np.arange(5).reshape((-1,1))

y_new = model.predict(x_new)
print("--------------------------------------------")
print(f"New inputs: {x_new.flatten()}")
print(f"Outputs: {y_new}")
print("--------------------------------------------")



"""
Multiple Linear Regression With scikit-learn
You can implement multiple linear regression following the same steps as you 
would for simple regression. The main difference is that your x array will now 
have two or more columns.
"""

x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]

x,y = np.array(x),np.array(y)

print("--------------------------------------------")
print(f"New x: {x}")
print(f"New y: {y}")
print("--------------------------------------------")

model = LinearRegression().fit(x,y)

r_sq = model.score(x, y)
print("--------------------------------------------")
print(f"R² (coeficiente de determinação): {r_sq}")
print(f"intercept (interceptador com eixo vertical): {model.intercept_}")
print(f"slope (coeficiente angular): {model.coef_}")
print("--------------------------------------------")

y_pred = model.predict(x)
print("--------------------------------------------")
print(f"predicted response:\n{y_pred}")
print("--------------------------------------------")

y_pred_func = model.intercept_ + np.sum(model.coef_ * x, axis = 1)
print("--------------------------------------------")
print(f"Using function: {model.intercept_} + ({model.coef_[0]} * x1) + ({model.coef_[1]} * x2)")
print(f"predicted response:\n{y_pred_func.flatten()}")
print("--------------------------------------------")

x_new = np.arange(10).reshape((-1,2))

y_new = model.predict(x_new)
print("--------------------------------------------")
print(f"New inputs:\n{x_new}")
print(f"Outputs: {y_new}")
print("--------------------------------------------")



"""
Polynomial Regression With scikit-learn
Implementing polynomial regression with scikit-learn is very similar to linear 
regression. There’s only one extra step: you need to transform the array of 
inputs to include nonlinear terms such as 𝑥².
"""

from sklearn.preprocessing import PolynomialFeatures

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])


#transformer = PolynomialFeatures(degree=2,include_bias=True)
#transformer.fit(x)
#x_ = transformer.fit(x).transform(x)
x_ = PolynomialFeatures(degree=2,include_bias=False).fit_transform(x)

model = LinearRegression().fit(x_,y)

r_sq = model.score(x_, y)
print("--------------------------------------------")
print(f"R² (coeficiente de determinação): {r_sq}")
print(f"intercept (interceptador com eixo vertical): {model.intercept_}")
print(f"slope (coeficiente angular): {model.coef_}")
print("--------------------------------------------")


y_pred = model.predict(x_)
print("--------------------------------------------")
print(f"predicted response:\n{y_pred}")
print("--------------------------------------------")



"""
Advanced Linear Regression With statsmodels
You can implement linear regression in Python by using the package statsmodels as well. Typically, this is desirable when you need more detailed results.

The procedure is similar to that of scikit-learn.
"""

import statsmodels.api as sm

x = [
  [0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]
]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)


x = sm.add_constant(x)

model = sm.OLS(y,x)

results = model.fit()

print("--------------------------------------------")
print(results.summary())
print("--------------------------------------------")
print(f"predicted response:\n{results.fittedvalues}")
print(f"predicted response:\n{results.predict(x)}")


