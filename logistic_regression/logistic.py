# I like to think of the likelihood function as 
#“the likelihood that our model will correctly predict any given y
#  value, given its corresponding feature vector x"

# Newton’s Method is an iterative equation solver:
# it is an algorithm to find the roots of a polynomial function.

# update rule for finding zeros of f(x^n) is :
#          x^n+1=x^n−f(x^n)∗∇f(x^n)−1
# We can substitute f(x^n)
#  with the gradient, ∇ℓ(θ) , leaving us with:

#       θn+1=θn+H−1ℓ(θ^)∇ℓ(θ)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
x = np.array(pd.read_csv('C:/Users/Darshan Mourya/Music/ML/winter/logistic_regression/logisticX.csv',header=None))
y = np.array(pd.read_csv('C:/Users/Darshan Mourya/Music/ML/winter/logistic_regression/logisticY.csv',header=None))
# adding a column of 1 to x for the intercept term

from template import GradDescent
from template import standardize_matrix

x=standardize_matrix(x)
x = np.hstack([np.ones((np.shape(x)[0],1)),x])
theta = np.zeros((x.shape[1],1))
theta = GradDescent(y,x,theta)

print(theta.shape)

xdata = np.arange(-5,5,0.1)
ydata = -(theta[0] + theta[1]*xdata)/theta[2]
#plotting the datset

type1 = x[y.ravel() == 1][:, 1:]
type2 = x[y.ravel() == 0][:, 1:]


plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Dataset")
plt.scatter(type1[:,0],type1[:,1],label="Y = 1",color='red',marker='.')
plt.scatter(type2[:,0],type2[:,1],label="Y = 0",color='green',marker='+')


plt.plot(xdata,ydata,label="Decision Boundary",color="blue")
plt.show()
