import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
x=np.array([])
y=np.array([])

x=np.loadtxt('C:/Users/Darshan Mourya/Music/ML/winter/Gaussian Discriminant Analysis/q4x.dat')
y=np.loadtxt('C:/Users/Darshan Mourya/Music/ML/winter/Gaussian Discriminant Analysis/q4y.dat',dtype=str)

y=y.reshape(-1,1)

a=np.zeros(y.shape)

for i in range (y.shape[0]):
    if (y[i][0]=='Alaska'):
        a[i][0]=0
    else:
        a[i][0]=1
y=a

from template import standardize_matrix

# Plotting the data

x=standardize_matrix(x)
# type1 = x[y.ravel() == 1]
# type2 = x[y.ravel() == 0]

# plt.xlabel("x1")
# plt.ylabel("x2")

# plt.scatter(type1[:, 0], type1[:, 1], label="Y = 1", color='blue', marker='.')
# plt.scatter(type2[:, 0], type2[:, 1], label="Y = 0", color='black', marker='+')

# plt.show()

#predicting and testing the model
from template import linear_gda_param

mu0,mu1,sigma,phi=linear_gda_param(x,y)
mu0=np.transpose(mu0)
mu1=np.transpose(mu1)

## NOTE refer to mathematics of this
## using the two probabilty distribution of the classes simplify to get the equation of the decision boundary

## log term
 
term1=np.log(phi/(1-phi))

## constatn term

term2=0.5*(np.matmul(np.transpose(mu0),np.linalg.solve(sigma,mu0)))
term2=term2-0.5*(np.matmul(np.transpose(mu1),np.linalg.solve(sigma,mu1)))

## x_multerm

term3=np.linalg.solve(sigma,mu1-mu0)

##  x_term

x_term=x[:,0]
## y

y_term=-(term1+term2+term3[0]*x_term)/term3[1]
y_term=np.transpose(y_term)
x_term=x_term.reshape(-1,1)

## plotting the decision boundary 

from template import standardize_matrix
x=standardize_matrix(x)
type1 = x[y.ravel() == 1]
type2 = x[y.ravel() == 0]

plt.xlabel("x1")
plt.ylabel("x2")

plt.scatter(type1[:, 0], type1[:, 1], label="Y = 1", color='blue', marker='.')
plt.scatter(type2[:, 0], type2[:, 1], label="Y = 0", color='black', marker='+')
plt.plot(x_term,y_term,color='red')
plt.show()

