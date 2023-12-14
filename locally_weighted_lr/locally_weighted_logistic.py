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
from template import standardize_matrix
x = np.array(pd.read_csv('C:/Users/Darshan Mourya/Music/ML/winter/locally_weighted_lr/x.csv', header=None))
y = np.array(pd.read_csv('C:/Users/Darshan Mourya/Music/ML/winter/locally_weighted_lr/y.csv', header=None))
#x=standardize_matrix(x)
x = np.hstack([np.ones((np.shape(x)[0],1)),x])
theta = np.zeros((x.shape[1],1))

from template import GradDescent
from template import hyp










type1 = x[y.ravel() == 1][:, 1:]
type2 = x[y.ravel() == 0][:, 1:]



# Assuming you have already trained the model and obtained the theta values
# ...

# Generate test data points on a meshgrid
X_test = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
X_test = np.column_stack((X_test[0].ravel(), X_test[1].ravel()))
#X_test=standardize_matrix(X_test)
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
print(X_test.shape)
# Add a column of ones for the intercept term

y_test_prob = []
for pt in X_test:
    theta=GradDescent(y,pt,x,theta,0.01)
    y_test_prob.append(hyp(theta,pt))
# Use the hypothesis function to get probabilities
y_test_prob=np.array(y_test_prob)
# Apply a threshold (e.g., 0.5) to get binary predictions
y_test_pred = (y_test_prob >= 0.5).astype(int)

type_1 = X_test[y_test_pred.ravel() == 1][:, 1:]
type_2 = X_test[y_test_pred.ravel() == 0][:, 1:]


# Plot the decision boundary and test data points
plt.figure()
# Set labels and title
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Logistic Regression Decision Boundary")
plt.scatter(type1[:, 0], type1[:, 1], label="Y = 1", color='black', marker='.')
plt.scatter(type2[:, 0], type2[:, 1], label="Y = 0", color='black', marker='+')
plt.scatter(type_1[:, 0], type_1[:, 1], label="Y = 1", color='blue', marker='.')
plt.scatter(type_2[:, 0], type_2[:, 1], label="Y = 0", color='pink', marker='+')


plt.show()




# adding a column of 1 to x for the intercept term

# from template import GradDescent
# from template import standardize_matrix

# x=standardize_matrix(x)
# x = np.hstack([np.ones((np.shape(x)[0],1)),x])
# theta = np.zeros((x.shape[1],1))
# theta = GradDescent(y,x,theta)

# print(theta)

# xdata = np.arange(-5,5,0.1)
# ydata = -(theta[0] + theta[1]*xdata)/theta[2]
# #plotting the datset

# type1 = x[y.ravel() == 1][:, 1:]
# type2 = x[y.ravel() == 0][:, 1:]


# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.title("Dataset")
# plt.scatter(type1[:,0],type1[:,1],label="Y = 1",color='red',marker='.')
# plt.scatter(type2[:,0],type2[:,1],label="Y = 0",color='green',marker='+')


# plt.plot(xdata,ydata,label="Decision Boundary",color="blue")
# plt.show()
