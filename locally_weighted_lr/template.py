import numpy as np
import pandas as pd
# hypothesis function

def weight(x, X, tau):
    distances = np.linalg.norm(X - x, axis=1) ** 2 / (2 * tau**2)
    weights = np.exp(-distances)
    return weights

def standardize_matrix(matrix):
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.values  # Convert DataFrame to NumPy array

    num_columns = matrix.shape[1]
    num_columns = matrix.shape[1]

    for i in range(num_columns):
        mean_val = np.mean(matrix[:, i])
        std_dev_val = np.std(matrix[:, i])
        matrix[:, i] = (matrix[:, i] - mean_val) / std_dev_val

    return matrix
def hyp(theta,x):
    temp=np.matmul(x,theta) # the matmal multiplies the x(m,n) by theta(n,1)
    return 1/(1+np.exp(-temp))
# The output of hyp(X, theta) is a column vector containing the sigmoid values
#  for each example in the input feature matrix 

    


#cost finding function, i.e. J(theta)

# def findCost(y, x, X, theta, tau):
#     cost=X*(weight(x,X,tau).reshape(X[0],1)*(y-hyp(theta,x)))-1e-2*theta
#     return cost

#finding gradient of ll

def deltall(y,x,X,theta,tau):

    z=np.multiply((weight(x,X,tau)).reshape(X.shape[0],1),(y-hyp(theta,X)))
    grad= np.matmul(np.transpose(X),z)-1e-10*(theta)
    return grad
def hessian(x,X,theta,tau):
    hessian = -np.matmul(np.matmul(np.transpose(X), np.diag(weight(x,X,tau)*hyp(theta,x)*(1-hyp(theta,x)))) , X) - 1e-10*np.eye(X.shape[1])    # (I*(hyp(theta,X).T*(I-hyp(theta,X)))
    return hessian





#Gradient descent function
# We shall minimmize the loss function 

def GradDescent(y,x,X,theta,tau):

    g=np.ones((X.shape[0],1))
    while (np.linalg.norm(g)>1e-6):
        
        
        #breaking condition defined
        g=deltall(y,x,X,theta,tau)

        #updating theta

        theta = theta - np.linalg.solve(hessian(x,X,theta,tau), g)
        
    return theta


