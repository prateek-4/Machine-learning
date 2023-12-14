import numpy as np
import pandas as pd
# hypothesis function
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

def findCost(y,X,theta):
    ll = np.sum(y * np.log(hyp(theta,X))  + (1 - y) * np.log(1 - hyp(theta,X)))
    return -ll # the cost function is negative of ll 


#finding gradient of ll

def deltall(y,X,theta):
    return np.matmul(np.transpose(X), y-hyp(theta,X))

def hessian(y,X,theta):
    diag = np.identity(np.shape(X)[0]) * np.matmul(np.transpose(hyp(theta,X)),1-hyp(theta,X))
    hessian = np.matmul(np.transpose(X),np.matmul(diag,X))
        # (I*(hyp(theta,X).T*(I-hyp(theta,X)))
    return -hessian




#defining a learning rate

learningRate = 0.1


#Gradient descent function
# We shall minimmize the loss function 

def GradDescent(y,X,theta):

    #initializing a starting cost
    prevcost = 1e5
    i=0
    while True and i<=200:
        i+=1

        cost = (findCost(y,X,theta))
        

        #breaking condition defined

        if (abs((cost-prevcost))<1e-10):
            break

        #updating theta

        theta = theta - np.matmul(np.linalg.inv(hessian(y,X,theta)),deltall(y,X,theta))
        
        prevcost = cost

    return theta


