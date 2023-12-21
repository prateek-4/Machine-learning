import numpy as np
import pandas as pd
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

# NOTE: for the shapes of each matrix write down on a piece of paper for better understnading
#############################################

def linear_gda_param(x,y):
    phi=np.sum(y)/y.shape[0]
    mu0=np.matmul(np.transpose(x),1-y)/np.sum(1-y)
    mu1=np.matmul(np.transpose(x),y)/np.sum(y)
    mu0=np.transpose(mu0)
    mu1=np.transpose(mu1)
    Z=np.array(x-mu0*(y==0)-mu1*(y==1))

    Z=np.matmul(np.transpose(Z),Z)/y.shape[0]
    #print(Z.shape)
    return mu0,mu1,Z,phi

def gauss(x,sigma,mu):
    cons=1/(((2*np.pi)**(x.shape[1]/2))*(np.linalg.det(sigma)**0.5))
    #print(x.shape)
    term=np.matmul(np.transpose(x-mu),np.linalg.solve(sigma,x-mu))
    return cons*np.exp(-0.5*(term))

def predict(x,X,Y):
    x=np.array(x)
    #print(x)
    mu0,mu1,sigma,phi=linear_gda_param(X,Y)
    y_pred=np.zeros((x.shape[0],1))
    for i in range(x.shape[0]):
        temp=x[i,:]
        print(temp.shape)
        p0=gauss(temp,sigma,mu0)*(1-phi)
        p1=gauss(temp,sigma,mu1)*(phi)
        if(p0>p1):
            y_pred[i]=0
        else:
            y_pred[i]=1
    return y_pred

    
