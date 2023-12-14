import numpy as np


#cost finding function, i.e. J(theta)

def findCost(y,X,theta):
    m = np.shape(X)[0]
    error = y - np.matmul(X,theta)
    return (np.matmul(np.transpose(error),error))/(2*m)


#finding gradient of J(theta)

def deltaJ(y,X,theta):
    m = np.shape(X)[0]
    return np.matmul(np.transpose(X), np.matmul(X,theta) - y)/m






#defining a learning rate

learningRate = 0.1


#Gradient descent function


def GradDescent(y,X,theta,learningRate, costList, thetaList0,thetaList1):

    #initializing a starting cost
    prevcost = 1e5
    while True:


        cost = findCost(y,X,theta)
        thetaList0 = np.append(thetaList0,theta[0])
        thetaList1 = np.append(thetaList1,theta[1])
        costList = np.append(costList,cost)
        

        #breaking condition defined

        if abs((cost-prevcost).item())<1e-10:
            break

        #modifying theta

        theta = theta - (learningRate)*(deltaJ(y,X,theta))
        
        prevcost = cost

    return theta,costList,thetaList0,thetaList1




#the actual theta is found using the next comment. Used this to verify precision and accuracy
#correcttheta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.transpose(X)),y)
