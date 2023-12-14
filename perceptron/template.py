import numpy as np
# This is the template for various ML related projects
# version 0.0.1
# * currently supports binary classification only
# version 0.0.2
# * Added standardalisation (Note: you have to test and see if the model requires standardization or not)
# * Added cost function mimization as a measure for convergence
#       Pros:
#       Adaptability: This method allows training to stop when a predefined improvement in the cost function is achieved. It adapts to the specific characteristics of the optimization problem.
#       Efficiency: Training can stop as soon as the optimization reaches a satisfactory level, 
#               potentially saving computational resources.
#       Cons:
#       Complexity: Implementing a convergence criterion based on a threshold requires monitoring the cost function during training,
#                    adding some complexity to the algorithm.
#   
#       overall the epoch method is simple to understand however you have to try different values of epoch for each training dataset

def act_fun(x):# we consider the basic heaviside step function
    return np.where(x>=0,1.0,0.0)

class template():
    def __init__(self,num_features,learning_rate=0.001,lmbda=1e-20,epoch=1000): #class constructor
        self.num_features=num_features
        self.lr=learning_rate
        self.wt=np.zeros((num_features+1,1),dtype=float)# a 2d array (a row vector)
        self.bs=np.zeros(1,dtype=float)
        self.lmbda= lmbda
        self.epoch=epoch
        self.cost_history=[]
        self.theta0=[]
        self.theta1=[]
    
    def predict(self,x):
        #print(f"x type: {type(x)}, self.wt type: {type(self.wt)}, self.bs type: {type(self.bs)}")
        linear=np.dot(x,self.wt)+self.bs
        predictions=act_fun(linear)
        return predictions
    def error(self,x,y):
        #print(x.shape)
        predictions=self.predict(x)
        error=y-predictions
        #print(error.shape)
        return error
    def cost(self,x,y):
        m=len(y)
        return np.sum(self.error(x,y)**2)*(1/(2*m))
    def train(self,x,y):
        #x=(x-np.mean(x))/np.std(x)
        m=len(y)
        prev_cost=1e5
        curr_cost=0
        while(abs((curr_cost-prev_cost))>=self.lmbda): # if want convergence by lambda
        #for e in range (self.epoch):# for each epoch we traverse through each training example
            
            for i in range(m): # for each training exaple (y.shape[0] represents the number of training examples)
                curr_cost=self.cost(x,y)
                errors=(1/m)*self.error(x[i].reshape(1,self.num_features),y[i]).reshape(-1)
                #   error returned is a 2d array([]) while the bs is one dimensional(1,)
                self.wt+=(self.lr)*(errors*x[i]).reshape(self.num_features,1)
                self.bs+=(self.lr)*errors
                self.cost_history.append(curr_cost)
                self.theta0.append(self.wt)
                self.theta1.append(self.bs)
                prev_cost=curr_cost
        
    def evaluate(self,x,y):
        #x=(x-np.mean(x))/np.std(x)
        predictions= self.predict(x).reshape(-1)
        accuracy=np.sum(predictions==y)/y.shape[0]
        return accuracy
        