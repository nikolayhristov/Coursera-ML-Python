import numpy as np

def sigmoid(z):
    if type(z) != 'numpy.ndarray':
        z = np.array(z)
    g = np.zeros(z.size)
    g = 1/(1+np.exp(-z))
    return g

def costFunction(theta,X,y):
    m = y.size
    J = 0
    grad = np.zeros(theta.size)
    #theta = theta.reshape((theta.shape[0],1))
    #y= y.reshape(m,1)
    
    
    #J = 1/m*np.sum(-y*np.log(sigmoid(X@theta))-(1-y)*np.log(1-sigmoid(X@theta)))
    #grad = 1/m*np.sum((sigmoid(X@theta)-y)*X,axis=0)
    h = sigmoid(np.matmul(X,theta))
    J = 1/m*(-np.matmul(y,np.log(h))-np.matmul(1-y,np.log(1-h)))
    grad = 1/m*np.matmul(X.T,h-y)
    
    return (J,grad)


        


        
