import numpy as np
import matplotlib.pyplot as plt
from costFunction import *
from scipy.optimize import minimize



def sigmoid(z):
    '''

    Parameters
    ----------
    z : float or np array

    Returns
    -------
    g : sigmoid function, evaluated pointwise on the array

    '''
    g = np.zeros(z.size)
    g = 1/(1+np.exp(-z))
    return g

def mapFeature(X1,X2):
    '''
    

    Parameters
    ----------
    X1 : number or numpy array of size m
    X2 : number or numpy array of size m

    Returns
    -------
    out : returns an mx28 dimensional array containing all combinations of X1^n X2^k
    with n and k <= 6
    

    '''
    
    degree = 6
    out = []
    if type(X1)==np.ndarray:
        out.append(list(np.ones(X1.shape[0])))
        for i in range(1,degree+1):
            for j in range(i+1):
                out.append(list(X1**(i-j)*X2**j))
    else:
        out.append(1)
        for i in range(1,degree+1):
            for j in range(i+1):
                out.append(X1**(i-j)*X2**j)

    out = np.array(out).T
    
    return out

def plotDecisionBoundary(theta,X,y):
    '''
    

    Parameters
    ----------
    Plots the decision boundary and data points given data (X,y) and model parameters theta

    '''
    u = np.linspace(-1,1.5,50)
    v = np.linspace(-1,1.5,50)
    
    z = np.zeros((u.size,v.size))
    
    for i in range(u.size):
        for j in range(v.size):
            z[i,j] = mapFeature(u[i],v[j]).dot(theta)
            
    z = z.T
    plt.contour(u,v,z, levels=0)
    

def costFunctionReg(theta,X,y,lamb):
    '''
    

    Parameters
    ----------
    theta : logistic regression model parameters
    X : mxn numpy array containing m training points and a n features
    y : size m array containing the responses of the training data
    lamb : logistic regression regularization parameter

    Returns
    -------
    J : the cost function for the regularized logistic regression model
    grad : gradient of the regularized logistic regression model

    '''
    m = y.shape[0]
    J = 0 
    grad = np.zeros(theta.shape[0])
    h = sigmoid(np.matmul(X,theta))
    lambvec = lamb*np.ones(theta.shape[0])
    lambvec[0]=0
    J = 1/m*(-np.matmul(y,np.log(h))-np.matmul(1-y,np.log(1-h)))+1/(2*m)*np.sum(lambvec*theta**2)
    grad = 1/m*np.matmul(X.T,h-y)+1/m*lambvec*theta
    return (J,grad)

def predict(theta,X):
    m = X.shape[0]
    
    p = np.round(sigmoid(np.matmul(X,theta)))
    return p
    
def main():
    data = np.loadtxt('ex2data2.txt',delimiter=',')
    X = data[:,:2]
    y = data[:,2]
    
    #plotting training data
    plt.figure()
    plt.scatter(X[y==1,0],X[y==1,1], c = 'k',marker='+',label='y=1')
    plt.scatter(X[y==0,0],X[y==0,1], c = 'y',marker='o',label='y=0')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.title('Scatter plot of training data')
    plt.legend()
    
    plt.figure()
    plt.scatter(X[y==1,0],X[y==1,1], c = 'k',marker='+',label='y=1')
    plt.scatter(X[y==0,0],X[y==0,1], c = 'y',marker='o',label='y=0')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.title('Scatter plot of training data')
    plt.legend()
    
    X = mapFeature(X[:,0],X[:,1])
    
    initial_theta = np.zeros(X.shape[1])
    lamb = 1
    cost, theta = costFunctionReg(initial_theta, X, y, lamb)
    
    initial_theta = np.ones(X.shape[1])
    cost, theta = costFunctionReg(initial_theta,X,y,10)
    
    costfunc = lambda theta: costFunctionReg(theta=theta,X=X,y=y,lamb=lamb)[0]
    costgrad = lambda theta: costFunctionReg(theta=theta,X=X,y=y,lamb=lamb)[1]
    
    res = minimize(costfunc, initial_theta, method='BFGS', jac=costgrad,
                   options={'gtol': 1e-6, 'disp': True})
    
    theta = res.x
    
    plotDecisionBoundary(theta, X, y)
    p = predict(theta,X)
    print('Train accuracy: {acc}'.format(acc= np.mean(p==y)*100))
    
if __name__ == "__main__":
    main()
