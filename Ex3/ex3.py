import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.io
from numpy.random import default_rng

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

def plotmnist(X,m=10,n=10):
	fig, axs = plt.subplots(m, n)
	l = max(m,n)
	if m>=n:
		for k in range(m*n):
			axs[k%l,k//l].imshow(X[k].reshape((20,20)).T,cmap='gray')
			#axs[k%l,k//l].set_title(y_train[k+offset])
			axs[k%l,k//l].axes.xaxis.set_ticks([])
			axs[k%l,k//l].axes.yaxis.set_ticks([])
	else:
		for k in range(m*n):
			axs[k//l,k%l].imshow(X[k].reshape((20,20)).T,cmap='gray')
			#axs[k//l,k%l].set_title(y_train[k+offset])
			axs[k//l,k%l].axes.xaxis.set_ticks([])
			axs[k//l,k%l].axes.yaxis.set_ticks([])
	plt.show()

def oneVsAll(X,y,num_labels,lamb):
    m = X.shape[0]
    n = X.shape[1]
    
    all_theta = np.zeros((num_labels,n))
    
    for c in range(1,11):
        initial_theta = np.zeros(n)
        costfunc = lambda theta: costFunctionReg(theta=theta,X=X,y=(y==c),lamb=lamb)[0]
        costgrad = lambda theta: costFunctionReg(theta=theta,X=X,y=(y==c),lamb=lamb)[1]
        
        res = minimize(costfunc, initial_theta, method='BFGS', jac=costgrad,
                       options={'gtol': 1e-6, 'disp': True})
        
        all_theta[c-1] = res.x
        
    return all_theta
        
        
def predictOneVsAll(theta,X):
    m = X.shape[0]
    num_labels = theta.shape[0]
    p = np.zeros(X.shape[0])
    
    p = np.argmax(sigmoid(np.matmul(theta,X.T)),axis=0)+1
    return p

def main():
    input_layer_size = 4000
    num_labels = 10
    data = scipy.io.loadmat('ex3data1-pythoncompatible.mat')
    data = data["data"][0][0]
    X = data[0]
    y = data[1]
    y = y.reshape((y.shape[0],))
    rng = default_rng(123)
    to_print = rng.integers(5000,size=100)
    plotmnist(X=X[to_print])
    
    X = np.column_stack((np.ones(X.shape[0]),X))
    
    theta = oneVsAll(X,y,num_labels,0.1)
    
    predictions = predictOneVsAll(theta,X)
    print('Train accuracy: {acc}'.format(acc= np.mean(predictions==y)*100))

if __name__ == "__main__":
    main()