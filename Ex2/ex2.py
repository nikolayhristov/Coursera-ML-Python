import numpy as np
import matplotlib.pyplot as plt
from costFunction import *
from scipy.optimize import minimize
'''
test cases

X = np.array([[1,8,1,6],[1,3,5,7],[1,4,9,2]])
y = np.array([1,0,1])
theta = np.array([-2,-1,1,2])
'''

        
        
data = np.loadtxt('ex2data1.txt',delimiter=',')
X = data[:,:2]
y = data[:,2]

#plotting training data
plt.figure()
plt.scatter(X[y==1,0],X[y==1,1], c = 'k',marker='+',label='Admitted')
plt.scatter(X[y==0,0],X[y==0,1], c = 'y',marker='o',label='Not admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.title('Scatter plot of training data')
plt.legend()

plt.figure()
plt.scatter(X[y==1,0],X[y==1,1], c = 'k',marker='+',label='Admitted')
plt.scatter(X[y==0,0],X[y==0,1], c = 'y',marker='o',label='Not admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.title('Scatter plot of training data')
plt.legend()
m,n = X.shape

X = np.column_stack((np.ones(m),X))

initial_theta = np.zeros(n+1)
cost, grad = costFunction(initial_theta,X,y)
print('Cost at initial theta (zeros): '+str(cost)+'\nGradients at initial theta (zeros): ' + str(grad))
test_theta = np.array([-24, 0.2, 0.2])
[cost, grad] = costFunction(test_theta, X, y)
print('Cost at test theta: %.3f\nGradients at test theta: ' % (cost) + str(grad))

func = lambda theta: costFunction(theta=theta,X=X,y=y)[0]
deriv = lambda theta: costFunction(theta=theta,X=X,y=y)[1]
res = minimize(func, initial_theta, method='BFGS', jac=deriv,
               options={'gtol': 1e-6, 'disp': True})

theta = res.x
cost = func(theta)


plot_x = np.array([np.min(X[:,1])-2,np.max(X[:,1])+2])
plot_y = -1/theta[2]*(theta[1]*plot_x+theta[0])
plt.plot(plot_x,plot_y)


plt.show()