import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.io
from numpy.random import default_rng

data = scipy.io.loadmat('ex4data1.mat')
X = data["X"]
y = data["y"]
weights = scipy.io.loadmat('ex4weights.mat')
theta1 = weights['Theta1']
theta2 = weights['Theta2']
input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10

