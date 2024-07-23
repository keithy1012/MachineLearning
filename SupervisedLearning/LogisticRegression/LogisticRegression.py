import numpy as np
from scipy.optimize import fmin_tnc

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def net_input(theta, x):
    dot_prod = 0
    for i in range (len(x)):
        dot_prod += theta[i] * x[i]
    return dot_prod

def prob(theta, x):
    return sigmoid(net_input(theta, x))

def cost_function(theta, x, y):
    m = len(x)
    t_cost = -1/m * np.sum(y * np.log(prob(theta, x)) + (1-y)*np.log(1-prob(theta, x)))
    return t_cost

def gradient(theta, x, y):
    m = len(x)
    grad = 0
    for ind in range(m):
        grad += x[ind] * sigmoid(net_input(theta, x)) - y[ind]
    grad /= m
    return grad

def fit(theta, x, y):
    opt_weights = fmin_tnc(func=cost_function, x0=theta, fprime=gradient,
                               args=(x, y.flatten()))
    return opt_weights[0]


def accruacy(x, acc_class, threshold):
    predicted = (predicted(x) >= threshold).astype(int)
    predicted = predicted.flatten()
    accuracy = np.mean(predicted == acc_class)
    return accuracy

def predict(x):
    theta = np.zeros(len(x))
    return prob(theta, x)