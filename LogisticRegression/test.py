# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as opt

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def costFunction(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    J = (-np.log(h).T.dot(y) - np.log(1 - h).T.dot(1 - y)) / m
    return J

def gradientDescent(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    grad = (X.T.dot(h-y)) / m
    return grad

def findMinTheta(theta, X, y):
    result = opt.fmin(costFunction, x0=theta, args=(X, y), maxiter=500, full_output=True)
    return result[0], result[1]

f = open("ex2data1.txt", 'r')
lines = f.readlines()
xList = list()
yList = list()
for line in lines:
    values = line.replace('\n','').split(',')
    for i, v in enumerate(values):
        values[i] = float(v)
    xList.append(values[:-1])
    yList.append([int(values[-1])])
f.close()

X_2 = np.array(xList)
Y = np.array(yList)

m = len(Y)



X_1 = np.ones((m,1))
X = np.hstack((X_1,X_2))

theta = np.zeros((X.shape[1],1))

FS = raw_input("Enter the first score:")
FS = int(FS)
SS = raw_input("Enter the second score:")
SS = int(SS)

theta, cost = findMinTheta(theta, X, Y)

print 'when theta = %s' % theta+' , cost  = %f'% cost

prob = sigmoid(np.array([[1,FS,SS]]).dot(theta.T))

print 'the probability is ', prob
