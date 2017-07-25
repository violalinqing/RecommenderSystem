# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as opt

class LR(object):

    def __init__(self, X ,Y):
        self.X = X
        self.Y = Y

    def loadfile(self,file_name):
        lines = []
        for line in open(file_name):
            if line == "\n":
                continue
            lines.append(line)
        return lines

    def sigmoid(self,z):
        return 1.0 / (1 + np.exp(-z))

    def costFunction(self,theta, X, y):
        m = y.size
        h = self.sigmoid(X.dot(theta))
        J = (-np.log(h).T.dot(y) - np.log(1 - h).T.dot(1 - y)) / m
        return J

    def gradientDescent(self,theta, X, y):
        m = y.size
        h = self.sigmoid(X.dot(theta))
        grad = (X.T.dot(h - y)) / m
        return grad

    def findMinTheta(self, theta, X, y):
        result = opt.fmin(self.costFunction, x0=theta, args=(X, y), maxiter=500, full_output=True)
        return result[0], result[1]

    def trainLR(self):

        self.theta = np.zeros((X.shape[1], 1))

        self.theta, cost = self.findMinTheta(self.theta, X, Y)

        print 'when theta = %s' % self.theta + ' , cost  = %f' % cost

    def testLR(self):
        dataset = self.loadfile("testdata.txt")
        matchCount = 0
        numSamples = 0
        for line in dataset:
            numSamples += 1
            fields = line.split(",")
            predict = self.sigmoid(np.array([[1, float(fields[0]), float(fields[1])]]).dot(self.theta.T)) > 0.5
            if predict == int(fields[2]):
                matchCount += 1
            if numSamples!= 0:
                accuracy = float(matchCount) / numSamples
        print 'the accuracy is ', accuracy

if __name__ == '__main__':

    f = open("ex2data1.txt", 'r')
    lines = f.readlines()
    xList = list()
    yList = list()
    for line in lines:
        values = line.replace('\n', '').split(',')
        for i, v in enumerate(values):
            values[i] = float(v)
        xList.append(values[:-1])
        yList.append([int(values[-1])])
    f.close()

    X_2 = np.array(xList)
    Y = np.array(yList)

    m = len(Y)

    X_1 = np.ones((m, 1))
    X = np.hstack((X_1, X_2))

    lr = LR(X,Y)
    lr.trainLR()
    lr.testLR()