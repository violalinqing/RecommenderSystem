# -*- coding: utf-8 -*-

import math
import random



class LogReg(object):

    def __init__(self, type, alpha = 0.01, iter = 50):
        self.alpha = alpha
        self.iter = iter
        self.type = type


    def loadData(self,filename):
        data = []
        label = []
        for line in open(filename):
            if line == "\n":
                continue
            fields = line.split(",")
            fea = []
            lab = float(fields[-1])
            for i in range(0, len(fields) - 1):
                value = float(fields[i])
                fea.append(value)
            label.append(lab)
            data.append(fea)
        return data, label


    def sigmoid(self,gamma):
        if gamma < 0:
            return 1 - 1 / (1 + math.exp(gamma))
        else:
            return 1 / (1 + math.exp(-gamma))


    def getMatResult(self, data):
        result = 0.0
        for i in range(0,len(data)):
            #print data[i],self.weights[i]
            result += data[i] * self.weights[i]
        return result


    def trainLR(self, data,label):

        self.weights = []
        for i in range(0, len(data[0])):
             self.weights.append(1.0)

        for i in range(0, self.iter):
            errors = []
            if self.type == "gradDescent":
                for k in range(0, len(data)):
                    result = self.getMatResult(data[k])
                    error = label[k] - self.sigmoid(result)
                    errors.append(error)
                for k in range(0, len(self.weights)):
                    updata = 0.0
                    for idx in range(0, len(errors)):
                        updata += errors[idx] * data[idx][k]
                    self.weights[k] -= self.alpha * updata
            elif self.type == "stocGradDescent":
                for k in range(0, len(data)):
                    result = self.getMatResult(data[k])
                    error = label[k] - self.sigmoid(result)
                    for idx in range(0, len(self.weights)):
                        self.weights[idx] -= self.alpha*error*data[k][idx]

            elif self.type == "betterStocGradDescent":
                for k in range(0, len(data)):
                    randomIndex = int(random.uniform(0,len(data)))
                    result = self.getMatResult(data[randomIndex])
                    error = label[k] - self.sigmoid(result)
                    for idx in range(0, len(self.weights)):
                        self.weights[idx] -= self.alpha*error*data[k][idx]




    def testLR(self,data, label):
        matchCount = 0
        numSamples = 0

        for i in range(len(data)):
            result = self.getMatResult(data[i])
            numSamples += 1
            predict = self.sigmoid(result) > 0.5
            if predict == int(label[i]):
                matchCount += 1
            if numSamples!= 0:
                accuracy = float(matchCount) / numSamples
        print 'the accuracy is ', accuracy


if __name__ == '__main__':
    lr = LogReg('GradDescent')
    traData, traLabel = lr.loadData('traindata.txt')
    testData, testLabel = lr.loadData('testdata.txt')
    lr.trainLR(traData, traLabel)
    lr.testLR(testData, testLabel)