# -*- coding: utf-8 -*-

import numpy as np

'''符号函数'''
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

'''
逻辑回归训练
'''
def train_logRegres(train_x, train_y, alpha,maxIter,optimizeType):
    numSamples, numFeatures = np.shape(train_x)
    alpha = alpha #步长
    maxIter = maxIter #迭代次数
    #权重 = theta
    weights = np.ones((numFeatures, 1)) #初始化参数为1

    for k in range(maxIter):
        if optimizeType == 'gradDescent': # 梯度下降算法
            output = sigmoid(train_x * weights)
            error = train_y - output
            weights = weights + alpha * train_x.transpose() * error
        elif optimizeType == 'stocGradDescent': # 随机梯度下降
            for i in range(numSamples):
                output = sigmoid(train_x[i, :] * weights)
                error = train_y[i, 0] - output
                weights = weights + alpha * train_x[i, :].transpose() * error
        elif optimizeType == 'smoothStocGradDescent': # 平稳随机梯度下降
            dataIndex = range(numSamples)
            for i in range(numSamples):
                alpha = 4.0 / (1.0 + k + i) + 0.01
                randIndex = int(np.random.uniform(0, len(dataIndex)))
                output = sigmoid(train_x[randIndex, :] * weights)
                error = train_y[randIndex, 0] - output
                weights = weights + alpha * train_x[randIndex, :].transpose() * error
                del(dataIndex[randIndex])
        else:
            raise NameError('Not support optimize method type!')

    print(weights)
    return weights

'''逻辑回归测试'''
def test_LogRegres(weights, test_x, test_y):
    numSamples, numFeatures = np.shape(test_x)
    matchCount = 0
    for i in xrange(numSamples):
        predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5
        if predict == bool(test_y[i, 0]):
            matchCount += 1
    accuracy = float(matchCount) / numSamples
    return accuracy

def loadFile(filename):
    train_x =[]
    train_y =[]
    fileIn = open(filename)
    for line in fileIn.readlines():
        lineArr = line.strip().split()
        train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])
        train_y.append(float(lineArr[2]))
    return np.mat(train_x), np.mat(train_y).transpose()


def logRegresMain():
    train_x, train_y = loadFile('traindata.txt')
    test_x ,test_y = loadFile('testdata.txt')

    alpha = 0.01
    maxIter = 200
    optimizeType = 'gradDescent'#调用的方法

    optimalWeights = train_logRegres(train_x, train_y, alpha,maxIter,optimizeType)

    accuracy = test_LogRegres(optimalWeights, test_x, test_y)

    print 'The classify accuracy is: %.3f%%' % (accuracy * 100)



if __name__=='__main__':
    logRegresMain()
