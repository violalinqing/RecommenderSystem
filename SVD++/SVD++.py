# -*- coding:utf-8 -*-

from __future__ import division, print_function
import os
import numpy as np
import pandas as pd

def get_train_dataset(filename):
   # frame = pd.read_csv(filename)
    my_matrix = numpy.loadtxt(open(filename, "rb"), delimiter=",", skiprows=0)
   # userID = list(set(frame['UserID'].values))
   # itemID = list(set(frame['MovieID'].values))


class SVDPlusPlus():
    """
    属性
    ---------
    n_factors : 隐式因子数
    n_epochs : 迭代次数
    lr : 学习速率
    lamda : 正则因子
    """

    def __init__(self, n_factors=20, n_epochs=20, lr=0.007, lamda=.002):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.lamda = lamda

    def train(self, train_dataset):
        user_num = train_dataset.matrix.shape[0]
        item_num = train_dataset.matrix.shape[1]
        self.train_dataset = train_dataset

        self.global_mean = train_dataset.global_mean

        self.bu = np.zeros(user_num, np.double)

        self.bi = np.zeros(item_num, np.double)

        self.p = np.zeros((user_num, self.n_factors), np.double) + .1

        self.q = np.zeros((item_num, self.n_factors), np.double) + .1

        #item preference facotor
        self.y = np.zeros((item_num, self.n_factors), np.double) + .1

        for current_epoch in range(self.n_epochs):
            print(" processing epoch {}".format(current_epoch))
            for u, i, r in train_dataset.all_ratings():
                #用户u点评的item集
                Nu = train_dataset.get_user(u)[0]
                I_Nu = len(Nu)
                sqrt_N_u = np.sqrt(I_Nu)

                #基于用户u点评的item集推测u的implicit偏好
                y_u = np.sum(self.y[Nu], axis=0)

                u_impl_prf = y_u / sqrt_N_u

                #预测值
                rp = self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.q[i], self.p[u] + u_impl_prf)

                #误差
                e_ui = r - rp

                #sgd
                self.bu[u] += self.lr * (e_ui - self.lamda * self.bu[u])
                self.bi[i] += self.lr * (e_ui - self.lamda * self.bi[i])
                self.p[u] += self.lr * (e_ui * self.q[i] - self.lamda * self.p[u])
                self.q[i] += self.lr * (e_ui * (self.p[u] + u_impl_prf) - self.lamda * self.q[i])
                for j in Nu:
                    self.y[j] += self.lr * (e_ui * self.q[j] / sqrt_N_u - self.lamda * self.y[j])

    def predict(self, u, i):
        Nu = self.train_dataset.get_user(u)[0]
        I_Nu = len(Nu)
        sqrt_N_u = np.sqrt(I_Nu)
        y_u = np.sum(self.y[Nu], axis=0) / sqrt_N_u

        est = self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.q[i], self.p[u] + y_u)
        return est

if __name__ == '__main__':
    file_name = os.path.abspath("u.data")