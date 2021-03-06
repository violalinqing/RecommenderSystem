# -*- coding:utf-8 -*-

from __future__ import division, print_function
import numpy as np


def loadfile(file_name):
    lines = []
    for line in open(file_name):
        if line == "\n":
            continue
        lines.append(line)
    return lines


class SVDPP(object):
    def __init__(self, dataset, n_factors, n_epochs, alpha, lamda, dataset_test):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.user_id = dict()
        self.item_id = dict()
        self.alpha = alpha
        self.lamda = lamda
        self.dataset_test = dataset_test

        self.user_num = 0
        self.item_num = 0

        self.min_rating = 0.0
        self.max_rating = 0.0

        # 读取dataset
        self.line_count = 0
        self.user_rating = dict()
        self.rating_all = 0.0
        for line in dataset:
            fields = line.split("\t")
            if fields[0] not in self.user_id:
                self.user_id.setdefault(fields[0], self.user_num)
                self.user_num += 1

            if fields[1] not in self.item_id:
                self.item_id.setdefault(fields[1], self.item_num)
                self.item_num += 1

            self.user_rating.setdefault(fields[0], {})
            self.user_rating[fields[0]][fields[1]] = float(fields[2])
            self.rating_all += float(fields[2])

            if float(fields[2]) > self.max_rating:
                self.max_rating = float(fields[2])
            if float(fields[2]) < self.min_rating:
                self.min_rating = float(fields[2])
            self.line_count += 1

        self.rating_matrix = dict()

        self.movie_ranting_mean = self.rating_all / self.line_count
        #初始化
        self.bu = np.zeros((self.user_num, 1), dtype=np.double)
        self.bi = np.zeros(self.item_num, np.double)
        self.p = np.zeros((self.user_num, self.n_factors), np.double) + .1
        self.q = np.zeros((self.item_num, self.n_factors), np.double) + .1
        self.y = np.zeros((self.item_num, self.n_factors), np.double) + .1

        #建立rating表
        for user in self.user_rating.keys():
            user_index = self.user_id[user]
            for item, rating in self.user_rating[user].items():
                item_index = self.item_id[item]
                self.rating_matrix.setdefault(user_index, {})
                self.rating_matrix[user_index][item_index] = rating

    def train(self):
        rmse = 0.0
        for current_epoch in range(self.n_epochs):
            self.rmse_iter = 0
            last_rmse = 1000000
            # 遍历所有user
            for u in self.rating_matrix:
                I_Nu = len(self.rating_matrix[u])
                sqrt_N_u = np.sqrt(I_Nu)
                y_u = 0
                # 基于用户u点评的item集推测u的implicit偏好
                for k in range(0, self.n_factors):
                    y_u += self.y[u][k]

                u_impl_prf = y_u / sqrt_N_u

                # 遍历所有item
                for i in self.rating_matrix[u]:
                    #print(u, "      ", i)
                    rp = self.movie_ranting_mean + self.bu[u] + self.bi[i] + np.dot(self.q[i], self.p[u] + u_impl_prf)

                    e_ui = self.rating_matrix[u][i] - rp

                    self.bu[u] += self.alpha * (e_ui - (self.lamda * self.bu[u]))
                    self.bi[i] += self.alpha * (e_ui - self.lamda * self.bi[i])

                    self.p[u] += self.alpha * (e_ui * self.q[i] - self.lamda * self.p[u])
                    self.q[i] += self.alpha * (e_ui * (self.p[u] + u_impl_prf) - self.lamda * self.q[i])
                    for j in self.rating_matrix[u]:

                        self.y[j] += self.alpha * (e_ui * self.q[j] / sqrt_N_u - self.lamda * self.y[j])

                    rmse += e_ui * e_ui
                    self.rmse_iter +=1

            #均方根误差
            rmse = np.sqrt(rmse / self.rmse_iter)
            print (" iter num:",current_epoch," rmse :", rmse)

            last_rmse = rmse
            self.alpha *= 0.9

            self.test(self.dataset_test)

    def test(self,dataset):

        rating_all = 0.0
        line_count = 0
        rmse_iter = 0
        rmse = 0.0
        for line in dataset:
            fields = line.split("\t")
            rating_all += float(fields[2])
            line_count += 1

        movie_ranting_mean = rating_all / line_count
        for line in dataset:
            fields = line.split("\t")
            u = int(fields[0])-1
            i = int(fields[1])-1
            I_Nu = len(self.rating_matrix[u])
            sqrt_N_u = np.sqrt(I_Nu)
            y_u = 0

            # 基于用户u点评的item集推测u的implicit偏好
            for k in range(0, self.n_factors):
                y_u += self.y[u][k]

            u_impl_prf = y_u / sqrt_N_u

            rp = self.movie_ranting_mean + self.bu[u] + self.bi[i] + np.dot(self.q[i], self.p[u] + u_impl_prf)

            e_ui = float(fields[2]) - rp

            rmse += e_ui * e_ui
            rmse_iter += 1

            # 均方根误差
        rmse = np.sqrt(rmse / rmse_iter)
        print(" test rmse :", rmse)




if __name__ == '__main__':
    dataset_train = loadfile("u2.base")
    dataset_test = loadfile("u2.test")
    svd = SVDPP(dataset_train, 100, 20, 0.001, 0.01, dataset_test)
    svd.train()
    svd.test(dataset_test)