# -*- coding: utf-8 -*-

import os
import math
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds


os.chdir('/Users/apple/Documents/workspace/SVDRecommenderbyLin')

ratings = pd.read_csv('ratings.csv')
movie_ranting_by_user = ratings.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
movie_ranting_mat = movie_ranting_by_user.as_matrix()
movie_ranting_mean = np.mean(movie_ranting_mat,0)
movie_ranting_mean = movie_ranting_mean.reshape(1,-1)

movie_ranting_mat_demeaned = movie_ranting_mat - movie_ranting_mean
U,S,V = svds(movie_ranting_mat_demeaned,150)
S = np.diag(S)

pre_ranting_mat = np.dot(np.dot(U,S),V) + movie_ranting_mean

prediction_df = pd.DataFrame(pre_ranting_mat, index = movie_ranting_by_user.index, columns = movie_ranting_by_user.columns)

def movie_recommender(rantings_df, pre_df, num):
    i = 0
    rec_df = pd.DataFrame(index = range(671*num+1), columns=['userId','rec_movieId','pre_ranting'])
    for user in range(1,672):
        movieList_of_user = list(rantings_df.loc[rantings_df['userId'] == user].iloc[:,1])
        pre_df_of_user = pre_df.loc[user].sort_values(ascending = False)
        count = 0
        for movie in pre_df_of_user.index.values:
            if count == num:
                break
            if movie in movieList_of_user:
                continue
            rec_df.loc[i,'userId'] = user
            rec_df.loc[i,'rec_movieId'] = movie
            rec_df.loc[i,'pre_ranting'] = pre_df.loc[user,movie]
            count += 1
            i += 1

    rec_df = rec_df.dropna()
    rec_df.to_csv('all_user_rec.csv',sep=',')

movie_recommender(ratings,prediction_df,15)

