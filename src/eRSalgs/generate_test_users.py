# generate dummy live users / testing users

import sys
import warnings
if not sys.warnoptions:
        warnings.simplefilter("ignore")
# There will be NumbaDeprecationWarnings here, use the above code to hide the warnings 

import pandas as pd 
import numpy as np 
import csv
import setpath



data_path = setpath.setpath()
rating_file = data_path + 'eRS_ratings_g20.csv'
ratings_train = pd.read_csv(rating_file)
    # ['user_id', 'movie_id', 'rating', 'timestamp']
ratings_train = ratings_train.rename(columns = {'user_id': 'user', 'movie_id': 'item'})
    # rename the columns to ['user', 'item', 'rating', 'timestamp']
# print(ratings_train.head(10))

users, counts = np.unique(ratings_train['user'], return_counts = True)
users_rating_count = pd.DataFrame({'user': users, 'count': counts}, columns = ['user', 'count'])

users_rating_count20 =  users_rating_count[users_rating_count['count'] == 20]
# testing_users = users_rating_count20.user.unique()[0:50]
testing_users = np.random.choice(users_rating_count20.user.unique(), 50, replace=False)
print(testing_users)
    # Pick 50 users with 20 movie ratings as testing data set
# print(len(testing_users))
    # 50
testing_users_ratings = ratings_train[ratings_train['user'].isin(testing_users)]    
print(testing_users_ratings.shape)
    # (1000, 4)
testing_users_ratings.to_csv(data_path + 'testing_users_rating.csv', index = False)
print(testing_users_ratings.user.unique())
print(testing_users_ratings.head(100))
