# eRS recommendations

import sys
import warnings
if not sys.warnoptions:
        warnings.simplefilter("ignore")
# There will be NumbaDeprecationWarnings here, use the above code to hide the warnings
         
import numpy as np
import pandas as pd
import setpath
import pickle
import diversification
import MF_predictor



### import testing dataset
data_path = setpath.setpath()
testing_data_file = data_path + 'testing_users_rating.csv'
testing_users_ratings = pd.read_csv(testing_data_file)
    # ['user', 'item', 'rating', 'timestamp']
available_testing_users = testing_users_ratings.user.unique()
    # 50 users, np.ndarray
#print(available_testing_users)
# The available 50 user IDs in the testing dataset for v2
# [   524   2027   2078   3305   6282  11711  14839  20369  23721  33123
#   35379  38445  38558  55951  65813  66025  66670  70732  72915  73028
#   74586  75728  79349  80994  85903  87117  88305  88945  93303  98141
#  101060 107427 108333 109325 110867 111191 111697 112053 113995 116332
#  120596 128009 141996 142773 142786 143158 145111 148466 152492 160253]
    
# The available 50 user IDs in the testing dataset for v3
# [   346   6941   8563   9262  12901  25423  26859  28252  29300  32813
#   34331  39996  49005  50605  59618  60989  63219  63627  64237  64501
#   65179  65801  68288  68298  68903  69085  72198  72593  75189  79616
#   80527  84356  87129  96295 104041 114739 124086 126497 131308 132115
#  138669 139293 141328 141619 145695 145941 152490 154209 155049 157076]
### import movie info and popularity
movie_info_file = data_path + 'eRS_movie_info_g20.csv'
movie_info = pd.read_csv(movie_info_file)
    # ['movie_id', 'imdb_id', 'title(year)', 'title', 'year', 'runtime', 'genre', 'aveRating', 'director', 'writer', 'description', 'cast', 'poster', 'count', 'rank']
movie_title = movie_info[['movie_id', 'title']]
movie_title = movie_title.rename({'movie_id' : 'item'}, axis = 1)
    # ['item', 'title']
item_popularity = movie_info[['movie_id', 'count', 'rank']]
item_popularity = item_popularity.rename({'movie_id' : 'item'}, axis = 1)
    # ['item', 'count', 'rank']
    
### Import the pre-trained MF model which was saved in an object
model_path = './data/'
f_import = open(model_path + 'eRS_implictMF0.pkl', 'rb')
trained_MF_model = pickle.load(f_import)
f_import.close()    

### import emotion data
item_emotions_file = data_path + 'eRS_emotions_g20.csv'
item_emotions_df = pd.read_csv(item_emotions_file)
    # ['imdb_id', 'anticipation', 'joy', 'trust', 'anger', 'fear', 'disgust', 'sadness', 'surprise']
movieID_imdbID = movie_info[['movie_id', 'imdb_id']]
item_emotions_df = pd.merge(item_emotions_df, movieID_imdbID, how = 'left', on = 'imdb_id')
    # ['imdb_id', 'anticipation', 'joy', 'trust', 'anger', 'fear', 'disgust', 'sadness', 'surprise', 'movie_id']
item_emotions_df = item_emotions_df.rename({'movie_id' : 'item'}, axis = 1)
    # ['imdb_id', 'anticipation', 'joy', 'trust', 'anger', 'fear', 'disgust', 'sadness', 'surprise', 'item']

### generate new_ratings of one dummy live user
# liveUserID = np.random.choice(available_testing_users)
liveUserID = 6941 
# or try a different userID from the testing data
# int64, 
ratings_liveUser = testing_users_ratings[testing_users_ratings['user'].isin([liveUserID])]
# print(ratings_liveUser)
    # ['user', 'item', 'rating', 'timestamp']
new_ratings = pd.Series(ratings_liveUser.rating.to_numpy(), index = ratings_liveUser.item)    
    
### predicting
numRec = 10 # set to 10 by default
[RSSA_preds, liveUser_feature] = MF_predictor.live_prediction(trained_MF_model, liveUserID, new_ratings, item_popularity)
    # ['item', 'score', 'count', 'rank', 'discounted_score']
    # liveUser_feature: np.ndarray
RSSA_preds_titled = pd.merge(RSSA_preds, movie_title, how = 'left', on = 'item')
    # ['item', 'score', 'count', 'rank', 'discounted_score', 'title']

# extract the not-rated-yet items
rated_items = ratings_liveUser.item.unique()
RSSA_preds_titled_noRated = RSSA_preds_titled[~RSSA_preds_titled['item'].isin(rated_items)]
     # ['item', 'score', 'count', 'rank', 'discounted_score', 'title']  
#print(RSSA_preds_titled.shape)    
#print(RSSA_preds_titled_noRated.shape)  


################################################################################################################################

##############               Generate Recommendations                 ##########################################################

################################################################################################################################
print('\n\neRS recommendations for user: %s' % liveUserID)
#===> 1 - Discounted TopN
traditional_preds_sorted = RSSA_preds_titled_noRated.sort_values(by = 'score', ascending = False)
    # ['item', 'score', 'count', 'rank', 'discounted_score', 'title']  
discounted_preds_sorted = RSSA_preds_titled_noRated.sort_values(by = 'discounted_score', ascending = False)
    # ['item', 'score', 'count', 'rank', 'discounted_score', 'title']  
recs_topN_traditional = traditional_preds_sorted.head(numRec)
recs_topN_discounted = discounted_preds_sorted.head(numRec)
#print('\nTraditional Top-N:')
#print(recs_topN_traditional)
# print(recs_topN_discounted[['item', 'count', 'rank', 'score', 'discounted_score', 'title']])
print('\n1 - Vanilla Top-N:')
print(recs_topN_discounted[['item', 'title']])


#===> 2 - Diverse Traditional top-200 items by latent features
num_topN = 200
candidates = discounted_preds_sorted.head(num_topN)[['item', 'discounted_score']]
    # ['item', 'score', 'count', 'rank', 'discounted_score', 'title'] 
    # contained only items not rated yet by the the user 
## get the item latent features
    # Refer to mf_common.py in lenskit\algorithms
        # The trained MF model includes the following attributes:
            # user_index_(pandas.Index): Users in the model (length=:math:`m`).
            # item_index_(pandas.Index): Items in the model (length=:math:`n`).
            # user_features_(numpy.ndarray): The :math:`m \\times k` user-feature matrix.
            # item_features_(numpy.ndarray): The :math:`n \\times k` item-feature matrix.    
item_features = trained_MF_model.item_features_
# print(type(item_features))
    # np.ndarray
# print(item_features)
item_index = trained_MF_model.item_index_
# print(type(item_index.values))
    # np.ndarray
[rec_diverseFeature, rec_itemFeature] = diversification.diversify_item_feature(candidates, item_features, item_index.values)
    # rec_diverseFeature: ['item', 'discounted_score']
    # rec_itemFeature : ['0', '1', ..., 'num_features-1'], num_features = 20 here
# print(rec_diverseFeature)
# print(rec_itemFeature)
rec_diverseFeature = pd.merge(rec_diverseFeature, movie_title, how = 'left', on = 'item')
    # ['item', 'discounted_score', 'title']
print('\n2 - Items diversified by latent features:')
print(rec_diverseFeature[['item', 'title']])


#===> 3 - Diverse Traditional top-200 items by WEIGHTED latent features - variance of latent features
weighting = 1 # indicating yes on weighting 
[rec_diverseFeatureW, rec_itemFeatureW] = diversification.diversify_item_feature(candidates, item_features, item_index.values, weighting)
    # diversify_item_feature take weighting = 0 by default as a parameter
    # rec_diverseFeature: ['item', 'discounted_score']
    # rec_itemFeature : ['0', '1', ..., 'num_features-1'], num_features = 20 here
rec_diverseFeatureW = pd.merge(rec_diverseFeatureW, movie_title, how = 'left', on = 'item')
    # ['item', 'discounted_score', 'title']
print('\n3 - Items diversified by WEIGHTED latent features:')
print(rec_diverseFeatureW[['item', 'title']])


#===> 4 - Diverse Traditional top-200 items by emotions
#print(trained_MF_model.item_index_)
    # Int64Index
#print(trained_MF_model.item_index_.get_indexer([98981]))
    # np.ndarray, [11315]
item_emotions = item_emotions_df[['anticipation', 'joy', 'trust', 'anger', 'fear', 'disgust', 'sadness', 'surprise']].to_numpy()
    # np.ndarray of 8 columns matches ['anticipation', 'joy', 'trust', 'anger', 'fear', 'disgust', 'sadness', 'surprise']
item_ids = item_emotions_df.item.unique()
[rec_diverseEmotion, rec_itemEmotion] = diversification.diversify_item_feature(candidates, item_emotions, item_ids)
    # rec_diverseEmotion: ['item', 'discounted_score']
    # rec_itemEmotion : ['0', '1', ..., 'num_emotions-1'], num_features = 8 here
# print(rec_diverseEmotion)
# print(rec_itemEmotion)
rec_diverseEmotion = pd.merge(rec_diverseEmotion, movie_title, how = 'left', on = 'item')
    # ['item', 'discounted_score', 'title']
print('\n4 - Items diversified by emotions:')
print(rec_diverseEmotion[['item', 'title']])


#===> 5 - Diverse Traditional top-200 items by WEIGHTED emotions - variance of latent features
weighting = 1 # for yes   
[rec_diverseEmotionW, rec_itemEmotionW] = diversification.diversify_item_feature(candidates, item_emotions, item_ids, weighting)
    # rec_diverseEmotionW: ['item', 'discounted_score']
    # rec_itemEmotionW : ['0', '1', ..., 'num_emotions-1'], num_features = 8 here
# print(rec_diverseEmotionW)
# print(rec_itemEmotionW)
rec_diverseEmotionW = pd.merge(rec_diverseEmotionW, movie_title, how = 'left', on = 'item')
    # ['item', 'discounted_score', 'title']
print('\n5 - Items diversified by WEIGHTED emotions:')
print(rec_diverseEmotionW[['item', 'title']])
