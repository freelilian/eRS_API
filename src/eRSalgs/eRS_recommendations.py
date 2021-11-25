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

### import movie info and get popularity
def import_movie_info(movie_info_filename):
    movie_info = pd.read_csv(movie_info_filename)
        # ['movie_id', 'imdb_id', 'title(year)', 'title', 'year', 'runtime', 'genre', 'aveRating', 'director', 'writer', 'description', 'cast', 'poster', 'count', 'rank']
    movie_title = movie_info[['movie_id', 'title']]
    movie_title = movie_title.rename({'movie_id' : 'item'}, axis = 1)
        # ['item', 'title']
    item_popularity = movie_info[['movie_id', 'count', 'rank']]
    item_popularity = item_popularity.rename({'movie_id' : 'item'}, axis = 1)
        # ['item', 'count', 'rank']
    movieID_imdbID = movie_info[['movie_id', 'imdb_id']]
        # ['movie_id', 'imdb_id']
    
    return item_popularity, movieID_imdbID, movie_title
    
### Import the pre-trained MF model 
def import_trained_model(model_filename):
    f_import = open(model_filename, 'rb')
    trained_MF_model = pickle.load(f_import)
    f_import.close()
    
    return trained_MF_model

### import emotion data
def get_emotion_data(item_emotions_filename, movieID_imdbID):
    '''
    The offline emotion data was saved with imdb_id, so need to be matched with 
    '''
    item_emotions_df = pd.read_csv(item_emotions_filename)
        # ['imdb_id', 'anticipation', 'joy', 'trust', 'anger', 'fear', 'disgust', 'sadness', 'surprise']
    item_emotions_df = pd.merge(item_emotions_df, movieID_imdbID, how = 'left', on = 'imdb_id')
        # ['imdb_id', 'anticipation', 'joy', 'trust', 'anger', 'fear', 'disgust', 'sadness', 'surprise', 'movie_id']
    item_emotions_df = item_emotions_df[['imdb_id', 'movie_id', 'anticipation', 'joy', 'trust', 'anger', 'fear', 'disgust', 'sadness', 'surprise']]
    # item_emotions_df.to_csv('./data/eRS_emotions_IDlinked.csv', index = False)
    item_emotions_df = item_emotions_df.rename({'movie_id' : 'item'}, axis = 1)
        # ['imdb_id', 'item', 'anticipation', 'joy', 'trust', 'anger', 'fear', 'disgust', 'sadness', 'surprise']

    item_emotions = item_emotions_df[['anticipation', 'joy', 'trust', 'anger', 'fear', 'disgust', 'sadness', 'surprise']].to_numpy()
        # np.ndarray of 8 columns matches ['anticipation', 'joy', 'trust', 'anger', 'fear', 'disgust', 'sadness', 'surprise']
    item_ids = item_emotions_df.item.to_numpy()
    
    return item_emotions, item_ids
    

### get new ratings of the dummy live user from the testing dataset
def get_dummy_live_user_ratings(liveUserID):
    data_path = setpath.setpath()
    testing_data_filename = data_path + 'testing_users_rating.csv'
    testing_users_ratings = pd.read_csv(testing_data_filename)
        # ['user', 'item', 'rating', 'timestamp']
    available_testing_users = testing_users_ratings.user.unique()
        # 50 users, np.ndarray
    #print(available_testing_users)
    # The available 50 user IDs in the testing dataset
    # [   524   2027   2078   3305   6282  11711  14839  20369  23721  33123
    #   35379  38445  38558  55951  65813  66025  66670  70732  72915  73028
    #   74586  75728  79349  80994  85903  87117  88305  88945  93303  98141
    #  101060 107427 108333 109325 110867 111191 111697 112053 113995 116332
    #  120596 128009 141996 142773 142786 143158 145111 148466 152492 160253]
    
    ## new ratings for live user
    ratings_liveUser = testing_users_ratings[testing_users_ratings['user'].isin([liveUserID])]
        # ['user', 'item', 'rating', 'timestamp']

    return ratings_liveUser
    

        
def get_RSSA_preds(liveUserID):
    data_path = setpath.setpath()
    movie_info_filename = data_path + 'eRS_movie_info_g20.csv'
    [eRS_item_popularity, eRS_movieID_imdbID, _] = import_movie_info(movie_info_filename)
    # eRS_item_popularity.to_csv('./data/eRS_item_popularity.csv', index = False)
    # eRS_movieID_imdbID.to_csv('./data/eRS_movieID_imdbID_links.csv', index = False)
    
    model_path = './data/'
    model_filename = model_path + 'eRS_implictMF0.pkl'
    trained_MF_model = import_trained_model(model_filename)
    
    ratings_liveUser = get_dummy_live_user_ratings(liveUserID) 
    new_ratings = pd.Series(ratings_liveUser.rating.to_numpy(), index = ratings_liveUser.item)
    # extract the not-rated-yet items
    rated_items = ratings_liveUser.item.unique()
    
    ## predicting
    [RSSA_preds, liveUser_feature] = MF_predictor.live_prediction(trained_MF_model, liveUserID, new_ratings, eRS_item_popularity)
        # ['item', 'score', 'count', 'rank', 'discounted_score']
        # liveUser_feature: np.ndarray
    # extract the not-rated-yet items
    RSSA_preds_of_noRatedItems = RSSA_preds[~RSSA_preds['item'].isin(rated_items)]
        # ['item', 'score', 'count', 'rank', 'discounted_score']
    
    return RSSA_preds_of_noRatedItems, trained_MF_model, eRS_movieID_imdbID

def topn(liveUserID):
    [RSSA_preds_of_noRatedItems, trained_MF_model, _] = get_RSSA_preds(liveUserID)
        # ['item', 'score', 'count', 'rank', 'discounted_score']
    ## recommendation
    numRec = 10
    traditional_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
    discounted_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'discounted_score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
    recs_topN_traditional = traditional_preds_sorted.head(numRec)
    recs_topN_discounted = discounted_preds_sorted.head(numRec)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
    
    
    return recs_topN_discounted.item.to_numpy(), recs_topN_discounted
        # a np.ndarray of the recommended item_ids (matches movie_id in the movieLens dataset)
        
def diversified_by_latent_feature(liveUserID):
    [RSSA_preds_of_noRatedItems, trained_MF_model, _] = get_RSSA_preds(liveUserID)
        # ['item', 'score', 'count', 'rank', 'discounted_score']
    traditional_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
    discounted_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'discounted_score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
        
    ## diversified by latent feature 
    num_topN = 200
    candidates = discounted_preds_sorted.head(num_topN)[['item', 'discounted_score']]
    ## recommendation
    item_features = trained_MF_model.item_features_
    # print(type(item_features))
        # np.ndarray
    # print(item_features)
    item_index = trained_MF_model.item_index_
    # print(type(item_index.values))
        # np.ndarray
    [rec_diverseFeature, rec_itemFeature] = diversification.diversify_item_feature(candidates, item_features, item_index.values)
        # rec_diverseFeature: ['item', 'discounted_score']
        # rec_itemFeature : ['anticipation', 'joy', 'trust', 'anger', 'fear', 'disgust', 'sadness', 'surprise']
    
    return rec_diverseFeature.item.to_numpy(), rec_diverseFeature
        # a np.ndarray of the recommended item_ids (matches movie_id in the movieLens dataset)
    
    
def diversified_by_weighted_latent_feature(liveUserID):
    [RSSA_preds_of_noRatedItems, trained_MF_model, _] = get_RSSA_preds(liveUserID)
        # ['item', 'score', 'count', 'rank', 'discounted_score']
    traditional_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
    discounted_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'discounted_score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
    
    ## diversified by weighted latent feature 
    num_topN = 200
    candidates = discounted_preds_sorted.head(num_topN)[['item', 'discounted_score']]
    ## recommendation
    item_features = trained_MF_model.item_features_
    # print(type(item_features))
        # np.ndarray
    # print(item_features)
    item_index = trained_MF_model.item_index_
    # print(type(item_index.values))
        # np.ndarray
    weighting = 1 # indicating yes on weighting 
    [rec_diverseFeatureW, rec_itemFeatureW] = diversification.diversify_item_feature(candidates, item_features, item_index.values, weighting)
        # rec_diverseFeatureW: ['item', 'discounted_score']
        # rec_itemFeatureW: ['anticipation', 'joy', 'trust', 'anger', 'fear', 'disgust', 'sadness', 'surprise']

    return rec_diverseFeatureW.item.to_numpy(), rec_diverseFeatureW
        # a np.ndarray of the recommended item_ids (matches movie_id in the movieLens dataset)
    
    
def diversified_by_emotion(liveUserID):
    [RSSA_preds_of_noRatedItems, trained_MF_model, movieID_imdbID] = get_RSSA_preds(liveUserID)
        # ['item', 'score', 'count', 'rank', 'discounted_score']
    traditional_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
    discounted_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'discounted_score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
    
    ## diversified by emotion
    num_topN = 200
    candidates = discounted_preds_sorted.head(num_topN)[['item', 'discounted_score']]
    ## recommendation
    data_path = setpath.setpath()
    item_emotions_filename = data_path + 'eRS_emotions_g20.csv'
    [item_emotions, item_ids] = get_emotion_data(item_emotions_filename, movieID_imdbID)
    [rec_diverseEmotion, rec_itemEmotion] = diversification.diversify_item_feature(candidates, item_emotions, item_ids)
        # rec_diverseEmotion: ['item', 'discounted_score']
        # rec_itemEmotion : ['0', '1', ..., 'num_emotions-1'], num_features = 8 here
 
    return rec_diverseEmotion.item.to_numpy(), rec_diverseEmotion
        # a np.ndarray of the recommended item_ids (matches movie_id in the movieLens dataset)
        
def diversified_by_weighted_emotion(liveUserID):
    [RSSA_preds_of_noRatedItems, trained_MF_model, movieID_imdbID] = get_RSSA_preds(liveUserID)
        # ['item', 'score', 'count', 'rank', 'discounted_score']
    traditional_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
    discounted_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'discounted_score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
        
    ## diversified by weighted emotion    
    num_topN = 200
    candidates = discounted_preds_sorted.head(num_topN)[['item', 'discounted_score']]
    ## recommendation
    item_emotions_filename = data_path + 'eRS_emotions_g20.csv'
    [item_emotions, item_ids] = get_emotion_data(item_emotions_filename, movieID_imdbID)
    weighting = 1 # indicating yes on weighting 
    [rec_diverseEmotionW, rec_itemEmotionW] = diversification.diversify_item_feature(candidates, item_emotions, item_ids, weighting)
        # rec_diverseEmotionW: ['item', 'discounted_score']
        # rec_itemEmotionW : ['0', '1', ..., 'num_emotions-1'], num_features = 8 here
 
    return rec_diverseEmotionW.item.to_numpy(), rec_diverseEmotionW
        # a np.ndarray of the recommended item_ids (matches movie_id in the movieLens dataset)


def get_features_for_viz(item_ids, feature_type):
    # item_ids: nd.array
    # feature_type: string either 'latent feature' or 'emotional signature'
    data_path = setpath.setpath()
    if feature_type == 'latent feature':
        model_filename = data_path + 'eRS_implictMF0.pkl'
        trained_MF_model = import_trained_model(model_filename)
        item_features = trained_MF_model.item_features_
        item_index = trained_MF_model.item_index_
        # feature_I =  item_features[:, 0]
        # feature_II =  item_features[:, 1]
        # print(feature_I)
        # print(feature_II)
        # print(type(item_index))
            # pandas.core.indexes.numeric.Int64Index
        # print(type(pd.Index(item_index.values)))
            # pandas.core.indexes.numeric.Int64Index
        item_two_features_df = pd.DataFrame({'item': item_index.values, 'feature1': item_features[:, 0], 'feature2': item_features[:, 1]}, columns = ['item', 'feature1', 'feature2'])
        rec_two_features_df = item_two_features_df[item_two_features_df['item'].isin(item_ids)]
        # print(rec_two_features_df)
        return rec_two_features_df
        
    elif feature_type == 'emotional signature':
        item_emotions_filename = data_path + 'eRS_emotions_IDlinked.csv'
        item_emotions = pd.read_csv(item_emotions_filename) 
            # ['imdb_id', 'movie_id', 'anticipation', 'joy', 'trust', 'anger', 'fear', 'disgust', 'sadness', 'surprise']
        item_emotions = item_emotions.rename({'movie_id' : 'item'}, axis = 1)
        rec_emotions = item_emotions[item_emotions['item'].isin(item_ids)]
        rec_emotions = rec_emotions.drop(columns=['imdb_id'])
            # ['item', 'anticipation', 'joy', 'trust', 'anger', 'fear', 'disgust', 'sadness', 'surprise']
        # print(rec_emotions)
        rec_emotions_ID_dropped = rec_emotions.drop(columns=['item'])
        rec_variance = rec_emotions_ID_dropped.std()
        rec_variance_sorted = rec_variance.sort_values(ascending = False)
            # pandas.core.series.Series
        largest_variance_emotions = rec_variance_sorted.index.values[:2]
        columns = np.insert(largest_variance_emotions, 0, 'item')
        print(columns)
        rec_two_emotions_df = rec_emotions[columns]
        # print(rec_two_emotions_df)
        return rec_two_emotions_df
        
    else:
        print("Wrong input of the feature type! Please either input \'latent feature\' or \'emotional signature\'")
        return 0

if __name__ == "__main__":

    data_path = setpath.setpath()
    testing_data_filename = data_path + 'testing_users_rating.csv'
    testing_users_ratings = pd.read_csv(testing_data_filename)
        # ['user', 'item', 'rating', 'timestamp']
    available_testing_users = testing_users_ratings.user.unique()
    
    print('Testing the eRS recommender now! \nHere are some dummy users who rated 20 movies each:  \n')
    print(available_testing_users)
    liveUserID = int(input("\n\nPlease Enter a user-ID to predict for: "))
    
    print("Recommended movie IDs: ")
    [rec_ids_topn, recs_topn]= topn(liveUserID)
    print(rec_ids_topn)
    
    [rec_ids_diverse_latent_feature, recs_latent_feature] = diversified_by_latent_feature(liveUserID)
    print(rec_ids_diverse_latent_feature)
    
    [rec_ids_diverse_weighted_latent_feature, recs_weighted_latent_feature] = diversified_by_weighted_latent_feature(liveUserID)
    print(rec_ids_diverse_weighted_latent_feature)

    [rec_ids_diverse_emotion, recs_emotion]= diversified_by_emotion(liveUserID)
    print(rec_ids_diverse_emotion)   
    
    [rec_ids_diverse_weighted_emotion, recs_weighted_emotion] = diversified_by_weighted_emotion(liveUserID)
    print(rec_ids_diverse_weighted_emotion)
    
    print('--------------Testing latent features----------')
    type_I = 'latent feature'
    type_II = 'emotional signature'
    test = get_features_for_viz(rec_ids_topn, type_I)
    print(test)
    print('--------------Testing emotional signature------')
    test = get_features_for_viz(rec_ids_topn, type_II)
    print(test)