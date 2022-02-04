"""
compute.py
"""
from typing import List
from models import Rating, Recommendation, Preference, LatentFeature, EmotionalSignature

import eRSalgs.eRS_recommender as eRS
import eRSalgs.diversification as div
import eRSalgs.setpath as setpath
import eRSalgs.eRS_recommender as eRS_recommender
import pandas as pd
import numpy as np
import os

def get_RSSA_preds(ratings: List[Rating], user_id) -> pd.DataFrame:
    rated_items = np.array([np.int64(rating.movielensId) for rating in ratings])
    new_ratings = pd.Series(np.array([np.float64(rating.rating) for rating in ratings]), index = rated_items)    

    data_path = os.path.join(os.path.dirname(__file__), './eRSalgs/data/eRS_item_popularity.csv')
    eRS_item_popularity = pd.read_csv(data_path) 
    
    model_path = os.path.join(os.path.dirname(__file__), './eRSalgs/data/eRS_implictMF.pkl')
    trained_MF_model = eRS.import_trained_model(model_path)

    ## predicting
    [RSSA_preds, liveUser_feature] = eRS.live_prediction(trained_MF_model, user_id, new_ratings, eRS_item_popularity)
        # ['item', 'score', 'count', 'rank', 'discounted_score']
        # liveUser_feature: np.ndarray
    # extract the not-rated-yet items
    RSSA_preds_of_noRatedItems = RSSA_preds[~RSSA_preds['item'].isin(rated_items)]
        # ['item', 'score', 'count', 'rank', 'discounted_score']
    
    return RSSA_preds_of_noRatedItems, trained_MF_model
        # return trained_MF_model since it will be needed in the diversification to extract the latent features

def get_features_for_viz(recommendations: List[Recommendation], feature_type: str):
    # recommendations: nd.array
    # feature_type: string either 'latent feature' or 'emotional signature'
    data_path = setpath.setpath()
    # item_ids = np.asarray(recommendations)
    item_ids = np.array([np.int64(recommendation.movielensId) for recommendation in recommendations])
    item_ids = item_ids.astype(int)
        
    # print(type(item_ids))
    if feature_type == 'latent feature':
        model_filename = data_path + 'eRS_implictMF.pkl'
        trained_MF_model = eRS_recommender.import_trained_model(model_filename)
        item_features = trained_MF_model.item_features_
        item_index = trained_MF_model.item_index_
        item_two_features_df = pd.DataFrame({'item': item_index.values, 'feature1': item_features[:, 0], 'feature2': item_features[:, 1]}, columns = ['item', 'feature1', 'feature2'])
        rec_two_features_df = item_two_features_df[item_two_features_df['item'].isin(item_ids)]
        # print(rec_two_features_df)
            # ['item', 'feature1', 'feature2']
        
        viz_values = []
        for index, row in rec_two_features_df.iterrows():
            viz_values.append(LatentFeature(str(np.int64(row['item'])), row['feature1'], row['feature2']))
    
        return viz_values
        
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
        # print(largest_variance_emotions)
            # ['(emotion1)', '(emotion1)'], print a list of emotion names, 2 elements
        rec_two_emotions_df = rec_emotions[largest_variance_emotions]
        # print(rec_two_emotions_df)
            # ['item', (emotion1), (emotion1)]
        columns = np.insert(largest_variance_emotions, 0, 'item')
        rec_two_emotions_df = rec_emotions[columns]
        # print(rec_two_emotions_df.columns)
     
        viz_values = []
        for index, row in rec_two_emotions_df.iterrows():
            viz_values.append(EmotionalSignature(str(np.int64(row['item'])), largest_variance_emotions[0], row.iloc[1], largest_variance_emotions[1], row.iloc[2]))
    
        return viz_values
        
    else:
        print("Wrong input of the feature type! Please either input \'latent feature\' or \'emotional signature\'")
        return 0
        
        
def predict_user_topN(ratings: List[Rating], user_id) -> List[str]:
    [RSSA_preds_of_noRatedItems, _] = get_RSSA_preds(ratings, user_id)
    # traditional_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
        # only needed when using the traditional predictions
    discounted_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'discounted_score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score']
        
    numRec = 10         
    recs_topN_discounted = discounted_preds_sorted.head(numRec)    
    
    recommendations = []
    for index, row in recs_topN_discounted.iterrows():
        # recommendations.append(Preference(str(np.int64(row['item'])), 'top_n'))
        recommendations.append(Recommendation(str(np.int64(row['item']))))
        
    return recommendations
    # *** Also needs to return vectors of the recommendations for viz
    
    
def predict_items_diversified_by_latent_feature(ratings: List[Rating], user_id) -> List[str]:
    [RSSA_preds_of_noRatedItems, trained_MF_model] = get_RSSA_preds(ratings, user_id)
    # traditional_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
        # only needed when using the traditional predictions
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
    numRec = 10     
    weighting = 0
    [rec_diverseFeature, rec_itemFeature] = div.diversify_item_feature(candidates, item_features, item_index.values, weighting, numRec)
        # rec_diverseFeature: ['item', 'discounted_score']
        # rec_itemFeature : ['0', '1', ..., 'num_features-1'], num_features = 20 here
    
    recommendations = []
    for index, row in rec_diverseFeature.iterrows():
        # recommendations.append(Preference(str(np.int64(row['item'])), 'top_n'))
        recommendations.append(Recommendation(str(np.int64(row['item']))))
        
    return recommendations
    # *** Also needs to return vectors of the recommendations for viz

def predict_items_diversified_by_weighted_latent_feature(ratings: List[Rating], user_id) -> List[str]:
    [RSSA_preds_of_noRatedItems, trained_MF_model] = get_RSSA_preds(ratings, user_id)
    # traditional_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
        # only needed when using the traditional predictions
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
        
    numRec = 10 
    weighting = 1
    [rec_diverseFeatureW, rec_itemFeatureW] = div.diversify_item_feature(candidates, item_features, item_index.values, weighting, numRec)
        # rec_diverseFeatureW: ['item', 'discounted_score']
        # rec_itemFeatureW: ['0', '1', ..., 'num_features-1'], num_features = 20 here
    
    recommendations = []
    for index, row in rec_diverseFeatureW.iterrows():
        # recommendations.append(Preference(str(np.int64(row['item'])), 'top_n'))
        recommendations.append(Recommendation(str(np.int64(row['item']))))
        
    return recommendations
    # *** Also needs to return vectors of the recommendations for viz

def predict_items_diversified_by_emotion(ratings: List[Rating], user_id) -> List[str]:
    [RSSA_preds_of_noRatedItems, _] = get_RSSA_preds(ratings, user_id)
    # traditional_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
        # only needed when using the traditional predictions
    discounted_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'discounted_score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
    
    ## diversified by emotion
    num_topN = 200
    candidates = discounted_preds_sorted.head(num_topN)[['item', 'discounted_score']]
    ## recommendation
    item_emotions_filename = os.path.join(os.path.dirname(__file__), './eRSalgs/data/eRS_emotions_IDlinked.csv')
    item_emotions = pd.read_csv(item_emotions_filename) 
        # ['imdb_id', 'movie_id', 'anticipation', 'joy', 'trust', 'anger', 'fear', 'disgust', 'sadness', 'surprise'
    item_emotions = item_emotions.rename({'movie_id' : 'item'}, axis = 1)
    item_emotions_ndarray = item_emotions[['anticipation', 'joy', 'trust', 'anger', 'fear', 'disgust', 'sadness', 'surprise']].to_numpy()
        # np.ndarray of 8 columns matches ['anticipation', 'joy', 'trust', 'anger', 'fear', 'disgust', 'sadness', 'surprise']
    item_ids = item_emotions.item.unique()
    
    numRec = 10         
    weighting = 0
    [rec_diverseEmotion, rec_itemEmotion] = div.diversify_item_feature(candidates, item_emotions_ndarray, item_ids, weighting, numRec)
        # rec_diverseEmotion: ['item', 'discounted_score']
        # rec_itemEmotion : ['anticipation', 'joy', 'trust', 'anger', 'fear', 'disgust', 'sadness', 'surprise']
    
    recommendations = []
    for index, row in rec_diverseEmotion.iterrows():
        # recommendations.append(Preference(str(np.int64(row['item'])), 'top_n'))
        recommendations.append(Recommendation(str(np.int64(row['item']))))
 
    return recommendations
    # *** Also needs to return vectors of the recommendations for viz

def predict_items_diversified_by_weighted_emotion(ratings: List[Rating], user_id) -> List[str]:
    [RSSA_preds_of_noRatedItems, _] = get_RSSA_preds(ratings, user_id)
    # traditional_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
        # only needed when using the traditional predictions
    discounted_preds_sorted = RSSA_preds_of_noRatedItems.sort_values(by = 'discounted_score', ascending = False)
        # ['item', 'score', 'count', 'rank', 'discounted_score']  
    
    ## diversified by emotion
    num_topN = 200
    candidates = discounted_preds_sorted.head(num_topN)[['item', 'discounted_score']]
    ## recommendation
    item_emotions_filename = os.path.join(os.path.dirname(__file__), './eRSalgs/data/eRS_emotions_IDlinked.csv')
    item_emotions = pd.read_csv(item_emotions_filename) 
        # ['imdb_id', 'movie_id', 'anticipation', 'joy', 'trust', 'anger', 'fear', 'disgust', 'sadness', 'surprise'
    item_emotions = item_emotions.rename({'movie_id' : 'item'}, axis = 1)
    item_emotions_ndarray = item_emotions[['anticipation', 'joy', 'trust', 'anger', 'fear', 'disgust', 'sadness', 'surprise']].to_numpy()
        # np.ndarray of 8 columns matches ['anticipation', 'joy', 'trust', 'anger', 'fear', 'disgust', 'sadness', 'surprise']
    item_ids = item_emotions.item.unique()
    
    numRec = 10    
    weighting = 1
    [rec_diverseEmotion, rec_itemEmotion] = div.diversify_item_feature(candidates, item_emotions_ndarray, item_ids, weighting, numRec)
        # rec_diverseEmotion: ['item', 'discounted_score']
        # rec_itemEmotion : ['anticipation', 'joy', 'trust', 'anger', 'fear', 'disgust', 'sadness', 'surprise']
    
    recommendations = []
    for index, row in rec_diverseEmotion.iterrows():
        # recommendations.append(Preference(str(np.int64(row['item'])), 'top_n'))
        recommendations.append(Recommendation(str(np.int64(row['item']))))
 
    return recommendations
    # *** Also needs to return vectors of the recommendations for viz
  
  
    
if __name__ == '__main__':

    fullpath_test = os.path.join(os.path.dirname(__file__), './eRSalgs/testing_rating_rated_items_extracted/ratings_set6_rated_only_Shahan.csv')
    liveUserID = 'Bart'
    ratings_liveUser = pd.read_csv(fullpath_test, encoding='latin1')
    #print(ratings_liveUser.head(20))
    
    ratings = []
    for index, row in ratings_liveUser.iterrows():
        ratings.append(Rating(row['item'], row['rating']))
    

    ## starting calling the methods
    recommendations = predict_user_topN(ratings, liveUserID)
    print('1. Traditional top-N recommendations')
    print(recommendations)
    latent_features = get_features_for_viz(recommendations, 'latent feature')
    print(latent_features)
    emotions = get_features_for_viz(recommendations, 'emotional signature')
    print(emotions)
    print()
    
    recommendations = predict_items_diversified_by_latent_feature(ratings, liveUserID)
    print('2. Diversified by latent features')
    print(recommendations)
    latent_features = get_features_for_viz(recommendations, 'latent feature')
    print(latent_features)    
    print()
    
    recommendations = predict_items_diversified_by_weighted_latent_feature(ratings, liveUserID)
    print('3. Diversified by weithged latent features')
    print(recommendations)
    latent_features = get_features_for_viz(recommendations, 'latent feature')
    print(latent_features)    
    print()
    
    recommendations = predict_items_diversified_by_emotion(ratings, liveUserID)
    print('4. Diversified by emotions')
    print(recommendations)
    emotions = get_features_for_viz(recommendations, 'emotional signature')
    print(emotions)    
    print()
    
    recommendations = predict_items_diversified_by_weighted_emotion(ratings, liveUserID)
    print('5. Diversified by weithged emotions')
    print(recommendations)
    emotions = get_features_for_viz(recommendations, 'emotional signature')
    print(emotions)    
    print()
    '''
    RSSA_team = ['Bart', 'Sushmita', 'Shahan', 'Aru', 'Mitali', 'Yash']
    for liveUserID in RSSA_team:
        fullpath_test = os.path.join(os.path.dirname(__file__), 'eRSalgs/testing_rating_rated_items_extracted/ratings_set6_rated_only_' + liveUserID + '.csv')
        ratings_liveUser = pd.read_csv(fullpath_test, encoding='latin1')
        ratings = []
        for index, row in ratings_liveUser.iterrows():
            ratings.append(Rating(row['item'], row['rating']))
        recommendations = predict_user_topN(ratings, liveUserID)
        for rec in recommendations:
            print(rec.movielensId, end = ', ')
        print()
    '''
    
    