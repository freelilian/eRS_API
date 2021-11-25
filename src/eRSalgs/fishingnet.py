import pandas as pd
import numpy as np
import itertools
import os
from scipy.spatial import distance
import time

def seeding(N, nb):
    ticks = []
    stop = 0
    ticks.append(stop)
    step = N / nb
    for i in range(nb):
        stop += step
        ticks.append(stop)
        
    return ticks
    # return ["{},{}".format(round(step*i), round(step*(i+1))) for i in range(nb)]

def fishingnet_diversification(diversification_candidates, cutoffCount = 50, numRecs = 80):
    # preds_aveScore_popularity = pd.merge(item_preds_ave_score, item_popularity, how = 'left', on = 'item')
    # diversification_candidates = preds_aveScore_popularity[preds_aveScore_popularity['count'] >= cutoffCount]
    
    # diversification_candidates: ['item', 'score', 'ave_score', 'count', 'rank']
    diversification_candidates.index = pd.Index(diversification_candidates['item'].values)
    candidate_vectors = diversification_candidates[['score', 'ave_score']].to_numpy()
    
    ticks = seeding(5, 12) # divide the 5 * 5 (predicted rating 0-5) grid into a 13*13 grid
    # print(ticks)
    coordinates = list(itertools.product(ticks,ticks))
    coordinates = np.asarray(coordinates, dtype=np.float64)
    
    diverse_items_index = []
    diverse_items_distance = []
    start = time.time()
    
    ## How to make it to be more effcient???
    for point in coordinates:
        distances_to_one_point = []
        for candidate in candidate_vectors:
            # dist = distance.euclidean(point, candidate)
            dist = distance.cityblock(point, candidate)
            distances_to_one_point.append(dist)
        distances_to_one_point_df = pd.DataFrame({'distance': distances_to_one_point})
        distances_to_one_point_df.index = diversification_candidates.index
        distances_to_one_point_df_sorted = distances_to_one_point_df.sort_values(by = 'distance', ascending = True)
        
        # check duplicate items, because multiple coordinate points can have the same closet item
        # still need to optimize the algorithm
        closest_index_one_point = distances_to_one_point_df_sorted.index[0]
        closest_distance_one_point = distances_to_one_point_df_sorted['distance'].to_numpy()[0]
        
        i = 0
        while (closest_index_one_point in diverse_items_index):
            i += 1
            closest_index_one_point = distances_to_one_point_df_sorted.index[i]
            closest_distance_one_point = distances_to_one_point_df_sorted['distance'].to_numpy()[i]
        diverse_items_index.append(closest_index_one_point)
        diverse_items_distance.append(closest_distance_one_point)
        '''

        diverse_items_index.append(closest_index_one_point)
        diverse_items_distance.append(closest_distance_one_point)
        '''
    
    # print(diverse_items_index)
    diverse_items_dist = pd.DataFrame({'item': diverse_items_index, 'distance': diverse_items_distance})
        # 169 points, correspond to 169 points in coordinates
    diverse_items_dist_sorted = diverse_items_dist.sort_values(by = 'distance', ascending = True)
    # print(diverse_items_dist_sorted.shape)
    diverse_80items_dist = diverse_items_dist_sorted.head(numRecs)
    # print(diverse_80items_dist.shape)
    diverse_80items = diversification_candidates[diversification_candidates['item'].isin(diverse_80items_dist.item)]
        # ['item', 'score', 'ave_score', 'count', 'rank']
    # print(diverse_80items.shape)    
    end = time.time()
    # print('Time spent: %.0f seconds.' % (end - start))
    
    return diverse_80items
    
    
def normalization(row, new_min, new_max, min, max):
    row['community'] = (new_max-new_min)*(row['ave_score'] - min)/(max - min) + new_min
    row['user'] = (new_max-new_min)*(row['score'] - min)/(max - min) + new_min
    return row    

def scaling_N_labeling(diverse_80items, new_min = 1, new_max = 5):
    ########--- Normalizing diverse_80items to [1, 5] ---######################
    # diverse_80items: ['item', 'score', 'ave_score', 'count', 'rank']
    # formula of scaling down a range of numbers with a known minimum and maximum value [a, b]
    # (b-a)(ndarray - min)/(max - min) + a
    
    # print(diverse_80items.head(10))
    min = np.min([np.min(diverse_80items['ave_score']), np.min(diverse_80items['score'])])
    max = np.max([np.max(diverse_80items['ave_score']), np.max(diverse_80items['score'])])
    # ints are getting upcasted into floats when calling the apply() function
    diverse_80items_rescared = diverse_80items.apply(normalization, args = (new_min, new_max, min, max), axis = 1)
        # https://www.kite.com/python/answers/how-to-apply-a-function-with-multiple-arguments-to-a-pandas-dataframe-in-python

    ########--- Label items for viz on diverse_80items ---######################
    global_average = np.mean([np.median(diverse_80items_rescared['community']), np.median(diverse_80items_rescared['user'])])
    # diverse_80items_rescared_labeled = diverse_80items_rescared.apply(labeling, args = (global_average), axis = 1)
        # error： labeling() argument after * must be an iterable, not numpy.float64
    
    def labeling(row):
        row['label_community'] = 1 if row['community'] >= global_average else 0
        row['label_user'] = 1 if row['user'] >= global_average else 0
        return row
    diverse_80items_rescared_labeled = diverse_80items_rescared.apply(labeling, axis = 1)
        
    diverse_80items_rescared_labeled = diverse_80items_rescared_labeled.astype({'item': 'int64', 'count': 'int64', 'rank': 'int64', 'label_community': 'int64', 'label_user': 'int64'})
    ave_community_rescared_score = np.mean(diverse_80items_rescared_labeled['community'])
    ave_user_rescared_score = np.mean(diverse_80items_rescared_labeled['user'])
    
    return diverse_80items_rescared_labeled, ave_community_rescared_score, ave_user_rescared_score


if __name__ == '__main__':
    
    date_file = os.path.join(os.path.dirname(__file__), 'data/predictions_for_Bart_with_averaged_item_scores_explicitMF.csv')
    item_preds_ave_score = pd.read_csv(date_file)

    data_path = os.path.join(os.path.dirname(__file__), 'data/item_popularity.csv')    
    item_popularity = pd.read_csv(data_path) 
   
    diverse_80items = fishingnet_diversification(item_preds_ave_score, item_popularity)
  
    new_min = 1
    new_max = 5
    [diverse_80items_rescared, ave_community_rescared_score, ave_user_rescared_score] = scaling_N_labeling(diverse_80items, new_min, new_max)
    
    print('Average predicted score of community: ', ave_community_rescared_score)
    print('Your average predicted score: ', ave_user_rescared_score)
    print(diverse_80items_rescared)
    
    # saved_as = os.path.join(os.path.dirname(__file__), 'toShow/diverse_items_for_Bart_cutoff_at_50ratings_fishing_net_cityblock_v1.csv')
    # diverse_80items_rescared.to_csv(saved_as, index = False)