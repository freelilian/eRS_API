"""

app.py

Lijie Guo
Clemson University
10/27/2021

Server for running the recommender algorithms. See
`models.py` for information about the input and
outputs.

"""

from pathlib import Path
import json

from flask import Flask, abort
from flask import request
from flask import render_template

from compute import predict_user_topN
from compute import predict_items_diversified_by_latent_feature
from compute import predict_items_diversified_by_weighted_latent_feature
from compute import predict_items_diversified_by_emotion
from compute import predict_items_diversified_by_weighted_emotion

from compute import get_features_for_viz

from models import Rating

app = Flask(__name__)


@app.route('/')
def show_readme():
    return render_template('README.html')

@app.route('/preferences', methods=['POST'])
def predict_preferences():
    req = request.json
    ratings = None

    try:
        ratings = req['ratings']
    except KeyError:
        abort(400)
        
    ratings = [Rating(**rating) for rating in ratings]    

    rec_funcs = {
        'top_N': predict_user_topN,
        'diversified_by_latent_feature': predict_items_diversified_by_latent_feature,
        'diversified_by_weighted_latent_feature': predict_items_diversified_by_weighted_latent_feature,
        'diversified_by_emotions': predict_items_diversified_by_emotion,
        'diversified_by_weighted_emotions': predict_items_diversified_by_weighted_emotion
    }
    recommendations = {k: f(ratings=ratings, user_id=0) for k, f in rec_funcs.items()}
        
    return dict(preferences=recommendations)

@app.route('/latent_features', methods=['POST'])
def predict_latent_features():
    req = request.json
    ratings = None

    try:
        ratings = req['ratings']
    except KeyError:
        abort(400)
        
    ratings = [Rating(**rating) for rating in ratings]    

    funcs_need_latent_features = {
        'top_N': predict_user_topN,
        'diversified_by_latent_feature': predict_items_diversified_by_latent_feature,
        'diversified_by_weighted_latent_feature': predict_items_diversified_by_weighted_latent_feature,
    }
    Two_latent_features = {k: get_features_for_viz(f(ratings=ratings, user_id=0), 'latent feature') for k, f in funcs_need_latent_features.items()}
    
    return dict(latent_features=Two_latent_features)
    
@app.route('/emotions', methods=['POST'])
def predict_emotions():
    req = request.json
    ratings = None

    try:
        ratings = req['ratings']
    except KeyError:
        abort(400)
        
    ratings = [Rating(**rating) for rating in ratings]    

    funcs_need_emotions = {
        'top_N': predict_user_topN,
        'diversified_by_emotions': predict_items_diversified_by_emotion,
        'diversified_by_weighted_emotions': predict_items_diversified_by_weighted_emotion
    }
    Two_emotions = {k: get_features_for_viz(f(ratings=ratings, user_id=0), 'emotional signature') for k, f in funcs_need_emotions.items()}
    
    return dict(emotions=Two_emotions)



if __name__ == '__main__':
    config_path = Path(__file__).parent / 'config.json'
    with open(config_path) as f:
        settings = json.load(f)
    app.run(port=settings['port'],
            host=settings['host'],
            debug=settings['debug'])
