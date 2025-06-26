import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from transformers import AutoModelForCausalLM

import joblib

from sklearn.metrics import r2_score

# Linear regression for the multivariate model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# NBEATSx
from neuralforecast.models import NBEATSx
from neuralforecast.core import NeuralForecast

# Warnings
import warnings
warnings.filterwarnings('ignore')

# json
import json

# Import functions for time_moe + statistical
from time_moe_covariates_scripts.time_moe__linear_regression import *






# SAVING THE RESULTS
# ----------------------------------------------------------------------------------------------------------------------
def get_index(folder_path, file_name):
    top_index = 0
    for file in os.listdir(folder_path):
        if file.endswith('.json') and file.startswith(file_name):
            whole_name = file.split('.')[0]
            if '_' not in whole_name:
                continue
            else:
                index = whole_name.split('_')[-1]
                if index.isdigit():
                    index = int(index)
                    if index > top_index:
                        top_index = index

    return top_index+1

def save_results(model_name, multivariate, r2_scores, predictions, feature_selection, best_features, best_params, file_path):
    # Create a dictionary to store the results
    results = {
        'model_name': model_name,
        'multivariate': multivariate,
        'r2_scores': r2_scores,
        'predictions': predictions,
        'feature_selection': feature_selection,
        'best_features': best_features,
        'best_params': best_params
    }

    with open(file_path, 'w') as file:
        json.dump(results, file, indent=4)


# EXPERIMENT SETTINGS
# ----------------------------------------------------------------------------------------------------------------------
# Load the experiment settings
with open('experiment_settings/statistical_models_experiment_settings.json', 'r') as file:
    experiment_settings = json.load(file)

# Load the data
aquifer_by_stations = joblib.load('../../data/interim/ground-water-and-weather-with-forecasts-and-additional-features.joblib')

# Transform date column to year, month and day columns
for key in aquifer_by_stations.keys():
    aquifer_by_stations[key]['year'] = aquifer_by_stations[key]['date'].dt.year
    aquifer_by_stations[key]['month'] = aquifer_by_stations[key]['date'].dt.month
    aquifer_by_stations[key]['day'] = aquifer_by_stations[key]['date'].dt.day


# TESTING THE MODELS
# ----------------------------------------------------------------------------------------------------------------------
for name, settings in experiment_settings.items():
    # Get the model name
    model_name = experiment_settings[name]['model_name']

    if model_name == 'time_moe + linear_regression':
        # Call the script that uses this setup
        pass
    elif model_name == 'linear_regression + time_moe':
        # Call the script that uses this setup
        pass
    elif model_name == 'linear_regression + Time-MoE + linear_regression':
        # Call the script that uses this setup
        pass
    elif model_name == 'hidden_layer + linear_regression':
        # Call the script that uses this setup
        pass
    elif model_name == 'hidden_layer + MLP':
        # Call the script that uses this setup
        pass
    elif model_name == 'hidden_layer + NBEATSx':
        # Call the script that uses this setup
        pass
    else:
        

    # Check if we need additional features
    if settings['additional_features']:
        # Set the multivariate variable to True
        multivariate = True

        if settings['feature_selection'] == 'ga':
            # Feature selection
            best_features = ga_feature_selection(model_name, settings['feature_selection_aquifer'], 
                                            settings['test_len'], settings['val_len'], 
                                            settings['horizon_max'], settings['target_feature'], 
                                            aquifer_by_stations)
        else:
            # Feature selection
            best_features = k_best_feature_selection(aquifers_list=settings['aquifers_list'],
                                                     test_len=settings['test_len'], 
                                                     horizon_max=settings['horizon_max'],
                                                     target_feature=settings['target_feature'], 
                                                     aquifer_by_stations=aquifer_by_stations,
                                                     k=settings['k'])
    else:
        best_features = {}
        for horizon in range(1, settings['horizon_max'] + 1):
            best_features[f'horizon_{horizon}'] = ['altitude_diff']

    # Hyperparameter tuning
    if settings['hyperparameter_tuning']:
        best_params = hyperparameter_tuning(model_name, settings['horizon_max'], 
                                        settings['aquifers_list'], best_features, 
                                        settings['target_feature'], settings['test_len'], 
                                        settings['val_len'], aquifer_by_stations)
    else:
        best_params = {}
        
    # Final training
    r2_average, r2_scores, predictions = final_training(model_name, settings['aquifers_list'], 
                                settings['test_len'], settings['horizon_max'], 
                                settings['target_feature'], aquifer_by_stations,
                                best_features, best_params)
    
    # Obtain the index of the file name (so every experiment has a unique name)
    index = get_index(folder_path='../results/statistical_models', file_name=name)

    # Save the results
    file_path = f'../results/statistical_models/{name}_{index}.json'
    save_results(model_name=model_name, 
                 multivariate=multivariate, 
                 r2_scores=r2_scores, 
                 predictions=predictions, 
                 feature_selection=settings['feature_selection'], 
                 best_features=best_features, 
                 best_params=best_params, 
                 file_path=file_path)

    # Print the results
    print("--------------------------------------------------------------------------------------------------")
    print(f"Model: {name}")
    print(f"R2 average: {r2_average}")
    print("--------------------------------------------------------------------------------------------------\n\n")