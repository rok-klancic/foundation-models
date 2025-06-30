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
from time_moe_covariates_scripts import time_moe_stat

# Import time
import time


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

def save_results(type, model_name, r2_scores, predictions, best_features, best_params, file_path, time_string):
    # Create a dictionary to store the results
    results = {
        'type': type,
        'model_name': model_name,
        'r2_scores': r2_scores,
        'predictions': predictions,
        'best_features': best_features,
        'best_params': best_params,
        'total_time': time_string
    }

    with open(file_path, 'w') as file:
        json.dump(results, file, indent=4)


# EXPERIMENT SETTINGS
# ----------------------------------------------------------------------------------------------------------------------
# Load the experiment settings
with open('experiment_settings/time_moe_covariate_models_experiment_settings.json', 'r') as file:
    experiment_settings = json.load(file)

# TESTING THE MODELS
# ----------------------------------------------------------------------------------------------------------------------
for name, settings in experiment_settings.items():
    # Time the experiment
    start_time = time.time()

    # Get the model name
    model_name = experiment_settings[name]['model_name']

    if name == 'time_moe + statistical':
        # Load the data
        aquifer_by_stations = joblib.load('../../data/interim/ground-water-and-weather-with-weather-and-timemoe-forecasts-and-additional-features.joblib')
        # Time-MoE forecast features (diff_0 means that the forecast is for the next day and so on)
        time_moe_forecast_features = ['forecast_altitude_diff_0',
                                      'forecast_altitude_diff_1',
                                      'forecast_altitude_diff_2',
                                      'forecast_altitude_diff_3',
                                      'forecast_altitude_diff_4']
    
        # Feature selection
        best_features = time_moe_stat.k_best_feature_selection(aquifers_list=settings['aquifers_list'],
                                                     test_len=settings['test_len'], 
                                                     horizon_max=settings['horizon_max'],
                                                     target_feature=settings['target_feature'], 
                                                     aquifer_by_stations=aquifer_by_stations,
                                                     k=settings['k'])
        
        # Hyperparameter tuning
        best_params = time_moe_stat.hyperparameter_tuning(model_name=model_name,
                                                          horizon_max=settings['horizon_max'], 
                                                          aquifers_list=settings['aquifers_list'], 
                                                          best_features=best_features, 
                                                          target_feature=settings['target_feature'], 
                                                          test_len=settings['test_len'], 
                                                          val_len=settings['val_len'], 
                                                          aquifer_by_stations=aquifer_by_stations,
                                                          time_moe_forecast_features=time_moe_forecast_features)
        
        # Final training
        r2_average, r2_scores, predictions = time_moe_stat.final_training(model_name=model_name,
                                                                          aquifers_list=settings['aquifers_list'], 
                                                                          test_len=settings['test_len'], 
                                                                          horizon_max=settings['horizon_max'],
                                                                          target_feature=settings['target_feature'],
                                                                          aquifer_by_stations=aquifer_by_stations,
                                                                          best_features=best_features,
                                                                          best_params=best_params,
                                                                          time_moe_forecast_features=time_moe_forecast_features)
        
        # Define the name for the results file
        saving_name = f'{name}_{model_name}'

    elif name == 'statistical + time_moe':
        # Call the script that uses this setup
        pass
    elif name == 'linear_regression + Time-MoE + linear_regression':
        # Call the script that uses this setup
        pass
    elif name == 'hidden_layer + linear_regression':
        # Call the script that uses this setup
        pass
    elif name == 'hidden_layer + MLP':
        # Call the script that uses this setup
        pass
    elif name == 'hidden_layer + NBEATSx':
        # Call the script that uses this setup
        pass
    else:
        raise ValueError(f"Model {name} not found")
    
    # Time the experimnt
    end_time = time.time()
    total_time = end_time - start_time
    # Convert total_time to hours, minutes, and seconds
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    # Put into string
    total_time_string = f"{hours}h {minutes}m {seconds}s"

    # Obtain the index of the file name (so every experiment has a unique name)
    index = get_index(folder_path='../results/time_moe_covariate_models', file_name=saving_name)

    # Save the results
    file_path = f'../results/time_moe_covariate_models/{saving_name}_{index}.json'
    save_results(type=name,
                 model_name=model_name,
                 r2_scores=r2_scores, 
                 predictions=predictions, 
                 best_features=best_features, 
                 best_params=best_params, 
                 file_path=file_path,
                 time_string=total_time_string)


    # Print the results
    print("--------------------------------------------------------------------------------------------------")
    print(f"Model: {name}")
    print(f"R2 average: {r2_average}")
    print(f"Total execution time: {total_time_string}")
    print("--------------------------------------------------------------------------------------------------\n\n")