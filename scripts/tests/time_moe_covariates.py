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
from time_moe_covariates_scripts import time_moe_stat, stat_time_moe, stat_timeMoe_stat, hidden_statistical, hidden_mlp, hidden_deep_learning

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

def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]
    else:
        return obj
    
def convert_numpy_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

def save_results(type, model_name, r2_scores, predictions, best_features, best_params, file_path, time_string):
    # Ensure lists
    predictions = convert_numpy_to_native(convert_ndarray_to_list(predictions))
    r2_scores = convert_numpy_to_native(convert_ndarray_to_list(r2_scores))
    best_features = convert_numpy_to_native(convert_ndarray_to_list(best_features))
    best_params = convert_numpy_to_native(convert_ndarray_to_list(best_params))

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
                                                                          val_len=settings['val_len'],
                                                                          horizon_max=settings['horizon_max'],
                                                                          target_feature=settings['target_feature'],
                                                                          aquifer_by_stations=aquifer_by_stations,
                                                                          best_features=best_features,
                                                                          best_params=best_params,
                                                                          time_moe_forecast_features=time_moe_forecast_features)
        
        # Define the name for the results file
        saving_name = f'{name}_{model_name}'

    elif name == 'statistical + time_moe':
        # Load the data
        aquifer_by_stations = joblib.load('../../data/interim/ground-water-and-weather-with-forecasts-and-additional-features.joblib')
        
        # Feature selection
        best_features = stat_time_moe.k_best_feature_selection(aquifers_list=settings['aquifers_list'],
                                                               test_len=settings['test_len'], 
                                                               horizon_max=settings['horizon_max'],
                                                               target_feature=settings['target_feature'], 
                                                               aquifer_by_stations=aquifer_by_stations,
                                                               k=settings['k'])
        
        # Hyperparameter tuning
        if model_name not in ['linear_regression']:
            best_params = stat_time_moe.hyperparameter_tuning(model_name=model_name,
                                                              horizon_max=settings['horizon_max'], 
                                                              aquifers_list=settings['aquifers_list'], 
                                                              best_features=best_features, 
                                                              target_feature=settings['target_feature'], 
                                                              test_len=settings['test_len'], 
                                                              val_len=settings['val_len'],
                                                              aquifer_by_stations=aquifer_by_stations)
        else:
            best_params = None

        # Getting the residuals of the statistical models
        residuals, statistical_predictions = stat_time_moe.final_training_statistical(model_name=model_name,
                                                             aquifers_list=settings['aquifers_list'],
                                                             test_len=settings['test_len'],
                                                             val_len=settings['val_len'],
                                                             horizon_max=settings['horizon_max'],
                                                             target_feature=settings['target_feature'],
                                                             aquifer_by_stations=aquifer_by_stations,
                                                             best_features=best_features,
                                                             best_params=best_params)
        
        # Final training
        r2_average, r2_scores, predictions = stat_time_moe.final_training_time_moe(horizon_max=settings['horizon_max'], 
                                                                                              aquifers_list=settings['aquifers_list'], 
                                                                                              test_len=settings['test_len'], 
                                                                                              residuals=residuals, 
                                                                                              context_length=settings['context_length'], 
                                                                                              aquifer_by_stations=aquifer_by_stations, 
                                                                                              statisctical_predictions=statistical_predictions)
        # Define the name for the results file
        saving_name = f'{name}_{model_name}'

    elif name == 'statistical + Time-MoE + statistical':
        # Load the data
        aquifer_by_stations = joblib.load('../../data/interim/ground-water-and-weather-with-forecasts-and-additional-features.joblib')

        # Feature selection
        best_features = stat_timeMoe_stat.k_best_feature_selection(aquifers_list=settings['aquifers_list'],
                                                               test_len=settings['test_len'], 
                                                               horizon_max=settings['horizon_max'],
                                                               target_feature=settings['target_feature'], 
                                                               aquifer_by_stations=aquifer_by_stations,
                                                               k=settings['k'])
        
        # Initial hyperparameter tuning
        if model_name not in ['linear_regression']:
            best_params_init  = stat_timeMoe_stat.hyperparameter_tuning_initial(model_name=model_name,
                                                                                horizon_max=settings['horizon_max'],
                                                                                aquifers_list=settings['aquifers_list'],
                                                                                best_features=best_features,
                                                                                target_feature=settings['target_feature'],
                                                                                test_len=settings['test_len'],
                                                                                val_len=settings['val_len'],
                                                                                aquifer_by_stations=aquifer_by_stations)
        else:
            best_params_init = None

        # Initial predictions
        statistical_predictions = stat_timeMoe_stat.prediction_initial(model_name=model_name,
                                                                       aquifers_list=settings['aquifers_list'],
                                                                       test_len=settings['test_len'],
                                                                       val_len=settings['val_len'],
                                                                       horizon_max=settings['horizon_max'],
                                                                       target_feature=settings['target_feature'],
                                                                       aquifer_by_stations=aquifer_by_stations,
                                                                       best_features=best_features,
                                                                       best_params=best_params_init)
        # Load the Time-MoE predictions
        time_moe_predictions = joblib.load('../../data/interim/ground-water-and-weather-with-weather-and-timemoe-forecasts-and-additional-features.joblib')
        
        # Time-MoE forecast features
        time_moe_forecast_features = ['forecast_altitude_diff_0',
                                      'forecast_altitude_diff_1',
                                      'forecast_altitude_diff_2',
                                      'forecast_altitude_diff_3',
                                      'forecast_altitude_diff_4']
        # Final hyperparameter tuning
        if model_name not in ['linear_regression']:
            best_params_final = stat_timeMoe_stat.hyperparameter_tuning_final(model_name=model_name,
                                                                          horizon_max=settings['horizon_max'],
                                                                          aquifers_list=settings['aquifers_list'],
                                                                          target_feature=settings['target_feature'],
                                                                          test_len=settings['test_len'],
                                                                          val_len=settings['val_len'],
                                                                          aquifer_by_stations=time_moe_predictions,
                                                                          statistical_predictions=statistical_predictions,
                                                                          time_moe_forecast_features=time_moe_forecast_features)
        else:
            best_params_final = None               
        # Final training
        r2_average, r2_scores, predictions = stat_timeMoe_stat.final_training(model_name=model_name,
                                                                                          aquifers_list=settings['aquifers_list'],
                                                                                          test_len=settings['test_len'],
                                                                                          val_len=settings['val_len'],
                                                                                          horizon_max=settings['horizon_max'],
                                                                                          target_feature=settings['target_feature'],
                                                                                          aquifer_by_stations=time_moe_predictions,
                                                                                          statistical_predictions=statistical_predictions,
                                                                                          best_params=best_params_final,
                                                                                          time_moe_forecast_features=time_moe_forecast_features)
        # Define the name for the results file
        saving_name = f'{name}_{model_name}'
        # Fix the name of the best_params_final for the saving
        best_params = best_params_final

    elif name == 'hidden_layer + statistical':
        # Load the data
        aquifer_by_stations = joblib.load('../../data/interim/ground-water-and-weather-with-forecasts-and-additional-features.joblib')
        time_moe_outputs = joblib.load('../../data/interim/time_moe_outputs.joblib')

        # Feature selection
        best_features = hidden_statistical.k_best_feature_selection(aquifers_list=settings['aquifers_list'],
                                                                   test_len=settings['test_len'],
                                                                   horizon_max=settings['horizon_max'],
                                                                   target_feature=settings['target_feature'],
                                                                   aquifer_by_stations=aquifer_by_stations,
                                                                   k=settings['k'])
        
        # Hyperparameter tuning
        if model_name not in ['linear_regression']:
            best_params = hidden_statistical.hyperparameter_tuning(model_name=model_name,
                                                                horizon_max=settings['horizon_max'],
                                                                aquifers_list=settings['aquifers_list'],
                                                                best_features=best_features,
                                                                target_feature=settings['target_feature'],
                                                                test_len=settings['test_len'],
                                                                val_len=settings['val_len'],
                                                                aquifer_by_stations=aquifer_by_stations,
                                                                time_moe_outputs=time_moe_outputs)
        else:
            best_params = None
        
        # Final training
        r2_average, r2_scores, predictions = hidden_statistical.final_training(model_name=model_name,
                                                                              aquifers_list=settings['aquifers_list'],
                                                                              test_len=settings['test_len'],
                                                                              val_len=settings['val_len'],
                                                                              horizon_max=settings['horizon_max'],
                                                                              target_feature=settings['target_feature'],
                                                                              aquifer_by_stations=aquifer_by_stations,
                                                                              best_features=best_features,
                                                                              best_params=best_params,
                                                                              time_moe_outputs=time_moe_outputs)
        
        # Define the name for the results file
        saving_name = f'{name}_{model_name}'

    elif name == 'hidden_layer + MLP':
        # Load the data
        aquifer_by_stations = joblib.load('../../data/interim/ground-water-and-weather-with-forecasts-and-additional-features.joblib')
        time_moe_outputs = joblib.load('../../data/interim/time_moe_outputs.joblib')
        
        # Feature selection
        best_features = hidden_mlp.k_best_feature_selection(aquifers_list=settings['aquifers_list'],
                                                                   test_len=settings['test_len'],
                                                                   horizon_max=settings['horizon_max'],
                                                                   target_feature=settings['target_feature'],
                                                                   aquifer_by_stations=aquifer_by_stations,
                                                                   k=settings['k'])
        
        # Hyperparameter tuning
        if model_name not in ['linear_regression']:
            best_params = hidden_mlp.hyperparameter_tuning(model_name=model_name,
                                                           horizon_max=settings['horizon_max'],
                                                           aquifers_list=settings['aquifers_list'],
                                                           best_features=best_features,
                                                           target_feature=settings['target_feature'],
                                                           test_len=settings['test_len'],
                                                           val_len=settings['val_len'],
                                                           aquifer_by_stations=aquifer_by_stations,
                                                           time_moe_outputs=time_moe_outputs)
        else:
            best_params = None
        
        # Final training
        r2_average, r2_scores, predictions = hidden_mlp.final_training(model_name=model_name,
                                                                              aquifers_list=settings['aquifers_list'],
                                                                              test_len=settings['test_len'],
                                                                              val_len=settings['val_len'],
                                                                              horizon_max=settings['horizon_max'],
                                                                              target_feature=settings['target_feature'],
                                                                              aquifer_by_stations=aquifer_by_stations,
                                                                              best_features=best_features,
                                                                              best_params=best_params,
                                                                              time_moe_outputs=time_moe_outputs)
        
        # Define the name for the results file
        saving_name = f'{name}_{model_name}'
    elif name == 'hidden_layer + deep_learning':
        # Load the data
        aquifer_by_stations = joblib.load('../../data/interim/ground-water-and-weather-with-forecasts-and-additional-features.joblib')
        # Shift the forecast features in the data
        aquifer_by_stations = hidden_deep_learning.shift_data(aquifers_list=settings['aquifers_list'],
                                                       aquifer_by_stations=aquifer_by_stations,
                                                       horizon_max=settings['horizon_max'])
        
        # Get the time moe outputs
        time_moe_outputs = joblib.load('../../data/interim/time_moe_outputs.joblib')
        # Convert to dataframes and add dates
        time_moe_outputs = hidden_deep_learning.time_moe_outputs_preprocess(time_moe_outputs)
        # Get the feature names
        time_moe_features = hidden_deep_learning.get_time_moe_feature_names(time_moe_outputs)

        # Set the best features to [] for the json saving
        best_features = []

        # Hyperparameter tuning
        best_params = hidden_deep_learning.hyperparameter_tuning(model_name=model_name,
                                                          horizon_max=settings['horizon_max'],
                                                          aquifers_list=settings['aquifers_list'],
                                                          target_feature=settings['target_feature'],
                                                          test_len=settings['test_len'],
                                                          val_len=settings['val_len'],
                                                          aquifer_by_stations=aquifer_by_stations,
                                                          time_moe_outputs=time_moe_outputs,
                                                          time_moe_features=time_moe_features)
        
        # Final training
        r2_average, r2_scores, predictions = hidden_deep_learning.final_training(model_name=model_name,
                                                                          aquifers_list=settings['aquifers_list'],
                                                                          test_len=settings['test_len'],
                                                                          horizon_max=settings['horizon_max'],
                                                                          target_feature=settings['target_feature'],
                                                                          aquifer_by_stations=aquifer_by_stations,
                                                                          best_params=best_params,
                                                                          time_moe_outputs=time_moe_outputs,
                                                                          time_moe_features=time_moe_features)
        
        # Define the name for the results file
        saving_name = f'{name}_{model_name}'

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