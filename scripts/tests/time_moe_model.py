import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM
import joblib
from sklearn.metrics import r2_score
import optuna
import json
import warnings
# Ignore warnings
warnings.filterwarnings("ignore")
import time

# HELPER FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
def predictions_to_predictions_diff(preceeding, predictions):
    #print(f"Preceeding: {preceeding}")
    # Convert predictions to numpy array if they're not already
    preceeding = np.array([preceeding])
    predictions = np.array(predictions)

    # Print dimensions of the arrays
    #print("Preceeding shape:", preceeding.shape)
    #print("Predictions shape:", predictions.shape)
    
    # Concatenate preceeding and predictions
    predictions_cat = np.concatenate((preceeding, predictions))

    # Create an array of differences between the predictions
    predictions_diff = np.diff(predictions_cat, axis=0)
    
    return predictions_diff


# HYPERPARAMETER TUNING
# ----------------------------------------------------------------------------------------------------------------------
def hyperparameter_tuning(horizon_max, aquifers_list, target_feature, val_len, test_len, aquifer_by_stations):
    def objective(trial):
        context_length = trial.suggest_int('context_length', horizon_max, 1460)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            'Maple728/TimeMoE-200M',
            device_map=device,
            trust_remote_code=True,
        ).to(device)
        
        # List for r2 results for different prediction horizons
        r2_scores = [[] for _ in range(horizon_max)]
        
        for aquifer in aquifers_list:
            # List for storing the predictions
            predictions = [[] for _ in range(horizon_max)]

            train_set = aquifer_by_stations[aquifer][target_feature].values[:-test_len]
        
            with torch.no_grad():
                # Iterate from day_len days before the end, to the last day
                for j in range(val_len + (horizon_max-1), 0, -1):
                    y = train_set[-(j + context_length):-j]
        
                    # Normalize the data
                    mean, std = y.mean(), y.std()
                    y = (y - mean) / std
                    
                    # Convert to tensor and add batch dimension, ensuring float32 dtype
                    input_data = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    forecast = model.generate(
                        inputs=input_data,
                        max_new_tokens=horizon_max
                    )
                    
                    # Convert back to numpy array
                    forecast = forecast[0][-horizon_max:].cpu().numpy()
                    forecast = forecast * std + mean

                    '''if target_feature == 'altitude':
                        # Convert the predictions to prediction differences
                        forecast = predictions_to_predictions_diff(aquifer_by_stations[aquifer][target_feature].values[-(j+1+test_len)], forecast)'''
        
                    # Store the results for every prediction horizon separately
                    for i in range(horizon_max):
                        predictions[i].append(forecast[i])
            
            for i in range(horizon_max):
                if i == 0:
                    predictions[i] = predictions[i][-val_len:]
                else:
                    predictions[i] = predictions[i][horizon_max-i-1:-i]

            '''if target_feature == 'altitude':
                # Calculate the r2 scores and store them in a list
                for i in range(horizon_max):
                    r2_scores[i].append(r2_score(aquifer_by_stations[aquifer][f'{target_feature}_diff'][-(val_len + test_len): -test_len], predictions[i]))
            else:'''
            # Calculate the r2 scores and store them in a list
            for i in range(horizon_max):
                r2_scores[i].append(r2_score(train_set[-val_len:], predictions[i]))

        # Calculate the average r2 score
        r2_average =  []
        for i in range(horizon_max):
            r2_average.append(np.mean(r2_scores[i]))

        loss = np.mean(r2_average)
        return loss

    # Run the optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1)
    
    # Return the best parameters
    return study.best_params


# FINAL TRAINING
# ----------------------------------------------------------------------------------------------------------------------
def final_training(horizon_max, aquifers_list, target_feature, test_len, aquifer_by_stations, best_params):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        'Maple728/TimeMoE-200M',
        device_map=device,
        trust_remote_code=True,
    ).to(device)
    
    # List for r2 results for different prediction horizons
    r2_scores = [[] for _ in range(horizon_max)]

    # Create a dictionary for the predictions from all of the different aquifers
    predictions_by_stations = {key: [] for key in aquifers_list}
    
    for aquifer in aquifers_list:
        # List for storing the predictions
        predictions = [[] for _ in range(horizon_max)]
    
        train_set = aquifer_by_stations[aquifer][target_feature]
    
        with torch.no_grad():
            # Iterate from day_len days before the end, to the last day
            for j in range(test_len + (horizon_max-1), 0, -1):
                y = train_set[-(j + best_params['context_length']):-j]
    
                # Normalize the data
                mean, std = y.mean(), y.std()
                y = (y - mean) / std
                
                # Convert to tensor and add batch dimension, ensuring float32 dtype
                input_data = torch.tensor(y.values, dtype=torch.float32).unsqueeze(0).to(device)
                
                forecast = model.generate(
                    inputs=input_data,
                    max_new_tokens=horizon_max
                )
                
                # Convert back to numpy array
                forecast = forecast[0][-horizon_max:].cpu().numpy()
                forecast = forecast * std + mean
    
                '''if target_feature == 'altitude':
                    # Convert the predictions to prediction differences
                    forecast = predictions_to_predictions_diff(train_set.values[-(j+1)], forecast)'''

                # Store the results for every prediction horizon separately
                for i in range(horizon_max):
                    predictions[i].append(forecast[i])
        
        for i in range(horizon_max):
            if i == 0:
                predictions[i] = predictions[i][-test_len:]
            else:
                predictions[i] = predictions[i][horizon_max-i-1:-i]

    
        # Add the predictios to the dictionary
        predictions_by_stations[aquifer] = predictions

        '''if target_feature == 'altitude':
            for i in range(horizon_max):
                r2_scores[i].append(r2_score(aquifer_by_stations[aquifer][f'{target_feature}_diff'][-test_len:], predictions[i]))

        else:'''
        # Calculate the r2 scores and store them in a list
        for i in range(horizon_max):
            r2_scores[i].append(r2_score(train_set[-test_len:], predictions[i]))
    
    # Calculate the average r2 score
    r2_average =  []
    for i in range(horizon_max):
        r2_average.append(np.mean(r2_scores[i]))
    
    # Return the predictions and the r2 scores
    return r2_average, r2_scores, predictions_by_stations


# SAVING THE RESULTS
# ----------------------------------------------------------------------------------------------------------------------
def convert_to_native(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    return obj

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

def save_results(model_name, multivariate, r2_scores, predictions, best_features, best_params, file_path):
    # Create a dictionary to store the results
    results = {
        'model_name': model_name,
        'multivariate': multivariate,
        'r2_scores': convert_to_native(r2_scores),
        'predictions': convert_to_native(predictions),
        'best_features': best_features,
        'best_params': best_params
    }

    with open(file_path, 'w') as file:
        json.dump(results, file, indent=4)


# EXPERIMENT SETTINGS
# ----------------------------------------------------------------------------------------------------------------------
# Load the experiment settings
with open('experiment_settings/time_moe_experiment_settings.json', 'r') as file:
    experiment_settings = json.load(file)


# TESTING THE MODELS
# ----------------------------------------------------------------------------------------------------------------------
for name, settings in experiment_settings.items():
    # Measure the time
    start_time = time.time()

    # Load the data
    aquifer_by_stations = joblib.load('../../data/interim/ground-water-and-weather-with-forecasts-and-additional-features.joblib')

    for aquifer in aquifer_by_stations.keys():
        aquifer_by_stations[aquifer] = aquifer_by_stations[aquifer][[settings['target_feature'], 'altitude_diff', 'date', 'station_id']]

    # Get the model name
    model_name = experiment_settings[name]['model_name']

    # Variable that tells us if the model is multivariate
    multivariate = False
    best_features = {}

    # Hyperparameter tuning
    '''best_params = hyperparameter_tuning(settings['horizon_max'], settings['aquifers_list'], 
                                        settings['target_feature'], settings['val_len'], settings['test_len'], 
                                        aquifer_by_stations)'''
    best_params = {'context_length': 720}
        
    # Final training
    r2_average, r2_scores, predictions = final_training(settings['horizon_max'], settings['aquifers_list'], 
                                                        settings['target_feature'], settings['test_len'], 
                                                        aquifer_by_stations, best_params)
    
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
    index = get_index(folder_path='../results/foundation_models', file_name=name)

    # Save the results
    file_path = f'../results/foundation_models/{name}_{index}.json'
    save_results(model_name, multivariate, r2_scores, predictions, best_features, best_params, file_path)

    # Print the results
    print("--------------------------------------------------------------------------------------------------")
    print(f"Model: {name}")
    print(f"R2 average: {r2_average}")
    print(f"Total execution time: {total_time_string}")
    print("--------------------------------------------------------------------------------------------------\n\n")
