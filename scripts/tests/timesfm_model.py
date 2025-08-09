# IMPORT THE NECESSARY LIBRARIES
# ----------------------------------------------------------------------------------------------------------------------
import optuna
import json
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import os
#TimesFM
import timesfm
# Time
import time


# IMPORTANT NOTE:
# This script only runs on linux machines

# CONSTANTS
# ----------------------------------------------------------------------------------------------------------------------
EXPERIMENT_SETTINGS_PATH = 'experiment_settings/timesfm_model_experiment_settings.json'
DATA_PATH = '../../data/interim/ground-water-and-weather-with-forecasts-and-additional-features.joblib'
RESULTS_PATH = '../results/foundation_models'
MODEL_CHECKPOINT = "google/timesfm-1.0-200m"

additional_parameters = [
 'tn',
 'tx',
 'nn_decodeText',
 'rr_decodeText',
 'ff_decodeText'
 ]

# Function for defining the model
def define_model(model_name, horizon):
    model = timesfm.TimesFm(
          hparams=timesfm.TimesFmHparams(
              backend="gpu",
              per_core_batch_size=32,
              horizon_len=horizon,
          ),
          checkpoint=timesfm.TimesFmCheckpoint(
              huggingface_repo_id=model_name),
      )
    return model


# HYPERPARAMETER TUNING
# ----------------------------------------------------------------------------------------------------------------------
def hyperparameter_tuning(model_name, horizon_max, aquifers_list, target_feature, val_len, test_len, aquifer_by_stations):
    def objective(trial):
        if model_name == 'timesfm_multivariate':
            context_len = trial.suggest_categorical('context_len', [64, 128, 256, 512])
            ridge = trial.suggest_float('ridge', 1e-4, 100.0, log=True)
            xreg_mode = trial.suggest_categorical('xreg_mode', ['timesfm + xreg', 'xreg + timesfm'])

            model = define_model(MODEL_CHECKPOINT, horizon_max) 

        
        elif model_name == 'timesfm_univariate':
            context_len = trial.suggest_categorical('context_len', [7, 30, 90, 180, 365, 512])
            model = define_model(MODEL_CHECKPOINT, horizon_max) 

        else:
            raise ValueError(f"Model {model_name} not supported for hyperparameter tuning")
        
        # List for r2 results for different prediction horizons
        r2_scores = [[] for _ in range(horizon_max)]
        
        for aquifer in aquifers_list:
            # List for storing the predictions
            predictions = [[] for _ in range(horizon_max)]

            # Get the dataset for the aquifer
            y = aquifer_by_stations[aquifer][:-test_len]

            # Iterate from val_len days before the end, to the last day
            for i in range(val_len + (horizon_max-1), 0, -1):
                # Define the input for the model
                y_temp = y[target_feature][-(context_len+i):-i].values

                # Scale the data
                scaler = StandardScaler()
                y_temp = scaler.fit_transform(y_temp.reshape(-1, 1)).ravel().tolist()

                # Make predictions
                if model_name == 'timesfm_univariate':
                    raw_forecast, _ = model.forecast(
                        inputs=[y_temp], 
                        freq=[0] * len(y_temp)
                    )
            
                elif model_name == 'timesfm_multivariate':
                    covariates = {}
                    for j in range(horizon_max+1):
                        for additional_parameter in additional_parameters:
                            if j != 0:
                                covariates[additional_parameter][0].append(y[f'{additional_parameter}_{j}'].iloc[-(i+1)])
                            else:
                                covariates[additional_parameter] = [(y[f'{additional_parameter}_{j}'].iloc[-(context_len+i):-i].values.tolist())]
                    
                    for key, value in covariates.items():
                        if len(value[0]) < context_len + horizon_max:
                            print(f"Covariate {key} has {len(value[0])} values")

                    cov_forecast, _ = model.forecast_with_covariates(  
                        inputs=[y_temp], # Wrap in list since inputs expects list of time series
                        dynamic_numerical_covariates= covariates,
                        freq=[0] * len(y_temp),
                        xreg_mode=xreg_mode,
                        ridge=ridge,
                        force_on_cpu=False,
                        normalize_xreg_target_per_input=True    # default
                    )
                # Unscale the forecast
                if model_name == 'timesfm_univariate':
                    forecast_unscaled = scaler.inverse_transform(raw_forecast[0].reshape(-1, 1)).ravel()
                elif model_name == 'timesfm_multivariate':
                    forecast_unscaled = scaler.inverse_transform(cov_forecast[0].reshape(-1, 1)).ravel()

                # Store the results for every prediction horizon separately
                for horizon in range(horizon_max):
                    predictions[horizon].append(forecast_unscaled[horizon])

            # Clean up the predictions
            for i in range(horizon_max):
                if i == 0:
                    predictions[i] = predictions[i][-val_len:]
                else:
                    predictions[i] = predictions[i][(horizon_max-i-1):-i]

            # Calculate the r2 scores and store them in a list
            for i in range(horizon_max):
                r2_scores[i].append(r2_score(y[target_feature][-val_len:], predictions[i]))
        
        # Calculate the average r2 score
        r2_average =  []
        
        for i in range(horizon_max):
            r2_average.append(np.mean(r2_scores[i]))
    
        # Set the loss as average of average r2 scores for different prediction horizons
        loss = np.mean(r2_average)
    
        return loss
    
    # Run the optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    # Return the best parameters
    return study.best_params


# FINAL TRAINING
# ----------------------------------------------------------------------------------------------------------------------
def final_training(model_name,
                   aquifers_list,
                   test_len,
                   horizon_max,
                   target_feature,
                   aquifer_by_stations,
                   best_params):
    # Initialize model
    if model_name == 'timesfm_multivariate':
        model = define_model(MODEL_CHECKPOINT, horizon_max) 

    elif model_name == 'timesfm_univariate':
        model = define_model(MODEL_CHECKPOINT, horizon_max) 
    
    else:
        raise ValueError(f"Model {model_name} not supported for hyperparameter tuning")


    # List for r2 results for different prediction horizons
    r2_scores = [[] for _ in range(horizon_max)]
    
    # Dictionary for storing the predictions
    predictions_by_stations = {key: [] for key in aquifers_list}
    
    for aquifer in aquifers_list:
        # List for storing the predictions
        predictions = [[] for _ in range(horizon_max)]
        
        # Get the dataset for the aquifer
        y = aquifer_by_stations[aquifer]
        
        # Iterate from val_len days before the end, to the last day
        for i in range(test_len + (horizon_max-1), 0, -1):
            # Define the input for the model
            y_temp = y[target_feature][-(best_params['context_len']+i):-i].values
        
            # Scale the data
            scaler = StandardScaler()
            y_temp = scaler.fit_transform(y_temp.reshape(-1, 1)).ravel().tolist()
        
            # Make predictions
            if model_name == 'timesfm_univariate':
                raw_forecast, _ = model.forecast(
                    inputs=[y_temp], 
                    freq=[0] * len(y_temp)
                )
        
            elif model_name == 'timesfm_multivariate':
                covariates = {}
                for j in range(horizon_max+1):
                    for additional_parameter in additional_parameters:
                        if j != 0:
                            covariates[additional_parameter][0].append(y[f'{additional_parameter}_{j}'].iloc[-(i+1)])
                        else:
                            covariates[additional_parameter] = [(y[f'{additional_parameter}_{j}'].iloc[-(best_params['context_len']+i):-i].values.tolist())]
            
                cov_forecast, _ = model.forecast_with_covariates(  
                    inputs=[y_temp], # Wrap in list since inputs expects list of time series
                    dynamic_numerical_covariates=covariates,
                    freq=[0] * len(y_temp),
                    xreg_mode=best_params['xreg_mode'],
                    ridge=best_params['ridge'],
                    force_on_cpu=False,
                    normalize_xreg_target_per_input=True    # default
                )
            # Unscale the forecast
            if model_name == 'timesfm_univariate':
                forecast_unscaled = scaler.inverse_transform(raw_forecast[0].reshape(-1, 1)).ravel()
            elif model_name == 'timesfm_multivariate':
                forecast_unscaled = scaler.inverse_transform(cov_forecast[0].reshape(-1, 1)).ravel()
        
            # Store the results for every prediction horizon separately
            for horizon in range(horizon_max):
                predictions[horizon].append(forecast_unscaled[horizon])
            
        # Clean the predictions
        for i in range(horizon_max):
            if i == 0:
                predictions[i] = predictions[i][-test_len:]
            else:
                predictions[i] = predictions[i][(horizon_max-i-1):-i]

        # Store the predictions to the dictionary
        predictions_by_stations[aquifer] = predictions

        # Calculate the r2 scores and store them in a list
        for i in range(horizon_max):
            r2_scores[i].append(r2_score(aquifer_by_stations[aquifer][target_feature][-test_len:], predictions[i]))

    # Return the average r2 scores
    r2_average =  []    
    for i in range(horizon_max):
        r2_average.append(np.mean(r2_scores[i]))

    return r2_average, r2_scores, predictions_by_stations


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

def save_results(model_name, multivariate, r2_scores, predictions, best_features, best_params, file_path, time_string):
    # Ensure lists
    predictions = convert_numpy_to_native(convert_ndarray_to_list(predictions))
    r2_scores = convert_numpy_to_native(convert_ndarray_to_list(r2_scores))
    best_features = convert_numpy_to_native(convert_ndarray_to_list(best_features))
    best_params = convert_numpy_to_native(convert_ndarray_to_list(best_params))
    
    # Create a dictionary to store the results
    results = {
        'model_name': model_name,
        'multivariate': multivariate,
        'r2_scores': r2_scores,
        'predictions': predictions,
        'best_features': best_features,
        'best_params': best_params,
        'total_time': time_string
    }

    with open(file_path, 'w') as file:
        json.dump(results, file, indent=4)

    print("Results saved to: ", file_path)


# EXPERIMENT SETTINGS
# ----------------------------------------------------------------------------------------------------------------------
# Load the experiment settings
with open(EXPERIMENT_SETTINGS_PATH, 'r') as file:
    experiment_settings = json.load(file)

# Load the data
aquifer_by_stations = joblib.load(DATA_PATH)

# Transform date column to year, month and day columns
for key in aquifer_by_stations.keys():
    aquifer_by_stations[key]['year'] = aquifer_by_stations[key]['date'].dt.year
    aquifer_by_stations[key]['month'] = aquifer_by_stations[key]['date'].dt.month
    aquifer_by_stations[key]['day'] = aquifer_by_stations[key]['date'].dt.day


# TESTING THE MODELS
# ----------------------------------------------------------------------------------------------------------------------
for name, settings in experiment_settings.items():
    # Measure the time
    start_time = time.time()

    # Get the model name
    model_name = experiment_settings[name]['model_name']

    # Variable that tells us if the model is multivariate
    multivariate = False

    # Check if we need additional features
    if settings['additional_features']:
        # Set the multivariate variable to True
        multivariate = True
        best_features = additional_parameters
    else:
        best_features = {}
        for horizon in range(1, settings['horizon_max'] + 1):
            best_features[f'horizon_{horizon}'] = ['altitude_diff']

    # Hyperparameter tuning
    if settings['hyperparameter_tuning']:
        best_params = hyperparameter_tuning(model_name=model_name, 
                                            horizon_max=settings['horizon_max'], 
                                            aquifers_list=settings['aquifers_list'], 
                                            target_feature=settings['target_feature'], 
                                            val_len=settings['val_len'],
                                            test_len=settings['test_len'], 
                                            aquifer_by_stations=aquifer_by_stations)
    else:
        best_params = {}
        
    # Final training
    r2_average, r2_scores, predictions = final_training(model_name=model_name,
                                                        aquifers_list=settings['aquifers_list'], 
                                                        test_len=settings['test_len'], 
                                                        horizon_max=settings['horizon_max'], 
                                                        target_feature=settings['target_feature'], 
                                                        aquifer_by_stations=aquifer_by_stations,
                                                        best_params=best_params)
    
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
    index = get_index(folder_path=RESULTS_PATH, file_name=name)

    # Save the results
    file_path = f'{RESULTS_PATH}/{name}_{index}.json'
    save_results(model_name=model_name, 
                 multivariate=multivariate, 
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