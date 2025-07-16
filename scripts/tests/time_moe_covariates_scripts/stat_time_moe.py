import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from transformers import AutoModelForCausalLM

import joblib

from sklearn.metrics import r2_score

# Linear regression, Ridge, RandomForest, GradientBoosting
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

# NBEATSx
from neuralforecast.models import NBEATSx
from neuralforecast.core import NeuralForecast

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Optuna
import optuna

# SelectKBest
from sklearn.feature_selection import SelectKBest, f_regression


# HYPERPARAMETER TUNING
# ----------------------------------------------------------------------------------------------------------------------
def hyperparameter_tuning(model_name,
                          horizon_max,
                          aquifers_list,
                          best_features,
                          target_feature,
                          val_len,
                          test_len,
                          aquifer_by_stations):
    def objective(trial):
        if model_name == 'random_forest':
            n_estimators = trial.suggest_int('n_estimators', 10, 500)
            max_depth = trial.suggest_categorical('max_depth', [None, 10, 20, 30, 50])
            max_features = trial.suggest_categorical('max_features', ["sqrt", "log2", 0.5, 1.0])
    
            # Initialize the RandomForestClassifier
            model = RandomForestRegressor(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        max_features=max_features,
                                        n_jobs=-1,
                                        random_state=42)
        elif model_name == 'gradient_boosting':
            max_depth = trial.suggest_categorical('max_depth', [3, 5, 7, 10, None])
            max_features = trial.suggest_categorical('max_features', [0.5, 0.75, 1.0])
            learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.2])
            min_samples_leaf = trial.suggest_categorical('min_samples_leaf', [1, 5, 10, 20, 40, 60])
            l2_regularization = trial.suggest_categorical('l2_regularization', [0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
            n_iter_no_change = trial.suggest_categorical('n_iter_no_change', [10, 20, 30, 40, 50])
            
            
            # Initialize the HistGradientBoostingRegressor
            model = HistGradientBoostingRegressor(max_iter=500, 
                                                  max_depth=max_depth,
                                                  max_features=max_features,
                                                  learning_rate=learning_rate,
                                                  min_samples_leaf=min_samples_leaf,
                                                  l2_regularization=l2_regularization,
                                                  early_stopping=True,
                                                  n_iter_no_change=n_iter_no_change,
                                                  random_state=42)
        
        elif model_name == 'ridge_regression':
            alpha = trial.suggest_loguniform('alpha', 1e-4, 1e4)

            model = Ridge(alpha=alpha)

        else:
            raise ValueError(f"Model {model_name} not supported for hyperparameter tuning")
        
        # List for r2 results for different prediction horizons
        r2_scores = [[] for _ in range(horizon_max)]
        
        for aquifer in aquifers_list:
        
            for horizon in range (1, horizon_max+1, 1):
                # Get the best features
                # Check if the best features are aquifer specific or not
                if aquifer in best_features.keys():
                    chosen_features = best_features[aquifer][f'horizon_{horizon}']
                else:
                    chosen_features = best_features[f'horizon_{horizon}']
                
                # Define the train and test set
                X_train = aquifer_by_stations[aquifer][chosen_features][:-(val_len + horizon + test_len)]
                y_train = aquifer_by_stations[aquifer][target_feature][horizon:-(val_len + test_len)]
        
                X_test = aquifer_by_stations[aquifer][chosen_features][-(val_len + horizon + test_len):-(horizon + test_len)]
                y_test = aquifer_by_stations[aquifer][target_feature][-(val_len + test_len):-test_len]
        
                # Train the model
                if model_name != 'gradient_boosting':
                    model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
        
                # Make predictions
                forecast = model.predict(X_test).tolist()
                
                # Calculate and save the r2 score
                r2_scores[horizon-1].append(r2_score(y_test, forecast))
        
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


# FEATURE SELECTION
# ----------------------------------------------------------------------------------------------------------------------
def k_best_feature_selection(aquifers_list, test_len, horizon_max, target_feature, aquifer_by_stations, k):
    # Dictionary to store the best features
    best_features = {}
    
    # Select k best features using SelectKBest
    selector = SelectKBest(score_func=f_regression, k=k)
    
    for aquifer in aquifers_list:
        # Get all features
        features = aquifer_by_stations[aquifer].drop(columns=['date', 'station_id', 'id', 'location_id']).columns

        best_features[aquifer] = {}
        # For each prediction horizon
        for horizon in range(1, horizon_max + 1):
            # Prepare data
            X = aquifer_by_stations[aquifer][features][:-(test_len + horizon)]
            y = aquifer_by_stations[aquifer][target_feature][horizon:-test_len]
            
            # Fit selector
            selector.fit(X, y)
            
            # Get selected feature names
            selected_features = features[selector.get_support()].tolist()
            
            # Store selected features for this horizon
            best_features[aquifer][f'horizon_{horizon}'] = selected_features
    
    # Return the best features
    return best_features

# FINAL TRAINING
# ----------------------------------------------------------------------------------------------------------------------
def final_training_statistical(model_name,
                   aquifers_list,
                   test_len,
                   val_len,
                   horizon_max,
                   target_feature,
                   aquifer_by_stations,
                   best_features,
                   best_params):
    
    # Make the test_len greater
    test_len = test_len*3 + 50

    # Initialize model
    if model_name == 'random_forest':
        model = RandomForestRegressor(n_estimators= best_params['n_estimators'],
                                 max_depth= best_params['max_depth'],
                                 max_features= best_params['max_features'],
                                 n_jobs=-1,
                                 random_state=42)
    elif model_name == 'gradient_boosting':
        model = HistGradientBoostingRegressor(max_iter= 500,
                                              max_depth= best_params['max_depth'],
                                              max_features= best_params['max_features'],
                                              learning_rate= best_params['learning_rate'],
                                              min_samples_leaf= best_params['min_samples_leaf'],
                                              l2_regularization= best_params['l2_regularization'],
                                              n_iter_no_change= best_params['n_iter_no_change'],
                                              early_stopping=True,
                                              random_state=42)
    elif model_name == 'ridge_regression':
        model = Ridge(alpha= best_params['alpha'])

    elif model_name == 'linear_regression':
        model = LinearRegression(n_jobs=-1)

    else:
        raise ValueError(f"Model {model_name} not supported for final training")


    # List for r2 results for different prediction horizons
    r2_scores = [[] for _ in range(horizon_max)]
    
    # Dictionary for storing the predictions
    predictions_by_stations = {key: [] for key in aquifers_list}
    
    for aquifer in aquifers_list:
        predictions = [] # make sure that the list is empty for every aquifer
    
        for horizon in range (1, horizon_max+1, 1):
            # Get the best features
            # Check if the best features are aquifer specific or not
            if aquifer in best_features.keys():
                chosen_features = best_features[aquifer][f'horizon_{horizon}']
            else:
                chosen_features = best_features[f'horizon_{horizon}']

            # Define the train and test set
            if model_name != 'gradient_boosting':
                X_train = aquifer_by_stations[aquifer][chosen_features][:-(test_len + horizon)]
                y_train = aquifer_by_stations[aquifer][target_feature][horizon:-test_len]
        
                X_test = aquifer_by_stations[aquifer][chosen_features][-(test_len + horizon):-horizon]
                y_test = aquifer_by_stations[aquifer][target_feature][-test_len:]

            else:
                X_train = aquifer_by_stations[aquifer][chosen_features][:-(val_len + horizon + test_len)]
                y_train = aquifer_by_stations[aquifer][target_feature][horizon:-(val_len + test_len)]
                
                X_val = aquifer_by_stations[aquifer][chosen_features][-(val_len + horizon + test_len):-(horizon + test_len)]
                y_val = aquifer_by_stations[aquifer][target_feature][-(val_len + test_len):-test_len]
                
                X_test = aquifer_by_stations[aquifer][chosen_features][-(test_len + horizon):-horizon]
                y_test = aquifer_by_stations[aquifer][target_feature][-test_len:]
    
            # Train the model
            if model_name != 'gradient_boosting':
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    
            # Make predictions
            forecast = model.predict(X_test).tolist()
    
            # Store to the predictions
            predictions.append(forecast)
            
            # Calculate and save the r2 score
            r2_scores[horizon-1].append(r2_score(y_test, forecast))

        # Store the predictions to the dictionary
        predictions_by_stations[aquifer].append(predictions)

    # Calculate the residuals
    # Shorten the data length to the prediction length
    for aquifer in aquifers_list:
        aquifer_by_stations[aquifer] = aquifer_by_stations[aquifer][-test_len:]
    
    # A new dataframe with residuals
    residuals = {key: [] for key in aquifers_list}
    
    for aquifer in aquifers_list:
        dataframe = pd.DataFrame()
        for i in range(1, horizon_max+1):
            dataframe[f'residual_{i}'] = aquifer_by_stations[aquifer][target_feature] - predictions_by_stations[aquifer][i-1]
        residuals[aquifer] = dataframe

    return residuals, predictions_by_stations

# ----------------------------------------------------------------------------------------------------------------------
def final_training_time_moe(horizon_max, 
                            aquifers_list, 
                            test_len, 
                            residuals, 
                            context_length, 
                            aquifer_by_stations, 
                            statisctical_predictions):
    
    # Load the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        'Maple728/TimeMoE-200M',
        device_map=device,
        trust_remote_code=True,
    ).to(device)
    
    # List for r2 results for different prediction horizons
    r2_scores = [[] for _ in range(horizon_max)]
    
    # Dictionary for storing the predictions
    predictions_by_stations = {key: [] for key in aquifers_list}

    for aquifer in aquifers_list:
        # List for storing the predictions
        predictions = [[] for _ in range(horizon_max)]
    
        for horizon in range(1, horizon_max+1, 1):
            with torch.no_grad():
                # Iterate from test_len days before the end, to the last day
                for j in range(test_len + (horizon-1), (horizon-1), -1):
                    y = residuals[aquifer][f'residual_{horizon}'][-(j + context_length):-j]
    
                    # Normalize the data
                    mean, std = y.mean(), y.std()
                    y = (y - mean) / std
                    
                    # Convert to tensor, add batch dimension, ensure float32 dtype, and move to device
                    input_data = torch.tensor(y.values, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    forecast = model.generate(
                        inputs=input_data,
                        max_new_tokens=horizon
                    )
                    
                    # Convert back to numpy array
                    forecast = forecast[0][-horizon:].cpu().numpy()
                    forecast = forecast * std + mean
    
                    # Store the results for every prediction horizon separately
                    predictions[horizon-1].append(forecast[horizon-1])
        
        # Transform the residuals to the predictions
        for i in range(horizon_max):
            predictions[i] = statisctical_predictions[aquifer][i][-test_len:] + np.array(predictions[i])

        # Store the predictions to the dictionary
        predictions_by_stations[aquifer].append(predictions)
    
        # Calculate the r2 scores and store them in a list
        for i in range(horizon_max):
            r2_scores[i].append(r2_score(aquifer_by_stations[aquifer]['altitude_diff'][-test_len:], predictions[i]))

        # Return the average r2 scores
        r2_average =  []    
        for i in range(horizon_max):
            r2_average.append(np.mean(r2_scores[i]))
    
    # Return the predictions and the r2 scores
    return r2_average, r2_scores, predictions_by_stations