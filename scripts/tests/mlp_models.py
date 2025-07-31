# IMPORT THE NECESSARY LIBRARIES
# ----------------------------------------------------------------------------------------------------------------------
import joblib
import pandas as pd
import numpy as np
#import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import optuna
import json
# Torch
import torch
import torch.nn as nn
# RandomForest
from sklearn.ensemble import RandomForestRegressor
# GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
# GAFeatureSelectionCV
from sklearn.model_selection import TimeSeriesSplit
#from sklearn_genetic import GAFeatureSelectionCV
#from sklearn_genetic.plots import plot_fitness_evolution
# Decision Tree
from sklearn.tree import DecisionTreeRegressor
# Linear Regression
from sklearn.linear_model import LinearRegression, Ridge
import os
#SelectKBest
from sklearn.feature_selection import SelectKBest, f_regression
# Time
import time

# CONSTANTS
# ----------------------------------------------------------------------------------------------------------------------
MAX_EPOCHS = 5000
EXPERIMENT_SETTINGS_PATH = 'experiment_settings/mlp_models_experiment_settings.json'
DATA_PATH = '../../data/interim/ground-water-and-weather-with-forecasts-and-additional-features.joblib'
RESULTS_PATH = '../results/mlp_models'


# MLP MODEL
# able to change the number of hidden layers and the dropout rate
# ----------------------------------------------------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout):
        super(MLP, self).__init__()
        layers = []
        in_features = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = hidden_size

        layers.append(nn.Linear(in_features, 1))  # Output layer
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Method for training the model
def mlp_fit(model, X_train, y_train, X_val, y_val, patience, criterion, optimizer, horizon):
    # Training loop with early stopping
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        # Training
        optimizer.zero_grad()
        outputs = model(X_train)
        train_loss = criterion(outputs, y_train)
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        model.train()

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1} for horizon {horizon}")
            # Load best model
            model.load_state_dict(best_model_state)
            return model
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model


# HYPERPARAMETER TUNING
# ----------------------------------------------------------------------------------------------------------------------
def hyperparameter_tuning(model_name, horizon_max, aquifers_list, best_features, target_feature, val_len, test_len, aquifer_by_stations):
    def objective(trial):
        if model_name == 'mlp':
                    hidden_layers = trial.suggest_categorical('hidden_layers', [[64], [32],
                                                                                [32, 16], [64, 32], [128, 64],
                                                                                [128, 64, 32], [64, 32, 16], [256, 128, 64],
                                                                                [128, 64, 32, 16], [256, 128, 64, 32], [512, 256, 128, 64]])
                    dropout = trial.suggest_categorical('dropout', [0.0, 0.1, 0.2, 0.5])
                    patience = trial.suggest_categorical('patience', [5, 10, 25, 50, 100])
                    lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
                    weight_decay = trial.suggest_categorical('weight_decay', [0, 1e-4, 1e-3, 1e-2, 5e-1, 1e-1, 1, 10])
            
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
        
                # Define the scaler
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()

                # Scale the features
                X_train = scaler_X.fit_transform(X_train)
                X_test = scaler_X.transform(X_test)
                y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
                y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

                # Convert to PyTorch tensors
                X_train_tensor = torch.FloatTensor(X_train)
                y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
                X_test_tensor = torch.FloatTensor(X_test)
                y_test_tensor = torch.FloatTensor(y_test_scaled).reshape(-1, 1)

                # Initialize model, loss function and optimizer
                model = MLP(input_size=X_train.shape[1],
                            hidden_layers=hidden_layers,
                            dropout=dropout)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                                
                # Train the model
                model = mlp_fit(model=model,
                                X_train=X_train_tensor,
                                y_train=y_train_tensor,
                                X_val=X_test_tensor,
                                y_val=y_test_tensor,
                                patience=patience,
                                criterion=criterion,
                                optimizer=optimizer,
                                horizon=horizon)
                
                # Make predictions
                model.eval()
                with torch.no_grad():
                    forecast = model(X_test_tensor).numpy()
                
                # Flatten
                forecast = np.ravel(forecast)
                
                # Unscale the predictions
                forecast = scaler_y.inverse_transform(forecast.reshape(-1, 1)).ravel()
                
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
def final_training(model_name,
                   aquifers_list,
                   test_len,
                   val_len,
                   horizon_max,
                   target_feature,
                   aquifer_by_stations,
                   best_features,
                   best_params):
    # Initialize model
    if model_name == 'mlp':
        pass

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
            X_train = aquifer_by_stations[aquifer][chosen_features][:-(val_len + horizon + test_len)]
            y_train = aquifer_by_stations[aquifer][target_feature][horizon:-(val_len + test_len)]
            
            X_val = aquifer_by_stations[aquifer][chosen_features][-(val_len + horizon + test_len):-(horizon + test_len)]
            y_val = aquifer_by_stations[aquifer][target_feature][-(val_len + test_len):-test_len]

            X_test = aquifer_by_stations[aquifer][chosen_features][-(test_len + horizon):-horizon]
            y_test = aquifer_by_stations[aquifer][target_feature][-test_len:]

            # Define the scaler
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            # Scale the features
            X_train = scaler_X.fit_transform(X_train)
            X_val = scaler_X.transform(X_val)
            X_test = scaler_X.transform(X_test)
            y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
            y_val = scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
            X_test_tensor = torch.FloatTensor(X_test)
            
            # Initialize model, loss function and optimizer
            model = MLP(input_size=X_train.shape[1],
                        hidden_layers=best_params['hidden_layers'],
                        dropout=best_params['dropout'])
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
            
            # Train the model
            model = mlp_fit(model=model,
                            X_train=X_train_tensor,
                            y_train=y_train_tensor,
                            X_val=X_val_tensor,
                            y_val=y_val_tensor,
                            patience=best_params['patience'],
                            criterion=criterion,
                            optimizer=optimizer,
                            horizon=horizon)
            
            # Make predictions
            model.eval()
            with torch.no_grad():
                forecast = model(X_test_tensor).numpy()
            
            # Flatten
            forecast = np.ravel(forecast)
            
            # Unscale the predictions
            forecast = scaler_y.inverse_transform(forecast.reshape(-1, 1)).ravel()
            
            # Store to the predictions
            predictions.append(forecast)
            
            # Calculate and save the r2 score
            r2_scores[horizon-1].append(r2_score(y_test, forecast))

        # Store the predictions to the dictionary
        predictions_by_stations[aquifer] = predictions

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

def save_results(model_name, multivariate, r2_scores, predictions, feature_selection, best_features, best_params, file_path, time_string):
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
        'feature_selection': feature_selection,
        'best_features': best_features,
        'best_params': best_params,
        'total_time': time_string
    }

    with open(file_path, 'w') as file:
        json.dump(results, file, indent=4)


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
    r2_average, r2_scores, predictions = final_training(model_name=model_name,
                                                        aquifers_list=settings['aquifers_list'], 
                                                        test_len=settings['test_len'], 
                                                        val_len=settings['val_len'],
                                                        horizon_max=settings['horizon_max'], 
                                                        target_feature=settings['target_feature'], 
                                                        aquifer_by_stations=aquifer_by_stations,
                                                        best_features=best_features, 
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
                 feature_selection=settings['feature_selection'], 
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