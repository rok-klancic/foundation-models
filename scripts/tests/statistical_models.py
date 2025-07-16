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


# HYPERPARAMETER TUNING
# ----------------------------------------------------------------------------------------------------------------------
def hyperparameter_tuning(model_name, horizon_max, aquifers_list, best_features, target_feature, val_len, test_len, aquifer_by_stations):
    def objective(trial):
        if model_name == 'random_forest':
            n_estimators = trial.suggest_int('n_estimators', 10, 500)
            max_depth = trial.suggest_categorical('max_depth', [None, 10, 20, 30, 50])
            max_features = trial.suggest_categorical('max_features', ["sqrt", "log2", 0.5, 1.0])
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    
            # Initialize the RandomForestClassifier
            model = RandomForestRegressor(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        max_features=max_features,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf,
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
              alpha = trial.suggest_float('alpha', 1e-4, 100.0, log=True)

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
    study.optimize(objective, n_trials=50)

    # Return the best parameters
    return study.best_params


# FEATURE SELECTION
# ----------------------------------------------------------------------------------------------------------------------
# Time series split class
class Last365TimeSeriesSplit(BaseCrossValidator):
    def __init__(self, n_splits, test_size):
        self.n_splits = n_splits
        self.test_size = test_size

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        test_start = n_samples - self.test_size
        train_indices = indices[:test_start]
        test_indices = indices[test_start:]
        yield train_indices, test_indices

def ga_feature_selection(model_name, aquifer, test_len, val_len, horizon_max, target_feature, aquifer_by_stations):
    # Initialize model
    if model_name == 'linear_regression':
        model = LinearRegression(n_jobs=-1)
    elif model_name == 'random_forest':
        model = RandomForestRegressor(n_jobs=-1, random_state=42)
    elif model_name == 'gradient_boosting':
        #model = GradientBoostingRegressor(random_state=42)
        model = HistGradientBoostingRegressor(random_state=42)
    else:
        raise ValueError(f"Model {model_name} not supported for feature selection")
    
    # Dictionary to store the best features
    best_features = {}
    for horizon in range(1, horizon_max + 1):
        best_features[f'horizon_{horizon}'] = []
    
    for horizon in range(1, horizon_max+1):
        X_train = aquifer_by_stations[aquifer][:-(test_len + horizon)].drop(columns=['date', 'station_id', 'id', 'location_id'])
        y_train = aquifer_by_stations[aquifer][target_feature][horizon:-(test_len)]
    
        # Initialize genetic algorithm feature selector with max_features set
        gafs = GAFeatureSelectionCV(
            estimator=model,
            cv=Last365TimeSeriesSplit(n_splits=1, test_size=val_len),
            scoring='r2',
            population_size=100,
            generations=25,
            n_jobs=-1,
            verbose=True,
            keep_top_k=5,
            elitism=True,
            max_features=40,  # Set the maximum number of features to select
            mutation_probability=0.2,
            crossover_probability=0.8
        )
    
        # Fit the feature selector
        gafs.fit(X_train, y_train)
        best_features[f'horizon_{horizon}'] = list(gafs.get_feature_names_out(X_train.columns))

    # Return the best features
    return best_features

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
    if model_name == 'random_forest':
        model = RandomForestRegressor(n_estimators= best_params['n_estimators'],
                                 max_depth= best_params['max_depth'],
                                 max_features= best_params['max_features'],
                                 min_samples_split= best_params['min_samples_split'],
                                 min_samples_leaf= best_params['min_samples_leaf'],
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
    elif model_name == 'linear_regression':
        model = LinearRegression(n_jobs=-1)

    elif model_name == 'ridge_regression':
        model = Ridge(alpha= best_params['alpha'])

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

            if model_name != 'gradient_boosting':
                # Define the train and test set
                X_train = aquifer_by_stations[aquifer][chosen_features][:-(test_len + horizon)]
                y_train = aquifer_by_stations[aquifer][target_feature][horizon:-test_len]
        
                X_test = aquifer_by_stations[aquifer][chosen_features][-(test_len + horizon):-horizon]
                y_test = aquifer_by_stations[aquifer][target_feature][-test_len:]

            else:
                # Define the train and test set
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

def save_results(model_name, multivariate, r2_scores, predictions, feature_selection, best_features, best_params, file_path, time_string):
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
                 file_path=file_path,
                 time_string=total_time_string)

    # Print the results
    print("--------------------------------------------------------------------------------------------------")
    print(f"Model: {name}")
    print(f"R2 average: {r2_average}")
    print(f"Total execution time: {total_time_string}")
    print("--------------------------------------------------------------------------------------------------\n\n")