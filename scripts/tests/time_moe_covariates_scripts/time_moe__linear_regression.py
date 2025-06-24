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



# HYPERPARAMETER TUNING
# ----------------------------------------------------------------------------------------------------------------------
def hyperparameter_tuning(model_name, horizon_max, aquifers_list, best_features, target_feature, val_len, test_len, aquifer_by_stations):
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
            #n_estimators = trial.suggest_int('n_estimators', 10, 500)
            max_iter = trial.suggest_int('max_iter', 10, 500)
            max_depth = trial.suggest_categorical('max_depth', [None, 10, 20, 30, 50])
            #max_features = trial.suggest_categorical('max_features', ["sqrt", "log2", 0.5, 1.0])
            max_features = trial.suggest_categorical('max_features', [0.5, 0.75, 1.0])
            learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.2])
            
            
            # Initialize the RandomForestClassifier
            '''model = GradientBoostingRegressor(n_estimators=n_estimators,
                                          max_depth=max_depth,
                                          max_features=max_features,
                                          random_state=42)'''
            model = HistGradientBoostingRegressor(max_iter=max_iter, 
                                                  max_depth=max_depth,
                                                  max_features=max_features,
                                                  learning_rate=learning_rate,
                                                  random_state=42)
            
        else:
            raise ValueError(f"Model {model_name} not supported for hyperparameter tuning")
        
        # List for r2 results for different prediction horizons
        r2_scores = [[] for _ in range(horizon_max)]
        
        for aquifer in aquifers_list:
        
            for horizon in range (1, horizon_max+1, 1):
                # Define the train and test set
                X_train = aquifer_by_stations[aquifer][best_features[f'horizon_{horizon}']][:-(val_len + horizon + test_len)]
                y_train = aquifer_by_stations[aquifer][target_feature][horizon:-(val_len + test_len)]
        
                X_test = aquifer_by_stations[aquifer][best_features[f'horizon_{horizon}']][-(val_len + horizon + test_len):-(horizon + test_len)]
                y_test = aquifer_by_stations[aquifer][target_feature][-(val_len + test_len):-test_len]
        
                # Train the model
                model.fit(X_train, y_train)
        
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