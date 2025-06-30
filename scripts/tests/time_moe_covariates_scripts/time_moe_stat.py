import numpy as np
from sklearn.metrics import r2_score

# Ridge regression
from sklearn.linear_model import Ridge

# Random forest
from sklearn.ensemble import RandomForestRegressor

# HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Optuna
import optuna

#SelectKBest
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
                          aquifer_by_stations,
                          time_moe_forecast_features):
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
            max_iter = trial.suggest_int('max_iter', 10, 500)
            max_depth = trial.suggest_categorical('max_depth', [None, 10, 20, 30, 50])
            max_features = trial.suggest_categorical('max_features', [0.5, 0.75, 1.0])
            learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.2])
            
            
            # Initialize the RandomForestClassifier
            model = HistGradientBoostingRegressor(max_iter=max_iter, 
                                                  max_depth=max_depth,
                                                  max_features=max_features,
                                                  learning_rate=learning_rate,
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
                X_train = aquifer_by_stations[aquifer][time_moe_forecast_features + chosen_features][:-(val_len + horizon + test_len)]
                y_train = aquifer_by_stations[aquifer][target_feature][horizon:-(val_len + test_len)]
        
                X_test = aquifer_by_stations[aquifer][time_moe_forecast_features + chosen_features][-(val_len + horizon + test_len):-(horizon + test_len)]
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
    study.optimize(objective, n_trials=50)

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
                   horizon_max,
                   target_feature,
                   aquifer_by_stations,
                   best_features,
                   best_params,
                   time_moe_forecast_features):
    # Initialize model
    if model_name == 'random_forest':
        model = RandomForestRegressor(n_estimators= best_params['n_estimators'],
                                 max_depth= best_params['max_depth'],
                                 max_features= best_params['max_features'],
                                 n_jobs=-1,
                                 random_state=42)
    elif model_name == 'gradient_boosting':
        model = HistGradientBoostingRegressor(max_iter= best_params['max_iter'],
                                              max_depth= best_params['max_depth'],
                                              max_features= best_params['max_features'],
                                              learning_rate= best_params['learning_rate'],
                                              random_state=42)
    elif model_name == 'ridge_regression':
        model = Ridge(alpha= best_params['alpha'])
    else:
        raise ValueError(f"Model {model_name} not supported for final training")


    # List for r2 results for different prediction horizons
    r2_scores = [[] for _ in range(horizon_max)]
    
    # List for storing the predictions (useful for visualization)
    predictions = []
    
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
            X_train = aquifer_by_stations[aquifer][time_moe_forecast_features + chosen_features][:-(test_len + horizon)]
            y_train = aquifer_by_stations[aquifer][target_feature][horizon:-test_len]
    
            X_test = aquifer_by_stations[aquifer][time_moe_forecast_features + chosen_features][-(test_len + horizon):-horizon]
            y_test = aquifer_by_stations[aquifer][target_feature][-test_len:]
    
            # Train the model
            model.fit(X_train, y_train)
    
            # Make predictions
            forecast = model.predict(X_test).tolist()
    
            # Store to the predictions
            predictions.append(forecast)
            
            # Calculate and save the r2 score
            r2_scores[horizon-1].append(r2_score(y_test, forecast))

    # Return the average r2 scores
    r2_average =  []    
    for i in range(horizon_max):
        r2_average.append(np.mean(r2_scores[i]))

    return r2_average, r2_scores, predictions