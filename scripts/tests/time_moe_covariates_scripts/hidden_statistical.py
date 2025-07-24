import numpy as np
from sklearn.metrics import r2_score

# Ridge regression
from sklearn.linear_model import Ridge, LinearRegression

# Random forest
from sklearn.ensemble import RandomForestRegressor

# HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# StandardScaler
from sklearn.preprocessing import StandardScaler

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
                          time_moe_outputs):
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
            X_data = time_moe_outputs[aquifer]
            y_data = aquifer_by_stations[aquifer][target_feature]
        
            for horizon in range (1, horizon_max+1, 1):
                # Get the best features
                # Check if the best features are aquifer specific or not
                if aquifer in best_features.keys():
                    chosen_features = best_features[aquifer][f'horizon_{horizon}']
                else:
                    chosen_features = best_features[f'horizon_{horizon}']
                
                # Define the additional data
                additional_data = aquifer_by_stations[aquifer][chosen_features]

                # Define the train and test set
                X_train = X_data[:-(val_len + test_len + horizon)]
                X_train_additional = additional_data[-len(X_data):-(val_len + test_len + horizon)]
                X_train = np.concatenate((X_train, X_train_additional), axis=1)
                y_train = y_data[-(len(X_data) - horizon):-(val_len + test_len)]
                
                X_test = X_data[-(val_len + test_len + horizon):-(horizon + test_len)]
                X_test_additional = additional_data[-(val_len + test_len + horizon):-(horizon + test_len)]
                X_test = np.concatenate((X_test, X_test_additional), axis=1)
                y_test = y_data[-(val_len + test_len):-test_len]
        
                # Define the scaler
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                
                # Scale the features
                X_train = scaler_X.fit_transform(X_train)
                X_test = scaler_X.transform(X_test)
                y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

                # Train the model
                if model_name != 'gradient_boosting':
                    model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
        
                # Make predictions
                forecast = model.predict(X_test).tolist()

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
                   best_params,
                   time_moe_outputs):
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
        # Get the time moe outputs and the target feature
        X_data = time_moe_outputs[aquifer]
        y_data = aquifer_by_stations[aquifer][target_feature]
    
        for horizon in range (1, horizon_max+1, 1):
            # Get the best features
            # Check if the best features are aquifer specific or not
            if aquifer in best_features.keys():
                chosen_features = best_features[aquifer][f'horizon_{horizon}']
            else:
                chosen_features = best_features[f'horizon_{horizon}']

            # Define the train and test set
            if model_name != 'gradient_boosting':
                # Get the additional data
                additional_data = aquifer_by_stations[aquifer][chosen_features]

                # Define the train and test set
                X_train = X_data[:-(test_len + horizon)]
                X_train_additional = additional_data[-len(X_data):-(test_len + horizon)]
                X_train = np.concatenate((X_train, X_train_additional), axis=1)
                y_train = y_data[-(len(X_data) - horizon):-test_len]
                
                X_test = X_data[-(test_len + horizon):-horizon]
                X_test_additional = additional_data[-(test_len + horizon):-horizon]
                X_test = np.concatenate((X_test, X_test_additional), axis=1)
                y_test = y_data[-test_len:]

                # Define the scaler
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                
                # Scale the features
                X_train = scaler_X.fit_transform(X_train)
                X_test = scaler_X.transform(X_test)
                y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

            else:
                # Define the additional data
                additional_data = aquifer_by_stations[aquifer][chosen_features]
                
                # Define the train and test set
                X_train = X_data[:-(val_len + test_len + horizon)]
                X_train_additional = additional_data[-len(X_data):-(val_len + test_len + horizon)]
                X_train = np.concatenate((X_train, X_train_additional), axis=1)
                y_train = y_data[-(len(X_data) - horizon):-(val_len + test_len)]
                
                X_val = X_data[-(val_len + test_len + horizon):-(horizon + test_len)]
                X_val_additional = additional_data[-(val_len + test_len + horizon):-(horizon + test_len)]
                X_val = np.concatenate((X_val, X_val_additional), axis=1)
                y_val = y_data[-(val_len + test_len):-test_len]

                X_test = X_data[-(test_len + horizon):-horizon]
                X_test_additional = additional_data[-(test_len + horizon):-horizon]
                X_test = np.concatenate((X_test, X_test_additional), axis=1)
                y_test = y_data[-test_len:]
                
                # Define the scaler
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                
                # Scale the features
                X_train = scaler_X.fit_transform(X_train)
                X_val = scaler_X.transform(X_val)
                X_test = scaler_X.transform(X_test)
                y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
                y_val = scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
    
            # Train the model
            if model_name != 'gradient_boosting':
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    
            # Make predictions
            forecast = model.predict(X_test).tolist()

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