import numpy as np
from sklearn.metrics import r2_score

# Torch
import torch
import torch.nn as nn

# StandardScaler
from sklearn.preprocessing import StandardScaler

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Optuna
import optuna

#SelectKBest
from sklearn.feature_selection import SelectKBest, f_regression

# CONSTANTS
# ----------------------------------------------------------------------------------------------------------------------
MAX_EPOCHS = 5000

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
        if model_name == 'mlp':
            hidden_layers = trial.suggest_categorical('hidden_layers', [[64], [32],
                                                                        [32, 16], [64, 32], [128, 64],
                                                                        [128, 64, 32], [64, 32, 16], [256, 128, 64],
                                                                        [128, 64, 32, 16], [256, 128, 64, 32], [512, 256, 128, 64]])
            dropout = trial.suggest_categorical('dropout', [0.0, 0.1, 0.2, 0.5])
            patience = trial.suggest_categorical('patience', [5, 10, 25,50])
            lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
            weight_decay = trial.suggest_loguniform('weight_decay', [1e-4, 1e-3, 1e-2, 5e-1, 1e-1, 0, 1, 10])


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

                # Convert to PyTorch tensors
                X_train_tensor = torch.FloatTensor(X_train)
                y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
                X_test_tensor = torch.FloatTensor(X_test)
                y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

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
                   best_params,
                   time_moe_outputs):
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
                            X_val=X_test_tensor,
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