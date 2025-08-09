import os
import optuna

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoModelForCausalLM
from torch.utils.data import TensorDataset, DataLoader

import joblib

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler

# Warnings
import warnings
warnings.filterwarnings('ignore')

# CONSTANTS
# ----------------------------------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIME_MOE_OUTPUTS_DIM = 768
MAX_EPOCHS = 5000

# DATA PREPROCESSING FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
# Function that removes the redundant columns from the weather data (removes the target variable ...)
def weather_data_clean(weather_data):
    columns_to_drop = ['date', 'station_id', 'id', 'location_id']
    for column in weather_data.columns:
        if 'altitude' in column:
            columns_to_drop.append(column)
    weather_data = weather_data.drop(columns=columns_to_drop)
    return weather_data

# Convert bool columns to float
def bool_to_float(dataframe):
    for column in dataframe.columns:
        if dataframe[column].dtype == 'bool':
            dataframe[column] = dataframe[column].astype(float)

    return dataframe

# Create sequences from the data
def create_sequences(time_moe_outputs, weather, targets, horizon_max):
    time_moe_list = []
    weather_list = []
    target_list = []

    for i in range(len(weather)-horizon_max):
            time_moe_list.append(time_moe_outputs[i])
            weather_list.append(weather[i])
            target_list.append(targets[i+1 : i+1+horizon_max])

    return time_moe_list, weather_list, target_list

# Create sequences for the test set
def create_sequences_test(time_moe_outputs, weather):
    time_moe_list = []
    weather_list = []

    for i in range(len(weather)-1):
        time_moe_list.append(time_moe_outputs[i])
        weather_list.append(weather[i])

    return time_moe_list, weather_list

# WEATHER MODEL
# ----------------------------------------------------------------------------------------------------------------------
class WeatherMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)  # [batch, output_dim]
    

def train_weather_model(weather_model, time_moe_model, train_loader, val_loader, optimizer, scheduler, loss_fn, max_epochs, horizon_max, weights, patience, max_norm):
    weather_model.train()
    best_validation_loss_sum = float('inf')
    patience_counter = 0
    for epoch in range(max_epochs):
        # Iterate over the train set
        for time_moe_output_batch, weather_batch, target_batch in train_loader:
            weather_batch = weather_batch.to(device)
            time_moe_output_batch = time_moe_output_batch.to(device)
            target_batch = target_batch.to(device)
    
            weather_embeddings = weather_model(weather_batch)
    
            head_input = (time_moe_output_batch + weather_embeddings) / 2
    
            # Call the predictin head
            prediction = time_moe_model.lm_heads[1](head_input)[:, :horizon_max]
    
            # Weighted loss
            losses_per_horizon = []
            for i in range(horizon_max):
                loss_i = loss_fn(prediction[:, i], target_batch[:, i])
                losses_per_horizon.append(loss_i)
    
            loss = sum(w * l for w, l in zip(weights, losses_per_horizon))/sum(weights)
    
            # Backpropagate the loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(weather_model.parameters()), max_norm=max_norm)
            optimizer.step()
    
        # Validation loss
        weather_model.eval()

        with torch.no_grad():
            validation_loss_sum = 0
            batch_count = 0

            # Iterate over the val set
            for time_moe_output_batch, weather_batch, target_batch in val_loader:
                batch_count += 1
                weather_batch = weather_batch.to(device)
                time_moe_output_batch = time_moe_output_batch.to(device)
                target_batch = target_batch.to(device)

                weather_embeddings = weather_model(weather_batch)
                
                head_input = (time_moe_output_batch + weather_embeddings) / 2

                # Call the predictin head
                prediction = time_moe_model.lm_heads[1](head_input)[:, :horizon_max]

                # Try weighted loss with the emphasis on the higher horizons
                losses_per_horizon = []
                for i in range(horizon_max):
                    loss_i = loss_fn(prediction[:, i], target_batch[:, i])
                    losses_per_horizon.append(loss_i)

                loss_validation = sum(w * l for w, l in zip(weights, losses_per_horizon))/sum(weights)

                validation_loss_sum += loss_validation.item()

            # Scheduler step (with validation loss)
            scheduler.step(validation_loss_sum/batch_count)

            # Early stopping check
            if validation_loss_sum < best_validation_loss_sum:
                best_validation_loss_sum = validation_loss_sum
                patience_counter = 0
                # Save best model
                best_weather_model_state = weather_model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                # Load best model
                weather_model.load_state_dict(best_weather_model_state)
                return weather_model
            
        weather_model.train()

    return weather_model
    

# HYPERPARAMETER TUNING
# ----------------------------------------------------------------------------------------------------------------------
def hyperparameter_tuning(model_name,
                          horizon_max,
                          aquifers_list,
                          target_feature,
                          val_len,
                          test_len,
                          aquifer_by_stations,
                          time_moe_outputs):
    def objective(trial):
        if model_name == 'weather_mlp':
            hidden_dim = trial.suggest_categorical('hidden_dim', [16, 32, 64, 128, 256, 768, 1024])
            dropout = trial.suggest_categorical('dropout', [0.0, 0.1, 0.2, 0.5])
            patience = trial.suggest_categorical('patience', [5, 10, 25, 50, 100])
            scheduler_patience_multiplier = trial.suggest_categorical('scheduler_patience_multiplier', [0.1, 0.3, 0.5, 0.7])
            lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
            weight_decay = trial.suggest_categorical('weight_decay', [0, 1e-5, 1e-4, 1e-3, 1e-2, 5e-1, 1e-1])
            delta = trial.suggest_categorical('delta', [0.1, 0.5, 1, 2, 5])
            scheduler_factor = trial.suggest_categorical('scheduler_factor', [0.1, 0.3, 0.5, 0.7])
            weights = trial.suggest_categorical('weights', [
                [1, 1, 1, 1, 1],           # uniform
                [1, 1.5, 2.5, 4, 6],      # progressive
                [1, 2, 3, 4, 5],          # linear increase
                [1, 1.2, 1.5, 2, 3],      # moderate increase
                [2, 1.8, 1.5, 1.2, 1],    # decreasing (recent more important)
                [1, 1, 2, 4, 8],          # exponential increase
                [6, 4, 2.5, 1.5, 1]       # exponential decrease
            ])
            max_norm = trial.suggest_categorical('max_norm', [0.1, 0.5, 1, 2, 5])
        else:
            raise ValueError(f"Model {model_name} not supported for hyperparameter tuning")
        
        # Initialize the time_moe_model
        time_moe_model = AutoModelForCausalLM.from_pretrained(
            'Maple728/TimeMoE-200M',
            device_map="cuda",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
            trust_remote_code=True,
        ).to(device)
        
        # Freeze the parameters of the time_moe_model.lm_heads
        for param in time_moe_model.lm_heads[1].parameters():
            param.requires_grad = False

        # List for r2 scores
        r2_scores = []

        for aquifer in aquifers_list:            
            # Define the scaler
            weather_scaler = StandardScaler()
            y_scaler = StandardScaler()
            
            # Define the train, validation and test sets
            time_moe_outputs_train = time_moe_outputs[aquifer][:-(val_len+test_len)]
            time_moe_outputs_val = time_moe_outputs[aquifer][-(val_len+test_len):-test_len]
            #time_moe_outputs_test = time_moe_outputs[aquifer][-(test_len+horizon_max):]
            
            weather_train = bool_to_float(weather_data_clean(aquifer_by_stations[aquifer][-len(time_moe_outputs[aquifer]):-(val_len + test_len)]))
            weather_val = bool_to_float(weather_data_clean(aquifer_by_stations[aquifer][-(val_len+test_len):-test_len]))
            #weather_test = bool_to_float(weather_data_clean(aquifer_by_stations[aquifer][-(test_len+horizon_max):]))
            weather_train = weather_scaler.fit_transform(weather_train)
            weather_val = weather_scaler.transform(weather_val)
            #weather_test = weather_scaler.transform(weather_test)
            
            y_train = aquifer_by_stations[aquifer][target_feature][-len(time_moe_outputs[aquifer]):-(val_len+test_len)]
            y_val = aquifer_by_stations[aquifer][target_feature][-(val_len+test_len):-test_len]
            y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
            y_val = y_scaler.transform(y_val.values.reshape(-1, 1)).flatten()
            #y_test = aquifer_by_stations[aquifer][target_feature][-test_len:]
            
            # Create sequences
            time_moe_outputs_train, weather_train, y_train = create_sequences(time_moe_outputs_train, weather_train, y_train, horizon_max)
            time_moe_outputs_val, weather_val, y_val = create_sequences(time_moe_outputs_val, weather_val, y_val, horizon_max)
            #time_moe_outputs_test, weather_test = create_sequences_test(time_moe_outputs_test, weather_test)
            
            # Convert to tensors
            time_moe_outputs_train = torch.tensor(time_moe_outputs_train, dtype=torch.float32)
            weather_train = torch.tensor(weather_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            
            time_moe_outputs_val = torch.tensor(time_moe_outputs_val, dtype=torch.float32)
            weather_val = torch.tensor(weather_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)
            
            #time_moe_outputs_test = torch.tensor(time_moe_outputs_test, dtype=torch.float32)
            #weather_test = torch.tensor(weather_test, dtype=torch.float32)
            
            # Create torch datasets and dataloaders
            train_dataset = TensorDataset(time_moe_outputs_train, weather_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
            
            val_dataset = TensorDataset(time_moe_outputs_val, weather_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
            
            #test_dataset = TensorDataset(time_moe_outputs_test, weather_test)
            #test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            
            # Define the models
            weather_model = WeatherMLP(input_dim=weather_train.shape[1], 
                                       output_dim=TIME_MOE_OUTPUTS_DIM,
                                       hidden_dim=hidden_dim,
                                       dropout_rate=dropout).to(device)
            
            # Define the optimizer
            optimizer = torch.optim.AdamW([
                {'params': weather_model.parameters(), 'lr': lr, 'weight_decay': weight_decay}
            ])
            scheduler_patience = patience * scheduler_patience_multiplier
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True)
            
            # Define the loss function
            loss_fn = nn.HuberLoss(delta=delta)

            weather_model = train_weather_model(weather_model=weather_model,
                                                time_moe_model=time_moe_model,
                                                train_loader=train_loader,
                                                val_loader=val_loader,
                                                optimizer=optimizer,
                                                scheduler=scheduler,
                                                loss_fn=loss_fn,
                                                max_epochs=MAX_EPOCHS,
                                                horizon_max=horizon_max,
                                                weights=weights,
                                                patience=patience,
                                                max_norm=max_norm)
            
            # Make predictions
            weather_model.eval()
            with torch.no_grad():
                target_list = []
                predictions_list = []
                # Iterate over batches            
                for time_moe_output_batch, weather_batch, target_batch in val_loader:
                    weather_batch = weather_batch.to(device)
                    time_moe_output_batch = time_moe_output_batch.to(device)
            
                    weather_embeddings = weather_model(weather_batch)
                    
                    head_input = (time_moe_output_batch + weather_embeddings) / 2
            
                    # Call the predictin head
                    prediction = time_moe_model.lm_heads[1](head_input)[:, :horizon_max].cpu().numpy()
                    prediction = y_scaler.inverse_transform(prediction)
            
                    # Store the targets and the predictions (for the r2 score calculation)
                    target_list.append(y_scaler.inverse_transform(target_batch.cpu().numpy()))
                    predictions_list.append(prediction)
                
                # Concatenate all batches
                target_list = np.concatenate(target_list, axis=0)
                predictions_list = np.concatenate(predictions_list, axis=0)

            
            # Calculate the r2 scores and store them in a list
            r2_scores.append(r2_score(target_list, predictions_list))
        
        # Calculate the average r2 score
        r2_average = np.mean(r2_scores)
    
        # Set the loss as average of average r2 scores for different prediction horizons
        loss = r2_average
    
        return loss
    
    # Run the optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1)

    # Return the best parameters
    return study.best_params


# FINAL TRAINING
# ----------------------------------------------------------------------------------------------------------------------
def final_training(model_name,
                   horizon_max,
                   aquifers_list,
                   target_feature,
                   val_len,
                   test_len,
                   aquifer_by_stations,
                   time_moe_outputs,
                   best_params):
    

    if model_name == 'weather_mlp':
        pass
    else:
        raise ValueError(f"Model {model_name} not supported for final training")
    
    # Initialize the time_moe_model
    time_moe_model = AutoModelForCausalLM.from_pretrained(
        'Maple728/TimeMoE-200M',
        device_map="cuda",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
        trust_remote_code=True,
    ).to(device)
    
    # Freeze the parameters of the time_moe_model.lm_heads
    for param in time_moe_model.lm_heads[1].parameters():
        param.requires_grad = False

    # List for r2 results for different prediction horizons
    r2_scores = [[] for _ in range(horizon_max)]
    
    # Dictionary for storing the predictions
    predictions_by_stations = {key: [] for key in aquifers_list}

    for aquifer in aquifers_list:  
        # List for storing the predictions
        predictions = [[] for _ in range(horizon_max)]

        # Define the scaler
        weather_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        # Define the train, validation and test sets
        time_moe_outputs_train = time_moe_outputs[aquifer][:-(val_len+test_len)]
        time_moe_outputs_val = time_moe_outputs[aquifer][-(val_len+test_len):-test_len]
        time_moe_outputs_test = time_moe_outputs[aquifer][-(test_len+horizon_max):]
        
        weather_train = bool_to_float(weather_data_clean(aquifer_by_stations[aquifer][-len(time_moe_outputs[aquifer]):-(val_len + test_len)]))
        weather_val = bool_to_float(weather_data_clean(aquifer_by_stations[aquifer][-(val_len+test_len):-test_len]))
        weather_test = bool_to_float(weather_data_clean(aquifer_by_stations[aquifer][-(test_len+horizon_max):]))
        weather_train = weather_scaler.fit_transform(weather_train)
        weather_val = weather_scaler.transform(weather_val)
        weather_test = weather_scaler.transform(weather_test)
        
        y_train = aquifer_by_stations[aquifer][target_feature][-len(time_moe_outputs[aquifer]):-(val_len+test_len)]
        y_val = aquifer_by_stations[aquifer][target_feature][-(val_len+test_len):-test_len]
        y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_val = y_scaler.transform(y_val.values.reshape(-1, 1)).flatten()
        y_test = aquifer_by_stations[aquifer][target_feature][-test_len:]
        
        # Create sequences
        time_moe_outputs_train, weather_train, y_train = create_sequences(time_moe_outputs_train, weather_train, y_train, horizon_max)
        time_moe_outputs_val, weather_val, y_val = create_sequences(time_moe_outputs_val, weather_val, y_val, horizon_max)
        time_moe_outputs_test, weather_test = create_sequences_test(time_moe_outputs_test, weather_test)
        
        # Convert to tensors
        time_moe_outputs_train = torch.tensor(time_moe_outputs_train, dtype=torch.float32)
        weather_train = torch.tensor(weather_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        
        time_moe_outputs_val = torch.tensor(time_moe_outputs_val, dtype=torch.float32)
        weather_val = torch.tensor(weather_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        
        time_moe_outputs_test = torch.tensor(time_moe_outputs_test, dtype=torch.float32)
        weather_test = torch.tensor(weather_test, dtype=torch.float32)
        
        # Create torch datasets and dataloaders
        train_dataset = TensorDataset(time_moe_outputs_train, weather_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        
        val_dataset = TensorDataset(time_moe_outputs_val, weather_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        
        test_dataset = TensorDataset(time_moe_outputs_test, weather_test)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        # Define the models
        weather_model = WeatherMLP(input_dim=weather_train.shape[1], 
                                    output_dim=TIME_MOE_OUTPUTS_DIM,
                                    hidden_dim=best_params['hidden_dim'],
                                    dropout_rate=best_params['dropout']).to(device)
        
        # Define the optimizer
        optimizer = torch.optim.AdamW([
            {'params': weather_model.parameters(), 'lr': best_params['lr'], 'weight_decay': best_params['weight_decay']}
        ])
        scheduler_patience = best_params['patience'] * best_params['scheduler_patience_multiplier']
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=best_params['scheduler_factor'], patience=scheduler_patience, verbose=True)
        
        # Define the loss function
        loss_fn = nn.HuberLoss(delta=best_params['delta'])

        weather_model = train_weather_model(weather_model=weather_model,
                                            time_moe_model=time_moe_model,
                                            train_loader=train_loader,
                                            val_loader=val_loader,
                                            optimizer=optimizer,
                                            scheduler=scheduler,
                                            loss_fn=loss_fn,
                                            max_epochs=MAX_EPOCHS,
                                            horizon_max=horizon_max,
                                            weights=best_params['weights'],
                                            patience=best_params['patience'],
                                            max_norm=best_params['max_norm'])
        
        # Make predictions
        weather_model.eval()
        with torch.no_grad():
            # Iterate over the batches        
            for time_moe_output_batch, weather_batch in test_loader:
                weather_batch = weather_batch.to(device)
                time_moe_output_batch = time_moe_output_batch.to(device)
        
                weather_embeddings = weather_model(weather_batch)

                head_input = (time_moe_output_batch + weather_embeddings) / 2
        
                # Call the predictin head
                prediction = time_moe_model.lm_heads[1](head_input)[:, :horizon_max].cpu().numpy()
                prediction = y_scaler.inverse_transform(prediction)
        
                # Store the results for every prediction horizon separately
                for i in range(horizon_max):
                    predictions[i] += prediction[:, i].tolist()           
        
        # Clean up the results
        for i in range(horizon_max):
            if i == 0:
                predictions[i] = predictions[i][-test_len:]
            else:
                predictions[i] = predictions[i][horizon_max-i-1:-i]

        # Add the predictios to the dictionary
        predictions_by_stations[aquifer] = predictions
        
        # Calculate the r2 scores and store them in a list
        for i in range(horizon_max):
            r2_scores[i].append(r2_score(y_test, predictions[i]))

    # Return the average r2 scores
    r2_average =  []    
    for i in range(horizon_max):
        r2_average.append(np.mean(r2_scores[i]))
    
    # Return the predictions and the r2 scores
    return r2_average, r2_scores, predictions_by_stations
