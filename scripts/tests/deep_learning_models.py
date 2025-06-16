# IMPORT THE NECESSARY LIBRARIES
# ----------------------------------------------------------------------------------------------------------------------
# N-BEATS, PatchTST
from neuralforecast.models import NBEATS, PatchTST
from neuralforecast.losses.pytorch import HuberLoss
from neuralforecast.core import NeuralForecast
import joblib
import pandas as pd
import numpy as np
import json
from sklearn.metrics import r2_score, make_scorer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
# Hyperparameter tuning
import optuna
# N-HiTS
from darts import TimeSeries
from darts.models import NHiTSModel
from torch.nn import MSELoss
# DeepAR
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.model.deepar import DeepAREstimator
from lightning.pytorch.callbacks import ModelCheckpoint
import os

# HYPERPARAMETER TUNING
# ----------------------------------------------------------------------------------------------------------------------
def hyperparameter_tuning(model_name, horizon_max, aquifers_list, target_feature, val_len, test_len, validation_size, aquifer_by_stations):
    def objective(trial):
        if model_name == 'n_beats':
            input_size = trial.suggest_int('input_size', horizon_max*2, 720)
            
            n_harmonics = trial.suggest_int('n_harmonics', 1, 5)
            n_polynomials = trial.suggest_int('n_polynomials', 1, 5)
            
            scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'robust'])
            learning_rate = trial.suggest_categorical('learning_rate', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        
            max_steps = trial.suggest_categorical('max_steps', [10, 25, 50, 10, 200])
        
            n_blocks_season = trial.suggest_int('n_blocks_season', 1, 3)
            n_blocks_trend = trial.suggest_int('n_blocks_trend', 1, 3)
            n_blocks_identity = trial.suggest_int('n_blocks_identity', 1, 3)
            
            mlp_units_n = trial.suggest_categorical('mlp_units_n', [32, 64, 128, 256, 512])
            num_hidden = trial.suggest_int('num_hidden', 1, 3)
            
            n_blocks = [n_blocks_season, n_blocks_trend, n_blocks_identity]
            mlp_units=[[mlp_units_n, mlp_units_n]]*num_hidden
        
            models = [NBEATS(h=horizon_max,input_size=input_size,
                        loss=HuberLoss(),
                        max_steps=max_steps,
                        learning_rate=learning_rate,
                        n_harmonics=n_harmonics,
                        n_polynomials=n_polynomials,
                        scaler_type=scaler_type,
                        mlp_units=mlp_units,
                        n_blocks=n_blocks,
                        accelerator='cuda',
                        logger=False)
                        ]
            model = NeuralForecast(models=models, freq='D')
        
        elif model_name == 'n_hits':
            input_chunk_length = trial.suggest_int('input_chunk_length', 5, 70)
            output_chunk_length = trial.suggest_int('output_chunk_length', 1, 10)
            num_stacks = trial.suggest_int('num_stacks', 1, 4)
            num_blocks = trial.suggest_int('num_blocks', 1, 3)
            num_layers = trial.suggest_int('num_layers', 2, 5)
            layer_widths = trial.suggest_categorical('layer_widths', [64, 128, 256, 512])
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2])
            learning_rate = trial.suggest_categorical('learning_rate', [1e-2, 1e-3, 1e-4])
            n_epochs = trial.suggest_int('n_epochs', 10, 200)
            
            model = NHiTSModel(input_chunk_length=input_chunk_length,
                             output_chunk_length=output_chunk_length,
                             num_stacks=num_stacks,
                             num_blocks=num_blocks,
                             num_layers=num_layers,
                             layer_widths=layer_widths,
                             dropout=dropout,
                             optimizer_kwargs={'lr': learning_rate},
                             n_epochs=n_epochs,
                             pl_trainer_kwargs={'logger': False, "accelerator": "gpu", "devices": [0]})
        elif model_name == 'deepar':
            num_layers = trial.suggest_int('num_layers', 1, 5)
            hidden_size = trial.suggest_int('hidden_size', 20, 200)
            context_length = trial.suggest_categorical('context_length', [5, 100, 365, 730])
            lr = trial.suggest_categorical('lr', [1e-2, 1e-3, 1e-4])
            weight_decay = trial.suggest_categorical('weight_decay', [1e-7, 1e-8, 1e-9])
            dropout_rate = trial.suggest_categorical('dropout_rate', [0.1, 0.2])
            max_epochs = trial.suggest_int('max_epochs', 10, 100)

            model = DeepAREstimator(prediction_length=horizon_max,
                            freq='D',
                            trainer_kwargs={'accelerator': 'gpu', 'max_epochs': max_epochs, 'logger': False},
                            num_layers=num_layers,
                            hidden_size=hidden_size,
                            context_length=context_length,
                            lr=lr,
                            weight_decay=weight_decay,
                            dropout_rate=dropout_rate)
        elif model_name == 'patchtst':
            input_size = trial.suggest_int('input_size', 5, 100)
            encoder_layers = trial.suggest_categorical('encoder_layers', [2, 4, 6, 8])
            n_heads = trial.suggest_categorical('n_heads', [8, 16, 32])
            hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
            linear_hidden_size = trial.suggest_categorical('linear_hidden_size', [128, 256, 512])
            dropout = trial.suggest_categorical('dropout', [0.1, 0.2])
            fc_dropout = trial.suggest_categorical('fc_dropout', [0.1, 0.2])
            head_dropout = trial.suggest_categorical('head_dropout', [0.1, 0.2])
            attn_dropout = trial.suggest_categorical('attn_dropout', [0.1, 0.2])
            patch_len = trial.suggest_categorical('patch_len', [16, 32, 48, 64])
            stride = trial.suggest_categorical('stride', [8, 16, 24, 32])
            revin = trial.suggest_categorical('revin', [True, False])
            learning_rate = trial.suggest_categorical('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5])
            max_steps = trial.suggest_int('max_steps', 100, 2000)
            
            models = [PatchTST(h=horizon_max,
                               input_size=input_size,
                               encoder_layers=encoder_layers,
                               n_heads=n_heads,
                               hidden_size=hidden_size,
                               linear_hidden_size=linear_hidden_size,
                               dropout=dropout,
                               fc_dropout=fc_dropout,
                               head_dropout=head_dropout,
                               attn_dropout=attn_dropout,
                               patch_len=patch_len,
                               stride=stride,
                               revin=revin,
                               learning_rate=learning_rate,
                               max_steps=max_steps,
                               scaler_type='standard',
                               logger=False)
                         ]
            model = NeuralForecast(models=models, freq='D')
        else:
            raise ValueError(f"Model {model_name} not supported for hyperparameter tuning")
        
        # List for r2 results for different prediction horizons
        r2_scores = [[] for _ in range(horizon_max)]
        
        for aquifer in aquifers_list:
            # List for storing the predictions
            predictions = [[] for _ in range(horizon_max)]
    
            # Get the dataset for the aquifer
            y = aquifer_by_stations[aquifer][:-test_len]
    
            if model_name == 'n_beats':
                # Rename the columns (library wants to have specific names)
                y = y.rename(columns={'date':'ds', target_feature:'y', 'station_id':'unique_id'})
        
                # Only keep these 3 columns
                y = y[['ds', 'y', 'unique_id']]
        
                # Fit the model
                model.fit(y[:-val_len], val_size=validation_size)
            
            elif model_name == 'n_hits':
                # Change to TimeSeries format (required by the library)
                y = TimeSeries.from_dataframe(y, time_col='date', value_cols=target_feature)
                
                # Fit the model
                model.fit(y[:-val_len])

            elif model_name == 'deepar':
                # Scale the data
                scaler = StandardScaler()
                y_temp = y[:-val_len]
                y_temp[target_feature] = scaler.fit_transform(y_temp[[target_feature]])
                
                # Change to TimeSeries format (required by the library)
                y_temp = PandasDataset.from_long_dataframe(y_temp, target=target_feature, item_id='station_id', 
                                                        timestamp='date', freq='D')
                
                # Fit the model
                predictor = model.train(y_temp)

            elif model_name == 'patchtst':
                # Rename the columns (library wants to have specific names)
                y = y.rename(columns={'date':'ds', target_feature:'y', 'station_id':'unique_id'})
                
                # Only keep these 3 columns
                y = y[['ds', 'y', 'unique_id']]
                
                # Fit the model
                model.fit(y[:-val_len], val_size=validation_size)

            else:
                raise ValueError(f"Model {model_name} not supported for hyperparameter tuning")
            
            # Iterate from day_len days before the end, to the last day
            for i in range(val_len + (horizon_max-1), 0, -1):
                
                # Predict
                if model_name == 'n_beats':
                    forecast = model.predict(df=y[:-i], verbose=0)
                
                elif model_name == 'n_hits':
                    forecast = model.predict(n=horizon_max, series=y[:-i])
                
                elif model_name == 'deepar':
                    # Scale the data
                    #scaler = StandardScaler()
                    y_temp = y[:-i]
                    y_temp['altitude_diff'] = scaler.transform(y_temp[['altitude_diff']])
                    y_temp = PandasDataset.from_long_dataframe(y_temp, target='altitude_diff', item_id='station_id', 
                                                                    timestamp='date', freq='D')
                    forecast = list(predictor.predict(y_temp))
                    forecast = scaler.inverse_transform([forecast[0].samples.mean(axis=0)])[0]
                
                elif model_name == 'patchtst':
                    forecast = model.predict(df=y[:-i])
               
                else:
                    raise ValueError(f"Model {model_name} not supported for hyperparameter tuning")
    
                
                # Store the results for every prediction horizon separately
                if model_name == 'n_beats':
                    for i in range(horizon_max):
                        predictions[i].append(forecast['NBEATS'].values[i])
                
                elif model_name == 'n_hits':
                    for i in range(horizon_max):
                        predictions[i].append(forecast.values()[i][0])
                
                elif model_name == 'deepar':
                    for i in range(horizon_max):
                        predictions[i].append(forecast[i])
                
                elif model_name == 'patchtst':
                    for i in range(horizon_max):
                        predictions[i].append(forecast['PatchTST'].values[i])
                
                else:
                    raise ValueError(f"Model {model_name} not supported for hyperparameter tuning")
            
            # Clean up the results
            for i in range(horizon_max):
                if i == 0:
                    predictions[i] = predictions[i][-val_len:]
                else:
                    predictions[i] = predictions[i][(horizon_max-i-1):-i]            
    
            # Calculate the r2 scores and store them in a list
            for i in range(horizon_max):
                if model_name == 'n_beats' or model_name == 'patchtst':
                    r2_scores[i].append(r2_score(y['y'][-val_len:], predictions[i]))
                else:
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
def final_training(model_name, aquifers_list, test_len, validation_size, horizon_max, target_feature, aquifer_by_stations, best_params):
    if model_name == 'n_beats':
        models = [NBEATS(h=horizon_max, 
                         loss=HuberLoss(),
                         accelerator='cuda',
                         input_size=best_params['input_size'],
                         n_harmonics=best_params['n_harmonics'],
                         n_polynomials=best_params['n_polynomials'],
                         scaler_type=best_params['scaler_type'],
                         learning_rate=best_params['learning_rate'],
                         max_steps=best_params['max_steps'],
                         n_blocks=[best_params['n_blocks_season'], best_params['n_blocks_trend'], best_params['n_blocks_identity']],
                         mlp_units=[[best_params['mlp_units_n'], best_params['mlp_units_n']]]*best_params['num_hidden'],
                         devices=[0],
                         logger=False)]
        model = NeuralForecast(models=models, freq='D')

    elif model_name == 'n_hits':
        model = NHiTSModel(
            input_chunk_length=best_params['input_chunk_length'],
            output_chunk_length=best_params['output_chunk_length'],
            num_blocks=best_params['num_blocks'],
            num_stacks=best_params['num_stacks'],
            num_layers=best_params['num_layers'],
            layer_widths=best_params['layer_widths'],
            dropout=best_params['dropout'],
            n_epochs=best_params['n_epochs'],
            optimizer_kwargs={'lr': best_params['learning_rate']},
            pl_trainer_kwargs={'logger': False, "accelerator": "gpu", "devices": [0]}
        )
    
    elif model_name == 'deepar':
        model = DeepAREstimator(prediction_length=horizon_max,
                         freq='D',
                         trainer_kwargs={'accelerator': 'gpu', 'max_epochs': 42, 'logger': False},
                         num_layers=best_params['num_layers'],
                         hidden_size=best_params['hidden_size'],
                         context_length=best_params['context_length'],
                         lr=best_params['lr'],
                         weight_decay=best_params['weight_decay'],
                         dropout_rate=best_params['dropout_rate'])
    
    elif model_name == 'patchtst':
        model = PatchTST(h=horizon_max,
                         input_size=best_params['input_size'],
                         encoder_layers=best_params['encoder_layers'],
                         n_heads=best_params['n_heads'],
                         hidden_size=best_params['hidden_size'],
                         linear_hidden_size=best_params['linear_hidden_size'],
                         dropout=best_params['dropout'],
                         fc_dropout=best_params['fc_dropout'],
                         head_dropout=best_params['head_dropout'],
                         attn_dropout=best_params['attn_dropout'],
                         patch_len=best_params['patch_len'],
                         stride=best_params['stride'],
                         revin=best_params['revin'],
                         learning_rate=best_params['learning_rate'],
                         max_steps=best_params['max_steps'],
                         scaler_type='standard',
                         logger=False)
        
        nf = NeuralForecast(
            models=[model],
            freq='D'
        )
    
    else:
        raise ValueError(f"Model {model_name} not supported for final training")
    
    # List for r2 results for different prediction horizons
    r2_scores = [[] for _ in range(horizon_max)]
    
    # Dictionary for storing the predictions
    predictions_by_stations = {key: [] for key in aquifers_list}
    
    for aquifer in aquifers_list:
        # List for storing the predictions
        predictions = [[] for _ in range(horizon_max)]
    
        # Get the dataset for the aquifer
        y = aquifer_by_stations[aquifer]
    
        if model_name == 'n_beats':
            # Rename the columns (library wants to have specific names)
            y = y.rename(columns={'date':'ds', target_feature:'y', 'station_id':'unique_id'})
            # Only keep these 3 columns
            y = y[['ds', 'y', 'unique_id']]
    
            # Fit the model
            model.fit(y[:-test_len], val_size=validation_size)

        elif model_name == 'n_hits':
            # Change the format to TimeSeries
            y = TimeSeries.from_dataframe(y, time_col='date', value_cols=target_feature)
            # Fit the model
            model.fit(y[:-test_len])

        elif model_name == 'deepar':
            # Scale the data
            scaler = StandardScaler()
            y_temp = y[:-test_len]
            y_temp[target_feature] = scaler.fit_transform(y_temp[[target_feature]])
            
            # Change to TimeSeries format (required by the library)
            y_temp = PandasDataset.from_long_dataframe(y_temp, target=target_feature, item_id='station_id', 
                                                            timestamp='date', freq='D')
            # Fit the model
            predictor = model.train(y_temp)

        elif model_name == 'patchtst':
            # Rename the columns (library wants to have specific names)
            y = y.rename(columns={'date':'ds', target_feature:'y', 'station_id':'unique_id'})
            # Only keep these 3 columns
            y = y[['ds', 'y', 'unique_id']]
            y_train = y[:-test_len]
            nf.fit(y_train, val_size=validation_size)

        else:
            raise ValueError(f"Model {model_name} not supported for final training")
        
        # Iterate from day_len days before the end, to the last day
        for i in range(test_len + (horizon_max-1), 0, -1):
            # Predict
            if model_name == 'n_beats':
                forecast = model.predict(df=y[:-i], verbose=0)                 
            
            elif model_name == 'n_hits':
                forecast = model.predict(n=horizon_max, series=y[:-i])
            
            elif model_name == 'deepar':
                #scaler = StandardScaler()
                y_temp = y[:-i]
                y_temp[target_feature] = scaler.transform(y_temp[[target_feature]])
                y_temp = PandasDataset.from_long_dataframe(y_temp, target=target_feature, item_id='station_id', 
                                                                timestamp='date', freq='D')
                forecast = list(predictor.predict(y_temp))
                forecast = scaler.inverse_transform([forecast[0].samples.mean(axis=0)])[0]
            
            elif model_name == 'patchtst':
                forecast = nf.predict(df=y[:-i])
            
            else:
                raise ValueError(f"Model {model_name} not supported for final training")
    
            # Store the results for every prediction horizon separately
            for i in range(horizon_max):
                if model_name == 'n_beats':
                    predictions[i].append(forecast['NBEATS'].values[i])
                elif model_name == 'n_hits':
                    predictions[i].append(forecast.values()[i][0])
                elif model_name == 'deepar':
                    predictions[i].append(forecast[i])
                elif model_name == 'patchtst':
                    predictions[i].append(forecast['PatchTST'].values[i])
                else:
                    raise ValueError(f"Model {model_name} not supported for final training")
        
        # Clean up the results
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
with open('experiment_settings/deep_learning_models_experiment_settings.json', 'r') as file:
    experiment_settings = json.load(file)

# Load the data
aquifer_by_stations = joblib.load('../../data/interim/ground-water-and-weather-with-forecasts-and-additional-features.joblib')

for aquifer in aquifer_by_stations.keys():
    aquifer_by_stations[aquifer] = aquifer_by_stations[aquifer][['altitude_diff', 'date', 'station_id']]


# TESTING THE MODELS
# ----------------------------------------------------------------------------------------------------------------------
for name, settings in experiment_settings.items():
    # Get the model name
    model_name = experiment_settings[name]['model_name']

    # Variable that tells us if the model is multivariate
    multivariate = False
    best_features = {}

    # Hyperparameter tuning
    best_params = hyperparameter_tuning(model_name, settings['horizon_max'], settings['aquifers_list'], 
                                        settings['target_feature'], settings['val_len'], settings['test_len'], 
                                        settings['validation_size'], aquifer_by_stations)
        
    # Final training
    r2_average, r2_scores, predictions = final_training(model_name, settings['aquifers_list'], 
                                                        settings['test_len'], settings['validation_size'], 
                                                        settings['horizon_max'], settings['target_feature'], 
                                                        aquifer_by_stations, best_params)
    
    # Obtain the index of the file name (so every experiment has a unique name)
    index = get_index(folder_path='../results/deep_learning_models', file_name=name)

    # Save the results
    file_path = f'../results/deep_learning_models/{name}_{index}.json'
    save_results(model_name, multivariate, r2_scores, predictions, best_features, best_params, file_path)

    # Print the results
    print("--------------------------------------------------------------------------------------------------")
    print(f"Model: {name}")
    print(f"R2 average: {r2_average}")
    print("--------------------------------------------------------------------------------------------------\n\n")