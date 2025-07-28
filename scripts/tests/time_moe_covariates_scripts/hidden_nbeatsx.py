# IMPORT THE NECESSARY LIBRARIES
# ----------------------------------------------------------------------------------------------------------------------
# N-BEATS, PatchTST
from neuralforecast.models import NBEATS, PatchTST, NBEATSx, NHITS
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
# Warnings
import warnings
warnings.filterwarnings('ignore')
# Import torch
import torch
# Import time
import time

# CONSTANTS
# ----------------------------------------------------------------------------------------------------------------------
hist_exog_list = ['altitude',
                  'day_time',
                  'precipitation',
                  'snow_accumulation',
                  'temperature_avg',
                  'temperature_min',
                  'temperature_max',
                  'cloud_cover_avg',
                  'cloud_cover_min',
                  'cloud_cover_max',
                  'humidity_avg',
                  'humidity_min',
                  'humidity_max',
                  'uv_index_avg',
                  'uv_index_min',
                  'uv_index_max',
                  'precipitation_probability_avg',
                  'precipitation_probability_min',
                  'precipitation_probability_max',
                  'precipitation_intensity_avg',
                  'precipitation_intensity_min',
                  'precipitation_intensity_max',
                  'tn_0',
                  'tx_0',
                  'nn_decodeText_0',
                  'rr_decodeText_0',
                  'ff_decodeText_0']

additional_parameters_list = [['tn_1', 'tx_1', 'nn_decodeText_1', 'rr_decodeText_1', 'ff_decodeText_1'],
                               ['tn_2', 'tx_2', 'nn_decodeText_2', 'rr_decodeText_2', 'ff_decodeText_2'],
                               ['tn_3', 'tx_3', 'nn_decodeText_3', 'rr_decodeText_3', 'ff_decodeText_3'],
                               ['tn_4', 'tx_4', 'nn_decodeText_4', 'rr_decodeText_4', 'ff_decodeText_4'],
                               ['tn_5', 'tx_5', 'nn_decodeText_5', 'rr_decodeText_5', 'ff_decodeText_5']]

DATE = 'date' # Name of the date column
STATION_ID = 'station_id' # Name of the station id column

# HYPERPARAMETER TUNING
# ----------------------------------------------------------------------------------------------------------------------
def hyperparameter_tuning(model_name,
                          horizon_max,
                          aquifers_list,
                          target_feature,
                          val_len,
                          test_len,
                          aquifer_by_stations,
                          time_moe_outputs,
                          time_moe_features,
                          additional_parameters_list=additional_parameters_list,
                          hist_exog_list=hist_exog_list,):
    def objective(trial):
        if model_name == 'n_beats_x':
            input_size = trial.suggest_categorical('input_size', [30, 60, 180, 365, 730])
            n_harmonics = trial.suggest_int('n_harmonics', 1, 5)
            n_polynomials = trial.suggest_int('n_polynomials', 1, 5)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
            max_steps = trial.suggest_categorical('max_steps', [200, 500, 1000])
            dropout_prob_theta = trial.suggest_float('dropout_prob_theta', 0.0, 0.3)
            weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)

            # Initialize the models
            models_list = []
            
            # Fill the models list with the models
            for i in range(horizon_max):
                stack_types = ['identity', 'exogenous'] if (i+1) == 1 else ['seasonality', 'trend', 'identity', 'exogenous']
                n_blocks = [1 for _ in range(len(stack_types))]
                models = [NBEATSx(h=i+1, 
                                accelerator='cuda',
                                input_size=input_size,
                                stack_types=stack_types,
                                n_harmonics=n_harmonics,
                                n_polynomials=n_polynomials,
                                learning_rate=learning_rate,
                                max_steps=max_steps,
                                hist_exog_list=hist_exog_list,
                                futr_exog_list=additional_parameters_list[i],
                                devices=[0],
                                logger=False,
                                scaler_type='standard',
                                dropout_prob_theta=dropout_prob_theta,
                                n_blocks=n_blocks,
                                optimizer=torch.optim.Adam,
                                optimizer_kwargs={'weight_decay': weight_decay})]
                model = NeuralForecast(models=models, freq='D')
                models_list.append(model)

        elif model_name == 'n_hits_multivariate':
            input_size = trial.suggest_categorical('input_size', [30, 60, 180, 365, 730])
            stack_types = trial.suggest_categorical('stack_types', [['identity', 'identity'],
                                                                    ['identity', 'identity', 'identity'], 
                                                                    ['identity', 'identity', 'identity', 'identity']])
            
            # n_blocks
            if len(stack_types) == 2:
                n_blocks = trial.suggest_categorical('n_blocks_2', [[1, 1], [2, 2]])
            elif len(stack_types) == 3:
                n_blocks = trial.suggest_categorical('n_blocks_3', [[1, 1, 1], [2, 2, 2]])
            elif len(stack_types) == 4:
                n_blocks = trial.suggest_categorical('n_blocks_4', [[1, 1, 1, 1], [2, 2, 2, 2]])
            else:
                raise ValueError(f"Stack types {stack_types} not supported")
            
            # n_freq_downsample
            if len(stack_types) == 2:
                n_freq_downsample = trial.suggest_categorical('n_freq_downsample_2', [[2, 1], [3, 1], [4, 1]])
            elif len(stack_types) == 3:
                n_freq_downsample = trial.suggest_categorical('n_freq_downsample_3', [[4, 2, 1], [3, 2, 1], [2, 1, 1]])
            elif len(stack_types) == 4:
                n_freq_downsample = trial.suggest_categorical('n_freq_downsample_4', [[4, 3, 2, 1], [3, 2, 1, 1], [2, 1, 1, 1]])
            else:
                raise ValueError(f"Stack types {stack_types} not supported")

            # mlp_units
            if len(stack_types) == 2:
                mlp_units = trial.suggest_categorical('mlp_units_2', [[[256, 256], [256, 256]], [[128, 128], [128, 128]], [[512, 512], [512, 512]], [[1024, 1024], [1024, 1024]]])
            elif len(stack_types) == 3:
                mlp_units = trial.suggest_categorical('mlp_units_3', [[[256, 256], [256, 256], [256, 256]], [[128, 128], [128, 128], [128, 128]], [[512, 512], [512, 512], [512, 512]], [[1024, 1024], [1024, 1024], [1024, 1024]]])
            elif len(stack_types) == 4:
                mlp_units = trial.suggest_categorical('mlp_units_4', [[[256, 256], [256, 256], [256, 256], [256, 256]], [[128, 128], [128, 128], [128, 128], [128, 128]], [[512, 512], [512, 512], [512, 512], [512, 512]], [[1024, 1024], [1024, 1024], [1024, 1024], [1024, 1024]]])
            else:
                raise ValueError(f"Stack types {stack_types} not supported")

            # n_pool_kernel_size
            if len(stack_types) == 2:
                n_pool_kernel_size = trial.suggest_categorical('n_pool_kernel_size_2', [[2, 1], [3, 1], [4, 1]])
            elif len(stack_types) == 3:
                n_pool_kernel_size = trial.suggest_categorical('n_pool_kernel_size_3', [[4, 2, 1], [3, 2, 1], [2, 1, 1]])
            elif len(stack_types) == 4:
                n_pool_kernel_size = trial.suggest_categorical('n_pool_kernel_size_4', [[4, 3, 2, 1], [3, 2, 1, 1], [2, 1, 1, 1]])
            else:
                raise ValueError(f"Stack types {stack_types} not supported")
            
            dropout_prob_theta = trial.suggest_float('dropout_prob_theta', 0.0, 0.3)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
            max_steps = trial.suggest_categorical('max_steps', [100, 200, 500, 1000])
            dropout_prob_theta = trial.suggest_float('dropout_prob_theta', 0.0, 0.3)
            weight_decay = trial.suggest_categorical('weight_decay', [0.0, 1e-5, 1e-4, 1e-3, 1e-2])
            activation = trial.suggest_categorical('activation', ['ReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'PReLU', 'Sigmoid'])
            
            # Initialize the models
            models_list = []
            # Fill the models list with the models
            for i in range(horizon_max):
                models = [NHITS(h=i+1, 
                                accelerator='cuda',
                                input_size=input_size,
                                max_steps=max_steps,
                                learning_rate=learning_rate,
                                hist_exog_list=hist_exog_list,
                                futr_exog_list=additional_parameters_list[i],
                                devices=[0],
                                logger=False,
                                scaler_type='standard',
                                dropout_prob_theta=dropout_prob_theta,
                                stack_types=stack_types,
                                n_freq_downsample=n_freq_downsample,
                                mlp_units=mlp_units,
                                n_pool_kernel_size=n_pool_kernel_size,
                                n_blocks=n_blocks,
                                optimizer=torch.optim.Adam,
                                optimizer_kwargs={'weight_decay': weight_decay},
                                activation=activation)]
                model = NeuralForecast(models=models, freq='D')
                models_list.append(model)
            
        else:
            raise ValueError(f"Model {model_name} not supported for hyperparameter tuning")
        
        # List for r2 results for different prediction horizons
        r2_scores = [[] for _ in range(horizon_max)]
        
        for aquifer in aquifers_list:
            # List for storing the predictions
            predictions = [[] for _ in range(horizon_max)]
    
            X_data = time_moe_outputs[aquifer]
            X_additional_data = aquifer_by_stations[aquifer]
            y = X_data.merge(X_additional_data, on='date', how='left')
            y = y[:-test_len]
    
            if model_name in ['n_beats_x', 'n_hits_multivariate']:
                # Rename the columns (library wants to have specific names)
                y = y.rename(columns={DATE:'ds', target_feature:'y', STATION_ID:'unique_id'})

                # Fit the models
                for i in range(horizon_max):
                    # Only keep the relevant columns
                    y_fit = y[['ds', 'y', 'unique_id']+additional_parameters_list[i]+hist_exog_list+time_moe_features]
                    models_list[i].fit(y_fit[:-val_len])

            else:
                raise ValueError(f"Model {model_name} not supported for hyperparameter tuning")
            
            # Iterate from val_len days before the end, to the last day
            for i in range(val_len + (horizon_max-1), 0, -1):
                
                # Predict
                if model_name == 'n_beats_x':
                    for j in range(horizon_max):
                        y_fit = y[['ds', 'y', 'unique_id']+additional_parameters_list[j]+hist_exog_list+time_moe_features]
                        if (i-(j+1)) >= 0 and (i-(j+1)) < val_len:
                            futr_df_index = -(i-(j+1)) if i-(j+1) > 0 else None
                            # Predict
                            forecast = models_list[j].predict(df=y_fit[:-i], futr_df=y[additional_parameters_list[j] + ['ds', 'unique_id']][-i:futr_df_index], verbose=0)
                            # Store the results for every prediction horizon separately
                            predictions[j].append(forecast['NBEATSx'].values[j])

                elif model_name == 'n_hits_multivariate':
                    for j in range(horizon_max):
                        y_fit = y[['ds', 'y', 'unique_id']+additional_parameters_list[j]+hist_exog_list+time_moe_features]
                        if (i-(j+1)) >= 0 and (i-(j+1)) < val_len:
                            futr_df_index = -(i-(j+1)) if i-(j+1) > 0 else None
                            # Predict
                            forecast = models_list[j].predict(df=y_fit[:-i], futr_df=y[additional_parameters_list[j] + ['ds', 'unique_id']][-i:futr_df_index], verbose=0)
                            # Store the results for every prediction horizon separately
                            predictions[j].append(forecast['NHITS'].values[j])
               
                else:
                    raise ValueError(f"Model {model_name} not supported for hyperparameter tuning")           
    
            # Calculate the r2 scores and store them in a list
            for i in range(horizon_max):
                if model_name in ['n_beats_x', 'n_hits_multivariate']:
                    r2_scores[i].append(r2_score(y['y'][-val_len:], predictions[i]))
                else:
                    raise ValueError(f"Model {model_name} not supported for hyperparameter tuning")
        
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

    # Clean up parameter names by removing trailing numbers
    cleaned_params = {}
    for key, value in study.best_params.items():
        parts = key.split('_')
        if parts[-1].isdigit():
            # Remove the number and underscore at the end
            new_key = '_'.join(parts[:-1])
            cleaned_params[new_key] = value
        else:
            cleaned_params[key] = value

    # Return the best parameters
    return cleaned_params


# FINAL TRAINING
# ----------------------------------------------------------------------------------------------------------------------
def final_training(model_name,
                   aquifers_list,
                   test_len,
                   horizon_max, 
                   target_feature,
                   aquifer_by_stations,
                   best_params,
                   time_moe_outputs,
                   time_moe_features,
                   additional_parameters_list=additional_parameters_list,
                   hist_exog_list=hist_exog_list): 
    if model_name == 'n_beats_x':
        # Initialize the models
        models_list = []
        
        # Fill the models list with the models
        for i in range(horizon_max):
            stack_types = ['identity', 'exogenous'] if (i+1) == 1 else ['seasonality', 'trend', 'identity', 'exogenous']
            n_blocks = [1 for _ in range(len(stack_types))]
            models = [NBEATSx(h=i+1, 
                            accelerator='cuda',
                            input_size=best_params['input_size'],
                            stack_types=stack_types,
                            n_harmonics=best_params['n_harmonics'],
                            n_polynomials=best_params['n_polynomials'],
                            learning_rate=best_params['learning_rate'],
                            max_steps=best_params['max_steps'],
                            hist_exog_list=hist_exog_list,
                            futr_exog_list=additional_parameters_list[i],
                            devices=[0],
                            logger=False,
                            scaler_type='standard',
                            dropout_prob_theta=best_params['dropout_prob_theta'],
                            n_blocks=n_blocks,
                            optimizer=torch.optim.Adam,
                            optimizer_kwargs={'weight_decay': best_params['weight_decay']})]
            model = NeuralForecast(models=models, freq='D')
            models_list.append(model)
    
    elif model_name == 'n_hits_multivariate':        
        # Initialize the models
        models_list = []

        # Fill the models list with the models
        for i in range(horizon_max):
            models = [NHITS(h=i+1, 
                            accelerator='cuda',
                            input_size=best_params['input_size'],
                            max_steps=best_params['max_steps'],
                            learning_rate=best_params['learning_rate'],
                            hist_exog_list=hist_exog_list,
                            futr_exog_list=additional_parameters_list[i],
                            devices=[0],
                            logger=False,
                            scaler_type='standard',
                            dropout_prob_theta=best_params['dropout_prob_theta'],
                            stack_types=best_params['stack_types'],
                            n_freq_downsample=best_params['n_freq_downsample'],
                            mlp_units=best_params['mlp_units'],
                            n_pool_kernel_size=best_params['n_pool_kernel_size'],
                            n_blocks=best_params['n_blocks'],
                            optimizer=torch.optim.Adam,
                            optimizer_kwargs={'weight_decay': best_params['weight_decay']},
                            activation=best_params['activation'])]
            model = NeuralForecast(models=models, freq='D')
            models_list.append(model)

    else:
        raise ValueError(f"Model {model_name} not supported for final training")
    
    # List for r2 results for different prediction horizons
    r2_scores = [[] for _ in range(horizon_max)]
    
    # Dictionary for storing the predictions
    predictions_by_stations = {key: [] for key in aquifers_list}
    
    for aquifer in aquifers_list:
        # List for storing the predictions
        predictions = [[] for _ in range(horizon_max)]
    
        X_data = time_moe_outputs[aquifer]
        X_additional_data = aquifer_by_stations[aquifer]
        y = X_data.merge(X_additional_data, on='date', how='left')
            

        if model_name in ['n_beats_x', 'n_hits_multivariate']:
            # Rename the columns (library wants to have specific names)
            y = y.rename(columns={DATE:'ds', target_feature:'y', STATION_ID:'unique_id'})
            
            # Fit the models
            for i in range(horizon_max):
                # Only keep the relevant columns
                y_fit = y[['ds', 'y', 'unique_id']+additional_parameters_list[i]+hist_exog_list+time_moe_features]
                models_list[i].fit(y_fit[:-test_len])

        else:
            raise ValueError(f"Model {model_name} not supported for final training")
        
        # Iterate from day_len days before the end, to the last day
        for i in range(test_len + (horizon_max-1), 0, -1):
            # Predict
            if model_name == 'n_beats_x':
                for j in range(horizon_max):
                    y_fit = y[['ds', 'y', 'unique_id']+additional_parameters_list[j]+hist_exog_list+time_moe_features]
                    if (i-(j+1)) >= 0 and (i-(j+1)) < test_len:
                        futr_df_index = -(i-(j+1)) if i-(j+1) > 0 else None
                        # Predict
                        forecast = models_list[j].predict(df=y_fit[:-i], futr_df=y[additional_parameters_list[j] + ['ds', 'unique_id']][-i:futr_df_index], verbose=0)
                        # Store the results for every prediction horizon separately
                        predictions[j].append(forecast['NBEATSx'].values[j])

            elif model_name == 'n_hits_multivariate':
                for j in range(horizon_max):
                    y_fit = y[['ds', 'y', 'unique_id']+additional_parameters_list[j]+hist_exog_list+time_moe_features]
                    if (i-(j+1)) >= 0 and (i-(j+1)) < test_len:
                        futr_df_index = -(i-(j+1)) if i-(j+1) > 0 else None
                        # Predict
                        forecast = models_list[j].predict(df=y_fit[:-i], futr_df=y[additional_parameters_list[j] + ['ds', 'unique_id']][-i:futr_df_index], verbose=0)
                        # Store the results for every prediction horizon separately
                        predictions[j].append(forecast['NHITS'].values[j])
            
            else:
                raise ValueError(f"Model {model_name} not supported for final training")

    
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


# HELPER FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------

# Convert the time moe outputs to a dictionary of dataframes
def time_moe_outputs_to_dict(time_moe_outputs):
    for aquifer in time_moe_outputs.keys():
        time_moe_outputs[aquifer] = pd.DataFrame(time_moe_outputs[aquifer])

    return time_moe_outputs

# Add dates to the time moe outpus
def time_moe_outputs_add_dates(time_moe_outputs):
    for aquifer in time_moe_outputs.keys():
        # Generate all dates between 2010 and 2017
        all_dates = pd.date_range(start='2010-01-01', end='2017-12-31', freq='D')
        
        # Take only the last dates based on the length of time_moe_outputs[aquifer]
        aquifer_length = len(time_moe_outputs[aquifer])
        time_moe_outputs[aquifer]['date'] = all_dates[-aquifer_length:]

    return time_moe_outputs

# Function that converts time moe outputs to dictionaries and adds dates
def time_moe_outputs_preprocess(time_moe_outputs):
    time_moe_outputs = time_moe_outputs_to_dict(time_moe_outputs)
    time_moe_outputs = time_moe_outputs_add_dates(time_moe_outputs)

    return time_moe_outputs

# Get the feature names of the time moe outputs
def get_time_moe_feature_names(time_moe_outputs):
    aquifer = list(time_moe_outputs.keys())[0]
    return time_moe_outputs[aquifer].columns.tolist()

# Shift the data
# Instead of having predictions on the day they were created,
# shift them so that they are on the day they were forecasted for
def shift_data(aquifers_list, aquifer_by_stations, horizon_max):
    for i in range(horizon_max):
        for aquifer in aquifers_list:
            aquifer_by_stations[aquifer][additional_parameters_list[i]] = aquifer_by_stations[aquifer][additional_parameters_list[i]].shift(i+1)
    
    for aquifer in aquifers_list:
        aquifer_by_stations[aquifer] = aquifer_by_stations[aquifer].iloc[horizon_max:]

    return aquifer_by_stations
