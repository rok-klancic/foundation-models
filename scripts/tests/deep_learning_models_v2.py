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
torch.set_float32_matmul_precision('medium')
# Import time
import time

# CONSTANTS
# ----------------------------------------------------------------------------------------------------------------------
PATH_TO_DATA = 'data/interim/ground-water-and-weather-with-forecasts-and-additional-features.joblib'
PATH_TO_ROOT = '../../'

# HELPER FUNCTIONS
def add_zero_rows(y, horizon_max):
    # prepare a dataframe with the same columns as y with length of horizon_max-1, 
    # and fill the y column with last value, ds with the continous days and unique_id with the same unique_id as y,
    # and other columns with the last value
    zero_rows = pd.DataFrame({})
    for column in y.columns:
        if column == 'ds':
            # Get the last date and generate future dates
            last_date = y[column].iloc[-1]
            zero_rows[column] = pd.date_range(
                start=last_date + pd.Timedelta(days=1), 
                periods=horizon_max-1, 
                freq='D'
            ).tolist()
        else:
            zero_rows[column] = [y[column].iloc[-1] for _ in range(horizon_max-1)]
    return pd.concat([y, zero_rows], ignore_index=True)

# HYPERPARAMETER TUNING
# ----------------------------------------------------------------------------------------------------------------------
def hyperparameter_tuning(model_name,
                          horizon_max,
                          aquifers_list,
                          target_feature,
                          val_len,
                          test_len,
                          validation_size,
                          aquifer_by_stations,
                          additional_parameters_list,
                          hist_exog_list):
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
            max_steps = trial.suggest_categorical('max_steps', [1000])
            early_stop_patience_steps = trial.suggest_categorical('early_stop_patience_steps', [10])
            val_check_steps = trial.suggest_categorical('val_check_steps', [50])
            dropout_prob_theta = trial.suggest_float('dropout_prob_theta', 0.0, 0.3)
            weight_decay = trial.suggest_categorical('weight_decay', [0.0, 1e-5, 1e-4, 1e-3, 1e-2])
            activation = trial.suggest_categorical('activation', ['ReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'PReLU', 'Sigmoid'])

            models = [NHITS(h=horizon_max, 
                            accelerator='cuda',
                            input_size=input_size,
                            max_steps=max_steps,
                            early_stop_patience_steps=early_stop_patience_steps,
                            val_check_steps=val_check_steps,
                            learning_rate=learning_rate,
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
        

        elif model_name == 'patchtst':
            input_size = trial.suggest_int('input_size', 5, 730)
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

        #elif model_name == 'n_beats_x':
        #    input_size = trial.suggest_categorical('input_size', [30, 60, 180, 365, 730])
        #    n_harmonics = trial.suggest_int('n_harmonics', 1, 5)
        #    n_polynomials = trial.suggest_int('n_polynomials', 1, 5)
        #    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
        #    max_steps = trial.suggest_categorical('max_steps', [200, 500, 1000])
        #    dropout_prob_theta = trial.suggest_float('dropout_prob_theta', 0.0, 0.3)
        #    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
#
        #    # Initialize the models
        #    models_list = []
        #    
        #    # Fill the models list with the models
        #    for i in range(horizon_max):
        #        stack_types = ['identity', 'exogenous'] if (i+1) == 1 else ['seasonality', 'trend', 'identity', 'exogenous']
        #        n_blocks = [1 for _ in range(len(stack_types))]
        #        models = [NBEATSx(h=i+1, 
        #                        accelerator='cuda',
        #                        input_size=input_size,
        #                        stack_types=stack_types,
        #                        n_harmonics=n_harmonics,
        #                        n_polynomials=n_polynomials,
        #                        learning_rate=learning_rate,
        #                        max_steps=max_steps,
        #                        hist_exog_list=hist_exog_list,
        #                        futr_exog_list=additional_parameters_list[i],
        #                        devices=[0],
        #                        logger=False,
        #                        scaler_type='standard',
        #                        dropout_prob_theta=dropout_prob_theta,
        #                        n_blocks=n_blocks,
        #                        #early_stop_patience_steps=30,
        #                        #val_check_steps=2,
        #                        optimizer=torch.optim.Adam,
        #                        optimizer_kwargs={'weight_decay': weight_decay})]
        #        model = NeuralForecast(models=models, freq='D')
        #        models_list.append(model)
#
        #elif model_name == 'n_hits_multivariate':
        #    input_size = trial.suggest_categorical('input_size', [30, 60, 180, 365, 730])
        #    stack_types = trial.suggest_categorical('stack_types', [['identity', 'identity'],
        #                                                            ['identity', 'identity', 'identity'], 
        #                                                            ['identity', 'identity', 'identity', 'identity']])
        #    
        #    # n_blocks
        #    if len(stack_types) == 2:
        #        n_blocks = trial.suggest_categorical('n_blocks_2', [[1, 1], [2, 2]])
        #    elif len(stack_types) == 3:
        #        n_blocks = trial.suggest_categorical('n_blocks_3', [[1, 1, 1], [2, 2, 2]])
        #    elif len(stack_types) == 4:
        #        n_blocks = trial.suggest_categorical('n_blocks_4', [[1, 1, 1, 1], [2, 2, 2, 2]])
        #    else:
        #        raise ValueError(f"Stack types {stack_types} not supported")
        #    
        #    # n_freq_downsample
        #    if len(stack_types) == 2:
        #        n_freq_downsample = trial.suggest_categorical('n_freq_downsample_2', [[2, 1], [3, 1], [4, 1]])
        #    elif len(stack_types) == 3:
        #        n_freq_downsample = trial.suggest_categorical('n_freq_downsample_3', [[4, 2, 1], [3, 2, 1], [2, 1, 1]])
        #    elif len(stack_types) == 4:
        #        n_freq_downsample = trial.suggest_categorical('n_freq_downsample_4', [[4, 3, 2, 1], [3, 2, 1, 1], [2, 1, 1, 1]])
        #    else:
        #        raise ValueError(f"Stack types {stack_types} not supported")
#
        #    # mlp_units
        #    if len(stack_types) == 2:
        #        mlp_units = trial.suggest_categorical('mlp_units_2', [[[256, 256], [256, 256]], [[128, 128], [128, 128]], [[512, 512], [512, 512]], [[1024, 1024], [1024, 1024]]])
        #    elif len(stack_types) == 3:
        #        mlp_units = trial.suggest_categorical('mlp_units_3', [[[256, 256], [256, 256], [256, 256]], [[128, 128], [128, 128], [128, 128]], [[512, 512], [512, 512], [512, 512]], [[1024, 1024], [1024, 1024], [1024, 1024]]])
        #    elif len(stack_types) == 4:
        #        mlp_units = trial.suggest_categorical('mlp_units_4', [[[256, 256], [256, 256], [256, 256], [256, 256]], [[128, 128], [128, 128], [128, 128], [128, 128]], [[512, 512], [512, 512], [512, 512], [512, 512]], [[1024, 1024], [1024, 1024], [1024, 1024], [1024, 1024]]])
        #    else:
        #        raise ValueError(f"Stack types {stack_types} not supported")
#
        #    # n_pool_kernel_size
        #    if len(stack_types) == 2:
        #        n_pool_kernel_size = trial.suggest_categorical('n_pool_kernel_size_2', [[2, 1], [3, 1], [4, 1]])
        #    elif len(stack_types) == 3:
        #        n_pool_kernel_size = trial.suggest_categorical('n_pool_kernel_size_3', [[4, 2, 1], [3, 2, 1], [2, 1, 1]])
        #    elif len(stack_types) == 4:
        #        n_pool_kernel_size = trial.suggest_categorical('n_pool_kernel_size_4', [[4, 3, 2, 1], [3, 2, 1, 1], [2, 1, 1, 1]])
        #    else:
        #        raise ValueError(f"Stack types {stack_types} not supported")
        #    
        #    dropout_prob_theta = trial.suggest_float('dropout_prob_theta', 0.0, 0.3)
        #    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
        #    max_steps = trial.suggest_categorical('max_steps', [100, 200, 500, 1000])
        #    dropout_prob_theta = trial.suggest_float('dropout_prob_theta', 0.0, 0.3)
        #    weight_decay = trial.suggest_categorical('weight_decay', [0.0, 1e-5, 1e-4, 1e-3, 1e-2])
        #    activation = trial.suggest_categorical('activation', ['ReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'PReLU', 'Sigmoid'])
        #    
        #    # Initialize the models
        #    models_list = []
        #    # Fill the models list with the models
        #    for i in range(horizon_max):
        #        models = [NHITS(h=i+1, 
        #                        accelerator='cuda',
        #                        input_size=input_size,
        #                        max_steps=max_steps,
        #                        learning_rate=learning_rate,
        #                        hist_exog_list=hist_exog_list,
        #                        futr_exog_list=additional_parameters_list[i],
        #                        devices=[0],
        #                        logger=False,
        #                        scaler_type='standard',
        #                        dropout_prob_theta=dropout_prob_theta,
        #                        stack_types=stack_types,
        #                        n_freq_downsample=n_freq_downsample,
        #                        mlp_units=mlp_units,
        #                        n_pool_kernel_size=n_pool_kernel_size,
        #                        n_blocks=n_blocks,
        #                        optimizer=torch.optim.Adam,
        #                        optimizer_kwargs={'weight_decay': weight_decay},
        #                        activation=activation)]
        #        model = NeuralForecast(models=models, freq='D')
        #        models_list.append(model)
            
        else:
            raise ValueError(f"Model {model_name} not supported for hyperparameter tuning")
        
        # List for r2 results for different prediction horizons
        r2_scores = [[] for _ in range(horizon_max)]
        
        for aquifer in aquifers_list:
            # List for storing the predictions
            predictions = [[] for _ in range(horizon_max)]
    
            # Get the dataset for the aquifer
            y = aquifer_by_stations[aquifer][:-test_len]

            true_diffs = y['altitude_diff'][-val_len:].values
    
            if model_name in ['n_beats', 'n_hits', 'patchtst']:
                # check if the altitude column exists
                if target_feature in y.columns:
                    print(f'{target_feature} column exists')
                else:
                    raise ValueError(f"Target feature {target_feature} not found in the dataset")

                # Rename the columns (library wants to have specific names)
                y = y.rename(columns={'date':'ds', target_feature:'y', 'station_id':'unique_id'})
        
                # Only keep these 3 columns
                y = y[['ds', 'y', 'unique_id']]

                # Add 0 rows to the end of the dataframe, so that the model can predict all of the test dates with all of the horizons
                y = add_zero_rows(y, horizon_max)

        
                # Fit and predict
                forecasts = model.cross_validation(
                    df=y,
                    val_size = validation_size,
                    test_size = val_len+(2*(horizon_max-1)),
                    step_size = 1,
                    n_windows = None
                )
                
                # Delete the added rows
                y = y.iloc[:-(horizon_max-1)]

                # Extract the forecasts for every prediction horizon

                # Sort the values by cutoff date
                forecasts = forecasts.sort_values(by=['cutoff', 'ds'])

                # Define the column name
                match model_name:
                    case 'n_beats':
                        column_name = 'NBEATS'
                    case 'n_hits':
                        column_name = 'NHITS'
                    case 'patchtst':
                        column_name = 'PatchTST'
                    case _:
                        raise ValueError(f"Unsupported model name: {model_name}")

                if target_feature == 'altitude_diff':
                    # Group the predicions by horizon
                    for horizon in range(horizon_max):
                        # Get the predictions that are horizon steps ahead from each cutoff date
                        horizon_predictions = forecasts.groupby('cutoff').nth(horizon)[column_name].values
                        predictions[horizon] = horizon_predictions   

                    # Clean up the results
                    for i in range(horizon_max):
                        if i == 0:
                            predictions[i] = predictions[i][-val_len:]
                        else:
                            predictions[i] = predictions[i][(horizon_max-i-1):-i]   

                    # Calculate the r2 scores and store them in a list
                    for i in range(horizon_max):
                        r2_scores[i].append(r2_score(y['y'][-val_len:], predictions[i]))

                elif target_feature == 'altitude':
                    # Get predictions organized by cutoff and horizon
                    altitude_predictions = []
                    for horizon in range(horizon_max):
                        horizon_preds = forecasts.groupby('cutoff').nth(horizon)[column_name].values
                        altitude_predictions.append(horizon_preds)
                    
                    # Convert to numpy array: shape (horizon_max, num_cutoffs)
                    altitude_matrix = np.array(altitude_predictions)
                    
                    # Get true values at cutoff dates
                    cutoff_dates = forecasts['cutoff'].unique()
                    true_values = y[y['ds'].isin(cutoff_dates)]['y'].values
                    
                    # Vectorized diff calculation
                    # Create sequences: [true_value, pred_h0, pred_h1, pred_h2, ...]
                    sequences = np.vstack([true_values, altitude_matrix]).T  # Shape: (num_cutoffs, horizon_max+1)
                    altitude_diffs = np.diff(sequences, axis=1)  # Shape: (num_cutoffs, horizon_max)
                    
                    # Convert to list format by horizon
                    predictions = [altitude_diffs[:, i].tolist() for i in range(horizon_max)]
                    
                    # Clean up the results
                    for i in range(horizon_max):
                        if i == 0:
                            predictions[i] = predictions[i][-val_len:]
                        else:
                            predictions[i] = predictions[i][(horizon_max-i-1):-i]  

                    # Calculate the r2 scores and store them in a list
                    for i in range(horizon_max):
                        r2_scores[i].append(r2_score(true_diffs, predictions[i]))

                else:
                    raise ValueError(f"Target feature {target_feature} not supported")




            #elif model_name in ['n_beats_x', 'n_hits_multivariate']:
            #    # Rename the columns (library wants to have specific names)
            #    y = y.rename(columns={'date':'ds', 'altitude_diff':'y', 'station_id':'unique_id'})
#
            #    # Fit the models
            #    for i in range(horizon_max):
            #        # Only keep the relevant columns
            #        y_fit = y[['ds', 'y', 'unique_id']+additional_parameters_list[i]+hist_exog_list]
            #        models_list[i].fit(y_fit[:-val_len], val_size=0)

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
    study.optimize(objective, n_trials=50)

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
                   validation_size,
                   horizon_max, 
                   target_feature,
                   aquifer_by_stations,
                   best_params,
                   additional_parameters_list,
                   hist_exog_list):
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
        models = [NHITS(h=horizon_max, 
                        accelerator='cuda',
                        input_size=best_params['input_size'],
                        max_steps=best_params['max_steps'],
                        early_stop_patience_steps=best_params['early_stop_patience_steps'],
                        val_check_steps=best_params['val_check_steps'],
                        learning_rate=best_params['learning_rate'],
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
        
        model = NeuralForecast(
            models=[model],
            freq='D'
        )
    
    #elif model_name == 'n_beats_x':
    #    # Initialize the models
    #    models_list = []
    #    
    #    # Fill the models list with the models
    #    for i in range(horizon_max):
    #        stack_types = ['identity', 'exogenous'] if (i+1) == 1 else ['seasonality', 'trend', 'identity', 'exogenous']
    #        n_blocks = [1 for _ in range(len(stack_types))]
    #        models = [NBEATSx(h=i+1, 
    #                        accelerator='cuda',
    #                        input_size=best_params['input_size'],
    #                        stack_types=stack_types,
    #                        n_harmonics=best_params['n_harmonics'],
    #                        n_polynomials=best_params['n_polynomials'],
    #                        learning_rate=best_params['learning_rate'],
    #                        max_steps=best_params['max_steps'],
    #                        hist_exog_list=hist_exog_list,
    #                        futr_exog_list=additional_parameters_list[i],
    #                        devices=[0],
    #                        logger=False,
    #                        scaler_type='standard',
    #                        dropout_prob_theta=best_params['dropout_prob_theta'],
    #                        n_blocks=n_blocks,
    #                        optimizer=torch.optim.Adam,
    #                        optimizer_kwargs={'weight_decay': best_params['weight_decay']})]
    #        model = NeuralForecast(models=models, freq='D')
    #        models_list.append(model)
    #
    #elif model_name == 'n_hits_multivariate':        
    #    # Initialize the models
    #    models_list = []
#
    #    # Fill the models list with the models
    #    for i in range(horizon_max):
    #        models = [NHITS(h=i+1, 
    #                        accelerator='cuda',
    #                        input_size=best_params['input_size'],
    #                        max_steps=best_params['max_steps'],
    #                        learning_rate=best_params['learning_rate'],
    #                        hist_exog_list=hist_exog_list,
    #                        futr_exog_list=additional_parameters_list[i],
    #                        devices=[0],
    #                        logger=False,
    #                        scaler_type='standard',
    #                        dropout_prob_theta=best_params['dropout_prob_theta'],
    #                        stack_types=best_params['stack_types'],
    #                        n_freq_downsample=best_params['n_freq_downsample'],
    #                        mlp_units=best_params['mlp_units'],
    #                        n_pool_kernel_size=best_params['n_pool_kernel_size'],
    #                        n_blocks=best_params['n_blocks'],
    #                        optimizer=torch.optim.Adam,
    #                        optimizer_kwargs={'weight_decay': best_params['weight_decay']},
    #                        activation=best_params['activation'])]
    #        model = NeuralForecast(models=models, freq='D')
    #        models_list.append(model)

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

        true_diffs = y['altitude_diff'][-test_len:].values
    
        if model_name in ['n_beats', 'n_hits', 'patchtst']:
            # Rename the columns (library wants to have specific names)
            y = y.rename(columns={'date':'ds', target_feature:'y', 'station_id':'unique_id'})
            # Only keep these 3 columns
            y = y[['ds', 'y', 'unique_id']]

            # Add 0 rows to the end of the dataframe, so that the model can predict all of the test dates with all of the horizons
            y = add_zero_rows(y, horizon_max)
    
            # Fit and predict
            forecasts = model.cross_validation(
                df=y,
                val_size = validation_size,
                test_size = test_len+(2*(horizon_max-1)),
                step_size = 1,
                n_windows = None
            )

            # Delete the added rows
            y = y.iloc[:-(horizon_max-1)]

            # Extract the forecasts for every prediction horizon

            # Sort the values by cutoff date
            forecasts = forecasts.sort_values(by=['cutoff', 'ds'])

            # Define the column name
            match model_name:
                case 'n_beats':
                    column_name = 'NBEATS'
                case 'n_hits':
                    column_name = 'NHITS'
                case 'patchtst':
                    column_name = 'PatchTST'
                case _:
                    raise ValueError(f"Unsupported model name: {model_name}")

            if target_feature == 'altitude_diff':
                # Group the predicions by horizon
                for horizon in range(horizon_max):
                    # Get the predictions that are horizon steps ahead from each cutoff date
                    horizon_predictions = forecasts.groupby('cutoff').nth(horizon)[column_name].values
                    predictions[horizon] = horizon_predictions

                # Shorten the predictions to the test length
                # If the target feature is altitude, convert the predictions to the altitude_diff

                for i in range(horizon_max):
                    if i == 0:
                        predictions[i] = predictions[i][-test_len:]
                    else:
                        predictions[i] = predictions[i][(horizon_max-i-1):-i]   

                # Calculate the r2 scores and store them in a list
                for i in range(horizon_max):
                    r2_scores[i].append(r2_score(y['y'][-test_len:], predictions[i]))

                # Store the predictions to the dictionary
                predictions_by_stations[aquifer] = predictions

            elif target_feature == 'altitude':
                # Get predictions organized by cutoff and horizon
                altitude_predictions = []
                for horizon in range(horizon_max):
                    horizon_preds = forecasts.groupby('cutoff').nth(horizon)[column_name].values
                    altitude_predictions.append(horizon_preds)
                
                # Convert to numpy array: shape (horizon_max, num_cutoffs)
                altitude_matrix = np.array(altitude_predictions)
                
                # Get true values at cutoff dates
                cutoff_dates = forecasts['cutoff'].unique()
                true_values = y[y['ds'].isin(cutoff_dates)]['y'].values
                
                # Vectorized diff calculation
                # Create sequences: [true_value, pred_h0, pred_h1, pred_h2, ...]
                sequences = np.vstack([true_values, altitude_matrix]).T  # Shape: (num_cutoffs, horizon_max+1)
                altitude_diffs = np.diff(sequences, axis=1)  # Shape: (num_cutoffs, horizon_max)
                
                # Convert to list format by horizon
                predictions = [altitude_diffs[:, i].tolist() for i in range(horizon_max)]
                
                # Clean up the results
                for i in range(horizon_max):
                    if i == 0:
                        predictions[i] = predictions[i][-test_len:]
                    else:
                        predictions[i] = predictions[i][(horizon_max-i-1):-i]  

                # Calculate the r2 scores and store them in a list
                for i in range(horizon_max):
                    r2_scores[i].append(r2_score(true_diffs, predictions[i]))

                # Store the predictions to the dictionary 
                predictions_by_stations[aquifer] = predictions

            else:
                raise ValueError(f"Target feature {target_feature} not supported")


        #elif model_name in ['n_beats_x', 'n_hits_multivariate']:
        #    # Rename the columns (library wants to have specific names)
        #    y = y.rename(columns={'date':'ds', 'altitude_diff':'y', 'station_id':'unique_id'})
        #    
        #    # Fit the models
        #    for i in range(horizon_max):
        #        # Only keep the relevant columns
        #        y_fit = y[['ds', 'y', 'unique_id']+additional_parameters_list[i]+hist_exog_list]
        #        models_list[i].fit(y_fit[:-test_len], val_size=0)

        else:
            raise ValueError(f"Model {model_name} not supported for final training")
        
    # Return the average r2 scores
    r2_average =  []    
    for i in range(horizon_max):
        r2_average.append(np.mean(r2_scores[i]))

    # Return the predictions and the r2 scores
    return r2_average, r2_scores, predictions_by_stations


# SHIFTING THE DATA
# ----------------------------------------------------------------------------------------------------------------------
def shift_data(aquifers_list, aquifer_by_stations, horizon_max, additional_parameters_list):
    for i in range(horizon_max):
        for aquifer in aquifers_list:
            aquifer_by_stations[aquifer][additional_parameters_list[i]] = aquifer_by_stations[aquifer][additional_parameters_list[i]].shift(i+1)
    
    for aquifer in aquifers_list:
        aquifer_by_stations[aquifer] = aquifer_by_stations[aquifer].iloc[horizon_max:]

    return aquifer_by_stations


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

def save_results(model_name,
                 multivariate,
                 r2_scores,
                 predictions,
                 best_features,
                 best_params,
                 file_path,
                 additional_parameters_list,
                 hist_exog_list):
    # Create a dictionary to store the results
    results = {
        'model_name': model_name,
        'multivariate': multivariate,
        'r2_scores': convert_to_native(r2_scores),
        'predictions': convert_to_native(predictions),
        'best_features': best_features,
        'best_params': best_params,
        'additional_parameters_list': additional_parameters_list,
        'hist_exog_list': hist_exog_list
    }

    with open(file_path, 'w') as file:
        json.dump(results, file, indent=4)


# EXPERIMENT SETTINGS
# ----------------------------------------------------------------------------------------------------------------------
# Time the experiment
start_time = time.time()

# Load the experiment settings
with open('experiment_settings/deep_learning_models_experiment_settings.json', 'r') as file:
    experiment_settings = json.load(file)

# Load the data
aquifer_by_stations = joblib.load(os.path.join(PATH_TO_ROOT, PATH_TO_DATA))

# TESTING THE MODELS
# ----------------------------------------------------------------------------------------------------------------------
for name, settings in experiment_settings.items():
    # Get the model name
    model_name = experiment_settings[name]['model_name']


    # Variable that tells us if the model is multivariate
    multivariate = experiment_settings[name]['additional_features']

    # If the model is univariate, remove additional features
    if not multivariate:
        for aquifer in aquifer_by_stations.keys():
            aquifer_by_stations[aquifer] = aquifer_by_stations[aquifer][['altitude', 'altitude_diff', 'date', 'station_id']]

    # List of additional parameters
    if multivariate:
        additional_parameters_list = settings['additional_parameters_list']
        hist_exog_list = settings['hist_exog_list']
    else:
        if model_name in ['n_beats_x', 'n_hits_multivariate']:
            raise ValueError(f"Model {model_name} is not supported for univariate data")
        additional_parameters_list = []
        hist_exog_list = []

    # Shift the data if needed
    if multivariate:
        aquifer_by_stations = shift_data(aquifers_list=settings['aquifers_list'],
                                         aquifer_by_stations=aquifer_by_stations,
                                         horizon_max=settings['horizon_max'],
                                         additional_parameters_list=settings['additional_parameters_list'])

    best_features = {}

    # Hyperparameter tuning
    best_params = hyperparameter_tuning(model_name=model_name,
                                        horizon_max=settings['horizon_max'],
                                        aquifers_list=settings['aquifers_list'], 
                                        target_feature=settings['target_feature'],
                                        val_len=settings['val_len'],
                                        test_len=settings['test_len'], 
                                        validation_size=settings['validation_size'],
                                        aquifer_by_stations=aquifer_by_stations,
                                        additional_parameters_list=additional_parameters_list,
                                        hist_exog_list=hist_exog_list)
        
    # Final training
    r2_average, r2_scores, predictions = final_training(model_name=model_name,
                                                        aquifers_list=settings['aquifers_list'], 
                                                        test_len=settings['test_len'],
                                                        validation_size=settings['validation_size'], 
                                                        horizon_max=settings['horizon_max'],
                                                        target_feature=settings['target_feature'],
                                                        aquifer_by_stations=aquifer_by_stations,
                                                        best_params=best_params,
                                                        additional_parameters_list=additional_parameters_list,
                                                        hist_exog_list=hist_exog_list)
    
    # Time the experiment
    end_time = time.time()
    total_time = end_time - start_time
    # Convert total time to hours, minutes, seconds
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    # Obtain the index of the file name (so every experiment has a unique name)
    index = get_index(folder_path='../results/deep_learning_models', file_name=name)

    # Save the results
    file_path = f'../results/deep_learning_models/{name}_{index}.json'
    save_results(model_name=model_name,
                 multivariate=multivariate,
                 r2_scores=r2_scores,
                 predictions=predictions,
                 best_features=best_features,
                 best_params=best_params,
                 file_path=file_path,
                 additional_parameters_list=additional_parameters_list,
                 hist_exog_list=hist_exog_list)

    # Print the results
    print("--------------------------------------------------------------------------------------------------")
    print(f"Model: {name}")
    print(f"R2 average: {r2_average}")
    print(f"Total time: {hours}h {minutes}m {seconds}s")
    print("--------------------------------------------------------------------------------------------------\n\n")