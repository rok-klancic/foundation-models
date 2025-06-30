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

        elif model_name == 'n_beats_x':
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
                                #early_stop_patience_steps=30,
                                #val_check_steps=2,
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

            elif model_name in ['n_beats_x', 'n_hits_multivariate']:
                # Rename the columns (library wants to have specific names)
                y = y.rename(columns={'date':'ds', 'altitude_diff':'y', 'station_id':'unique_id'})

                # Fit the models
                for i in range(horizon_max):
                    # Only keep the relevant columns
                    y_fit = y[['ds', 'y', 'unique_id']+additional_parameters_list[i]+hist_exog_list]
                    models_list[i].fit(y_fit[:-val_len], val_size=0)

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

                elif model_name == 'n_beats_x':
                    for j in range(horizon_max):
                        y_fit = y[['ds', 'y', 'unique_id']+additional_parameters_list[j]+hist_exog_list]
                        if (i-(j+1)) >= 0 and (i-(j+1)) < val_len:
                            futr_df_index = -(i-(j+1)) if i-(j+1) > 0 else None
                            # Predict
                            forecast = models_list[j].predict(df=y_fit[:-i], futr_df=y[additional_parameters_list[j] + ['ds', 'unique_id']][-i:futr_df_index], verbose=0)
                            # Store the results for every prediction horizon separately
                            predictions[j].append(forecast['NBEATSx'].values[j])

                elif model_name == 'n_hits_multivariate':
                    for j in range(horizon_max):
                        y_fit = y[['ds', 'y', 'unique_id']+additional_parameters_list[j]+hist_exog_list]
                        if (i-(j+1)) >= 0 and (i-(j+1)) < val_len:
                            futr_df_index = -(i-(j+1)) if i-(j+1) > 0 else None
                            # Predict
                            forecast = models_list[j].predict(df=y_fit[:-i], futr_df=y[additional_parameters_list[j] + ['ds', 'unique_id']][-i:futr_df_index], verbose=0)
                            # Store the results for every prediction horizon separately
                            predictions[j].append(forecast['NHITS'].values[j])
               
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

                elif model_name in ['n_beats_x', 'n_hits_multivariate']:
                    pass
                else:
                    raise ValueError(f"Model {model_name} not supported for hyperparameter tuning")
            if model_name not in ['n_hits_multivariate', 'n_beats_x']:
                # Clean up the results
                for i in range(horizon_max):
                    if i == 0:
                        predictions[i] = predictions[i][-val_len:]
                    else:
                        predictions[i] = predictions[i][(horizon_max-i-1):-i]            
    
            # Calculate the r2 scores and store them in a list
            for i in range(horizon_max):
                if model_name in ['n_beats', 'patchtst', 'n_beats_x', 'n_hits_multivariate']:
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
    
    elif model_name == 'n_beats_x':
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

        elif model_name in ['n_beats_x', 'n_hits_multivariate']:
            # Rename the columns (library wants to have specific names)
            y = y.rename(columns={'date':'ds', 'altitude_diff':'y', 'station_id':'unique_id'})
            
            # Fit the models
            for i in range(horizon_max):
                # Only keep the relevant columns
                y_fit = y[['ds', 'y', 'unique_id']+additional_parameters_list[i]+hist_exog_list]
                models_list[i].fit(y_fit[:-test_len], val_size=0)

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

            elif model_name == 'n_beats_x':
                for j in range(horizon_max):
                    y_fit = y[['ds', 'y', 'unique_id']+additional_parameters_list[j]+hist_exog_list]
                    if (i-(j+1)) >= 0 and (i-(j+1)) < test_len:
                        futr_df_index = -(i-(j+1)) if i-(j+1) > 0 else None
                        # Predict
                        forecast = models_list[j].predict(df=y_fit[:-i], futr_df=y[additional_parameters_list[j] + ['ds', 'unique_id']][-i:futr_df_index], verbose=0)
                        # Store the results for every prediction horizon separately
                        predictions[j].append(forecast['NBEATSx'].values[j])

            elif model_name == 'n_hits_multivariate':
                for j in range(horizon_max):
                    y_fit = y[['ds', 'y', 'unique_id']+additional_parameters_list[j]+hist_exog_list]
                    if (i-(j+1)) >= 0 and (i-(j+1)) < test_len:
                        futr_df_index = -(i-(j+1)) if i-(j+1) > 0 else None
                        # Predict
                        forecast = models_list[j].predict(df=y_fit[:-i], futr_df=y[additional_parameters_list[j] + ['ds', 'unique_id']][-i:futr_df_index], verbose=0)
                        # Store the results for every prediction horizon separately
                        predictions[j].append(forecast['NHITS'].values[j])
            
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
                elif model_name in ['n_beats_x', 'n_hits_multivariate']:
                    pass
                else:
                    raise ValueError(f"Model {model_name} not supported for final training")
        
        # Clean up the results
        if model_name not in ['n_hits_multivariate', 'n_beats_x']:
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
aquifer_by_stations = joblib.load('../../data/interim/ground-water-and-weather-with-forecasts-and-additional-features.joblib')

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
            aquifer_by_stations[aquifer] = aquifer_by_stations[aquifer][['altitude_diff', 'date', 'station_id']]

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