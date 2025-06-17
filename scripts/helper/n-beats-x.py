# N-BEATS
from neuralforecast.models import NBEATS, PatchTST, NBEATSx
from neuralforecast.losses.pytorch import HuberLoss
from neuralforecast.core import NeuralForecast

import joblib
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, make_scorer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import RandomizedSearchCV

# Hyperparameter tuning
import optuna

# N-HiTS
from darts import TimeSeries
from darts.models import NHiTSModel
from torch.nn import MSELoss

# Warnings
import warnings
warnings.filterwarnings('ignore')

##### Standard scaling
def standard_scaling(x):
    mean = np.mean(np.abs(x))
    s = np.std(x)
    if s == 0:
        return x    
    return (x - mean)/s

def standard_unscaling(original, scaled):
    mean = np.mean(np.abs(original))
    s = np.std(original)

    return (scaled * s) + mean

# Get the data
aquifer_by_stations = joblib.load('../../data/interim/ground-water-and-weather-with-forecasts-and-additional-features.joblib')
aquifers_list = [85065, 85064]
additional_parameters = [
 'altitude',
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
 'ff_decodeText_0',
 'tn_1',
 'tx_1',
 'nn_decodeText_1',
 'rr_decodeText_1',
 'ff_decodeText_1',
 'tn_2',
 'tx_2',
 'nn_decodeText_2',
 'rr_decodeText_2',
 'ff_decodeText_2',
 'tn_3',
 'tx_3',
 'nn_decodeText_3',
 'rr_decodeText_3',
 'ff_decodeText_3',
 'tn_4',
 'tx_4',
 'nn_decodeText_4',
 'rr_decodeText_4',
 'ff_decodeText_4',
 'tn_5',
 'tx_5',
 'nn_decodeText_5',
 'rr_decodeText_5',
 'ff_decodeText_5',
 'wwsyn_decodeText_0_RA',
 'wwsyn_decodeText_0_RASN',
 'wwsyn_decodeText_0_SHRA',
 'wwsyn_decodeText_0_SHRASN',
 'wwsyn_decodeText_0_SHSN',
 'wwsyn_decodeText_0_SN',
 'wwsyn_decodeText_0_nil',
 'dd_decodeText_0_E',
 'dd_decodeText_0_N',
 'dd_decodeText_0_NE',
 'dd_decodeText_0_NW',
 'dd_decodeText_0_S',
 'dd_decodeText_0_SE',
 'dd_decodeText_0_SW',
 'dd_decodeText_0_W',
 'dd_decodeText_0_nil',
 'wwsyn_decodeText_1_RA',
 'wwsyn_decodeText_1_RASN',
 'wwsyn_decodeText_1_SHRA',
 'wwsyn_decodeText_1_SHSN',
 'wwsyn_decodeText_1_SN',
 'wwsyn_decodeText_1_nil',
 'dd_decodeText_1_E',
 'dd_decodeText_1_N',
 'dd_decodeText_1_NE',
 'dd_decodeText_1_NW',
 'dd_decodeText_1_S',
 'dd_decodeText_1_SE',
 'dd_decodeText_1_SW',
 'dd_decodeText_1_W',
 'dd_decodeText_1_nil',
 'wwsyn_decodeText_2_RA',
 'wwsyn_decodeText_2_RASN',
 'wwsyn_decodeText_2_SHRA',
 'wwsyn_decodeText_2_SHSN',
 'wwsyn_decodeText_2_SN',
 'wwsyn_decodeText_2_nil',
 'dd_decodeText_2_E',
 'dd_decodeText_2_N',
 'dd_decodeText_2_NE',
 'dd_decodeText_2_NW',
 'dd_decodeText_2_S',
 'dd_decodeText_2_SE',
 'dd_decodeText_2_SW',
 'dd_decodeText_2_W',
 'dd_decodeText_2_nil',
 'wwsyn_decodeText_3_FZRA',
 'wwsyn_decodeText_3_RA',
 'wwsyn_decodeText_3_RASN',
 'wwsyn_decodeText_3_SHRA',
 'wwsyn_decodeText_3_SHSN',
 'wwsyn_decodeText_3_SN',
 'wwsyn_decodeText_3_nil',
 'dd_decodeText_3_E',
 'dd_decodeText_3_N',
 'dd_decodeText_3_NE',
 'dd_decodeText_3_NW',
 'dd_decodeText_3_S',
 'dd_decodeText_3_SE',
 'dd_decodeText_3_SW',
 'dd_decodeText_3_W',
 'dd_decodeText_3_nil',
 'wwsyn_decodeText_4_FZRA',
 'wwsyn_decodeText_4_RA',
 'wwsyn_decodeText_4_RASN',
 'wwsyn_decodeText_4_SHRA',
 'wwsyn_decodeText_4_SHRASN',
 'wwsyn_decodeText_4_SHSN',
 'wwsyn_decodeText_4_SN',
 'wwsyn_decodeText_4_nil',
 'dd_decodeText_4_E',
 'dd_decodeText_4_N',
 'dd_decodeText_4_NE',
 'dd_decodeText_4_NW',
 'dd_decodeText_4_S',
 'dd_decodeText_4_SE',
 'dd_decodeText_4_SW',
 'dd_decodeText_4_W',
 'dd_decodeText_4_nil',
 'wwsyn_decodeText_5_FZRA',
 'wwsyn_decodeText_5_RA',
 'wwsyn_decodeText_5_RASN',
 'wwsyn_decodeText_5_SHRA',
 'wwsyn_decodeText_5_SHSN',
 'wwsyn_decodeText_5_SN',
 'wwsyn_decodeText_5_nil',
 'dd_decodeText_5_E',
 'dd_decodeText_5_ENE',
 'dd_decodeText_5_N',
 'dd_decodeText_5_NE',
 'dd_decodeText_5_NNW',
 'dd_decodeText_5_NW',
 'dd_decodeText_5_S',
 'dd_decodeText_5_SE',
 'dd_decodeText_5_SW',
 'dd_decodeText_5_W',
 'dd_decodeText_5_nil']
##### Hyperparameter tuning
# Define the horizon and the day_len
horizon = 5
day_len = 100
test_len = 365
aquifers_list = [85065, 85064]
# Scale the additional features
for aquifer in aquifers_list:
    for feature in additional_parameters:
        aquifer_by_stations[aquifer][feature] = standard_scaling(aquifer_by_stations[aquifer][feature])
# Define the function which contains parameters to tune and the model

def objective(trial):
    input_size = trial.suggest_int('input_size', 5, 20)
    
    n_harmonics = trial.suggest_int('n_harmonics', 1, 5)
    n_polynomials = trial.suggest_int('n_polynomials', 1, 5)
    
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1)

    max_steps = trial.suggest_int('max_steps', 10, 600)

    validation_size = trial.suggest_int('val_size', 5, 15)

    scaling = trial.suggest_categorical('scaling', [True, False])

    

    models = [NBEATSx(h=horizon,input_size=input_size,
                 max_steps=max_steps,
                 learning_rate=learning_rate,
                 n_harmonics=n_harmonics,
                 n_polynomials=n_polynomials,
                 hist_exog_list=additional_parameters,
                 accelerator='cuda',
                 logger=False)
                 ]
    model = NeuralForecast(models=models, freq='D')

    # List for r2 results for different prediction horizons
    r2_scores = [[] for _ in range(horizon)]
    
    for aquifer in aquifers_list:
        # List for storing the predictions
        predictions = [[] for _ in range(5)]

        # Get the dataset for the aquifer
        y = aquifer_by_stations[aquifer][:-test_len]

        # Rename the columns (library wants to have specific names)
        y = y.rename(columns={'date':'ds', 'altitude_diff':'y', 'station_id':'unique_id'})

        # Scaling
        if scaling:
            y['y'] = standard_scaling(y['y'])

        # Fit the model
        model.fit(y[:-day_len], val_size=validation_size)

        # Iterate from day_len days before the end, to the last day
        for i in range(day_len + (horizon-1), 0, -1):
            
            # Predict
            forecast = model.predict(df=y[:-i], verbose=0)

            # Unscale
            if scaling:
                forecast['NBEATSx'] = standard_unscaling(aquifer_by_stations[aquifer]['altitude_diff'], forecast['NBEATSx'])
                

            # Store the results for every prediction horizon separately
            for i in range(horizon):
                predictions[i].append(forecast['NBEATSx'].values[i])
        
        # Clean up the results
        predictions[0] = predictions[0][-day_len:]
        predictions[1] = predictions[1][3:-1]
        predictions[2] = predictions[2][2:-2]
        predictions[3] = predictions[3][1:-3]
        predictions[4] = predictions[4][0:-4]

        # Calculate the r2 scores and store them in a list
        for i in range(horizon):
            r2_scores[i].append(r2_score(y['y'][-day_len:], predictions[i]))
    
    # Calculate the average r2 score
    r2_average =  []
    
    for i in range(5):
        r2_average.append(np.mean(r2_scores[i]))

    # Set the loss as average of average r2 scores for different prediction horizons
    loss = np.mean(r2_average)

    print(r2_average)

    return loss
# Run the optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print(study.best_params)
print(study.best_value)