import timesfm
import pandas as pd
import joblib

import numpy as np

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


# Get the data
aquifer_by_stations = joblib.load('../../data/interim/ground-water-and-weather-with-forecasts-and-additional-features.joblib')
aquifers_list = [85065]
horizon = 5 # prediction horizon
day_len = 365 # number of days to forecast
context_len = 2048

# Define the model parameters
# For Torch
model = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
          backend="gpu",
          per_core_batch_size=32,
          horizon_len=horizon,
          num_layers=50,
          use_positional_embedding=False,
          context_len=context_len,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          #huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
          huggingface_repo_id="google/timesfm-2.0-500m-jax"),
  )

print("Loaded the model successfully")
print("Yeah, motherfucker")

# List for r2 results for different prediction horizons
r2_scores = [[] for _ in range(horizon)]

for aquifer in aquifers_list:
    print(f"Prišli smo do aquifera {aquifer}")
    print("Gremo, pičke!")
    # List for storing the predictions
    predictions = [[] for _ in range(5)]

    #scaler = StandardScaler()
    #aquifer_by_stations[aquifer]['altitude_diff'] = scaler.fit_transform(aquifer_by_stations[aquifer]['altitude_diff'].values.reshape(-1, 1))

    # Iterate from day_len days before the end, to the last day
    for i in range(day_len + (horizon-1), 0, -1):
        y = aquifer_by_stations[aquifer]
        
        raw_forecast, _ = model.forecast(
          inputs=[y['altitude_diff'][:-i].values.tolist()], freq=[0] * len(y['altitude_diff'][:-i].values.tolist())
        )

        covariates = {
            'rr_decodeText': []
        }
        for j in range(horizon+1):
            if j != 0:
                covariates['rr_decodeText'][0].append(aquifer_by_stations[aquifer][f'rr_decodeText_{j}'].iloc[-i])
            else:
                covariates['rr_decodeText'].append(aquifer_by_stations[aquifer][f'rr_decodeText_{j}'].iloc[-(context_len+i):-i].values.tolist())

        cov_forecast, ols_forecast = model.forecast_with_covariates(  
              inputs=[y['altitude_diff'][-(context_len+i):-i].values.tolist()], # Wrap in list since inputs expects list of time series
              dynamic_numerical_covariates= covariates,
              freq=[0] * len(y['altitude_diff'][-(context_len+i):-i].values.tolist()),
              xreg_mode="timesfm + xreg",
              ridge=1,
              force_on_cpu=False,
              normalize_xreg_target_per_input=False    # default
        )

        # Store the results for every prediction horizon separately
        for h in range(horizon):
            #predictions[h].append(scaler.inverse_transform(cov_forecast[0][h].reshape(-1, 1))[0][0])
            predictions[h].append(cov_forecast[0][h])
            #predictions[h].append(raw_forecast[0][h])
    
    for i in range(horizon):
        if i == 0:
            predictions[i] = predictions[i][-day_len:]
        else:
            predictions[i] = predictions[i][(horizon-i-1):-i]

    # Calculate the r2 scores and store them in a list
    for i in range(horizon):
        r2_scores[i].append(r2_score(aquifer_by_stations[aquifer]['altitude_diff'][-day_len:], predictions[i]))

# Calculate the average r2 score
r2_average =  []
std_dev = []

for i in range(horizon):
    r2_average.append(np.mean(r2_scores[i]))
    std_dev.append(np.std(r2_scores[i]))
print("R2 scores:")
print(r2_average)
plt.figure(figsize=(8, 4))
plt.plot(aquifer_by_stations[aquifer]['date'][-day_len:], aquifer_by_stations[aquifer]['altitude_diff'][-day_len:], color="royalblue", label="true data")
plt.plot(aquifer_by_stations[aquifer]['date'][-day_len:], predictions[1], color="tomato", label="forecast0")
#plt.plot(aquifer_by_stations[aquifer]['date'][-day_len:], predictions[1], color="orange", label="forecast1")
#plt.plot(aquifer_by_stations[aquifer]['date'][-day_len:], predictions[2], color="green", label="forecast2")
#plt.plot(aquifer_by_stations[aquifer]['date'][-day_len:], predictions[3], color="purple", label="forecast3")
#plt.plot(aquifer_by_stations[aquifer]['date'][-day_len:], predictions[4], color="brown", label="forecast4")
plt.legend()
plt.grid()
plt.show()