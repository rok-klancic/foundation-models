import pandas as pd
import joblib
import os

DICTIONARIES_PATHS = ['data\interim\ground-water-and-weather-no-new-features.joblib',
                      'data\interim\ground-water-and-weather-with-forecasts-and-additional-features.joblib',
                      'data\interim\ground-water-and-weather-with-weather-and-timemoe-forecasts-and-additional-features.joblib',
                      ]

DATAFRAMES_PATHS = ['data\interim\weather-forecast-slovenia-5-days.joblib']

PATH_TO_ROOT = '../../'

#MAIN
for path in DICTIONARIES_PATHS:
    # Get the data
    data = joblib.load(os.path.join(PATH_TO_ROOT, path))
    for key, value in data.items():
        data[key] = value.copy()

    joblib.dump(data, os.path.join(PATH_TO_ROOT, path))

for path in DATAFRAMES_PATHS:
    data =  joblib.load(os.path.join(PATH_TO_ROOT, path))
    joblib.dump(data.copy(), os.path.join(PATH_TO_ROOT, path))
