import joblib
import pandas as pd
import os


# This script combines the aquifer data with the weather forecast data.

# CONSTANTS
# all of the paths are relative to the root of the project,
# except for the path that point to the root of the project
AQUIFER_BY_STATIONS_PATH = 'data/interim/ground-water-and-weather-no-new-features.joblib'
ROOT_PATH = 'C:/Users/Rok/Desktop/Rok/IJS/foundation-models'
WEATHER_FORECAST_PATH = 'data/interim/weather-forecast-slovenia-5-days.joblib'
SAVING_PATH = 'data/processed/ground-water-and-weather-with-forecasts.joblib'


# get aquifer data
try:
    aquifer_by_stations = joblib.load(os.path.join(ROOT_PATH, AQUIFER_BY_STATIONS_PATH))
except FileNotFoundError:
    print(f"File not found: {os.path.join(ROOT_PATH, AQUIFER_BY_STATIONS_PATH)}")
    exit(1)

# get weather forecast data
try:
    weather_forecast = joblib.load(os.path.join(ROOT_PATH, WEATHER_FORECAST_PATH))
except FileNotFoundError:
    print(f"File not found: {os.path.join(ROOT_PATH, WEATHER_FORECAST_PATH)}")
    exit(1)

# set the index as a date column
weather_forecast['date'] = weather_forecast.index


# combine the data
for key, station in aquifer_by_stations.items():
    aquifer_by_stations[key] = pd.merge(station, weather_forecast, on='date', how='left')

# save the data
joblib.dump(aquifer_by_stations, os.path.join(ROOT_PATH, SAVING_PATH))