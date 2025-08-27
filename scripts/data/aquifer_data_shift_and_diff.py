import joblib
import pandas as pd
import os
import gc


# CONSTANTS
# all of the paths are relative to the root of the project,
# except for the path that point to the root of the project
AQUIFER_BY_STATIONS_PATH = 'data/interim/ground-water-and-weather-no-new-features.joblib'
ROOT_PATH = '../../'
SAVING_PATH = 'data/interim/ground-water-and-weather-shifts-and-diffs-2025-08-25.joblib'
DAYS_TO_SHIFT = 10
MAX_DAYS_AHEAD = 5
DAYS_TO_AVG = 10
COLUMNS_TO_SHIFT = ['precipitation', 'snow_accumulation', 'temperature_avg',
       'temperature_min', 'temperature_max', 'cloud_cover_avg',
       'cloud_cover_min', 'cloud_cover_max', 'humidity_avg', 'humidity_min', 'humidity_max', 
       'precipitation_probability_avg', 'precipitation_probability_min', 'precipitation_probability_max',
        'precipitation_intensity_avg', 'precipitation_intensity_min', 'precipitation_intensity_max',
        'altitude_diff_1', 'altitude_diff_2', 'altitude_diff_3', 'altitude_diff_4', 'altitude_diff_5']
FEATURE_TO_DIFF = 'altitude'

# FUNCTIONS
#def create_diffs(data, max_days_ahead, feature_name):
#    for i in range(1, max_days_ahead+1):
#        data[f'{feature_name}_diff_{i}'] = data[feature_name].diff(i)
#
#        # shift the column, so it contains NaN at the end
#        data[f'{feature_name}_diff_{i}'] = data[f'{feature_name}_diff_{i}'].shift(-i)
#
#        # fill the NaN values with the previous value
#        data[f'{feature_name}_diff_{i}'] = data[f'{feature_name}_diff_{i}'].ffill()
#
#    return data

def create_diffs(data, max_days_ahead, feature_name):
    new_columns = {}

    for i in range(1, max_days_ahead+1):
        diff = data[feature_name].diff(i)
        # shift the column, so it contains NaN at the end
        diff = diff.shift(-i)
        # fill the NaN values with the previous value
        diff = diff.ffill()

        new_columns[f'{feature_name}_diff_{i}'] = diff

    new_df = pd.DataFrame(new_columns, index=data.index)
    new_data = pd.concat([data, new_df], axis=1)

    # Clean up
    del new_columns, new_df
    return new_data

#def create_shifts(data, days_to_shift, columns_to_shift):
#    # Iterate over all of the columns in the columns_to_shift
#    for column in columns_to_shift:
#        # Iterate over all shifts
#        for shift in range (1, days_to_shift+1):
#            first_value = data[column].iloc[0]
#            data[f'{column}_shift{shift}'] = data[column].shift(shift)
#            # Fill the first values (NaN) with the first values from original columns
#            data[f'{column}_shift{shift}'] = data[f'{column}_shift{shift}'].fillna(first_value)
#
#    return data

def create_shifts(data, days_to_shift, columns_to_shift):
    new_columns = {}
    
    for column in columns_to_shift:
        for shift in range(1, days_to_shift+1):
            first_value = data[column].iloc[0]
            # shift the column and fill the nan values
            shifted = data[column].shift(shift).fillna(first_value)
            # save to the dictionary
            new_columns[f'{column}_shift{shift}'] = shifted
    
    # Add all new columns at once
    new_df = pd.DataFrame(new_columns, index=data.index)
    result = pd.concat([data, new_df], axis=1)

    # Clean up
    del new_columns, new_df
    return result

#def create_avg(data, days_to_avg, columns_to_avg):
#    for column in columns_to_avg:
#        # Iterate over all averages
#        for avg in range(2, days_to_avg+1):
#            data[f'{column}_average{avg}'] = data[column].rolling(window=avg, min_periods=1).mean()
#
#    return data

def create_avg(data, days_to_avg, columns_to_avg):
    new_columns = {}

    for column in columns_to_avg:
        for avg in range(2, days_to_avg+1):
            averaged = data[column].rolling(window=avg, min_periods=1).mean()
            # add the averaged column to the dictionary
            new_columns[f'{column}_average{avg}'] = averaged

    # Add all new columns at once
    new_df = pd.DataFrame(new_columns, index=data.index)
    result = pd.concat([data, new_df], axis=1)
    
    # Clean up
    del new_columns, new_df
    return result



# MAIN

# get aquifer data
try:
    aquifer_by_stations = joblib.load(os.path.join(ROOT_PATH, AQUIFER_BY_STATIONS_PATH))
except FileNotFoundError:
    print(f"File not found: {os.path.join(ROOT_PATH, AQUIFER_BY_STATIONS_PATH)}")
    exit(1)

## Create diffs for predicting multiple days ahead
#for key, data in aquifer_by_stations.items():
#    aquifer_by_stations[key] = create_diffs(data, MAX_DAYS_AHEAD, FEATURE_TO_DIFF)
#
## Create shifts of the data from 1 to DAYS_TO_SHIFT days ahead
#for key, data in aquifer_by_stations.items():
#    aquifer_by_stations[key] = create_shifts(data, DAYS_TO_SHIFT, COLUMNS_TO_SHIFT)
#
## Get the features that are to be averaged
## these are the features that were shifted + their shifts
#features_to_avg = COLUMNS_TO_SHIFT
#for feature in COLUMNS_TO_SHIFT:
#    for shift in range(1, DAYS_TO_SHIFT+1):
#        features_to_avg.append(f'{feature}_shift{shift}')
#
## Average the features
#for key, data in aquifer_by_stations.items():
#    aquifer_by_stations[key] = create_avg(data, DAYS_TO_AVG, features_to_avg)


# Process each station one at a time and clean up
for key, data in list(aquifer_by_stations.items()):
    print(f"Processing station: {key}")
    
    # Create diffs for predicting multiple days ahead
    data = create_diffs(data, MAX_DAYS_AHEAD, FEATURE_TO_DIFF)
    
    # Create shifts of the data from 1 to DAYS_TO_SHIFT days ahead
    data = create_shifts(data, DAYS_TO_SHIFT, COLUMNS_TO_SHIFT)
    
    # Get the features that are to be averaged
    features_to_avg = COLUMNS_TO_SHIFT.copy()
    for feature in COLUMNS_TO_SHIFT:
        for shift in range(1, DAYS_TO_SHIFT+1):
            features_to_avg.append(f'{feature}_shift{shift}')
    
    # Average the features
    data = create_avg(data, DAYS_TO_AVG, features_to_avg)
    
    # Update the dictionary with processed data
    aquifer_by_stations[key] = data
    
    # Force garbage collection after each station
    del data, features_to_avg
    gc.collect()
    
    print(f"Completed station: {key}")

# Save the data
joblib.dump(aquifer_by_stations.copy(), os.path.join(ROOT_PATH, SAVING_PATH))