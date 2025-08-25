# Imports
import pandas as pd
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Constants and dictionaries
SAVE_PATH_FIGURES = '../../reports/all_models/figures'
RESULTS_PATH = '../../scripts/results'
SAVE_PATH_TABLES = '../../reports/all_models/tables'
MODEL_FOLDER_NAMES = ['deep_learning_models', 'mlp_models', 'time_moe_covariates', 'foundation_models', 'statistical_models', 'time_moe_covariate_models']

# Dictionary of the models based on their type (folder name)
MODEL_DICT = {
    'deep_learning_models': ['deepar', 'n_beats_x', 'n_hits_multivariate', 'n-beats', 'patch-tst', 'n-hits'],
    'mlp_models': ['mlp_multivariate', 'mlp_univariate'],
    'foundation_models': ['chronos', 'time_moe', 'timesfm_multivariate', 'timesfm_univariate'],
    'statistical_models': ['gradient_boosting_multivariate', 'gradient_boosting_univariate', 
                           'linear_regression_multivariate', 'linear_regression_univariate',
                           'random_forest_multivariate', 'random_forest_univariate',
                           'ridge_regression_multivariate', 'ridge_regression_univariate'],
    'time_moe_covariate_models': ['hidden_layer + MLP_mlp',
                                  'hidden_layer + statistical_gradient_boosting',
                                  'hidden_layer + statistical_random_forest',
                                  'hidden_layer + statistical_ridge_regression',
                                  'hidden_layer + statistical_linear_regression',
                                  'hidden_layer + deep_learning_n_beats_x',
                                  'hidden_layer + deep_learning_n_hits_multivariate',
                                  'statistical + time_moe_gradient_boosting',
                                  'statistical + time_moe_random_forest',
                                  'statistical + time_moe_ridge_regression',
                                  'statistical + time_moe_linear_regression',
                                  'statistical + Time-MoE + statistical_gradient_boosting',
                                  'statistical + Time-MoE + statistical_random_forest',
                                  'statistical + Time-MoE + statistical_ridge_regression',
                                  'statistical + Time-MoE + statistical_linear_regression',
                                  'time_moe + statistical_gradient_boosting',
                                  'time_moe + statistical_random_forest',
                                  'time_moe + statistical_ridge_regression',
                                  'time_moe + statistical_linear_regression',
                                  'last_layer_weather_injection_weather_mlp']
}

# Function that finds the highest index of a file in the folder.   
def find_highest_index(path, file_name):
    top_index = -1
    for file in os.listdir(path):
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

    return top_index

# Function that concatenates the index and the file_name to get the file_name with the highest index.
def concatenate_index_and_file_name(file_name, index):
    name = f'{file_name}_{index}'
    return name

# Function that averages the results
def average_results(results):
    average = []
    for result in results:
        average.append(round(np.mean(result), 3))
    return average

def save_pandas_as_figure(df, path):
    fig, ax = plt.subplots(figsize=(len(df.columns)*4, 5))  # Set figure size
    ax.axis('off')  # Hide axes
    
    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    
    plt.tight_layout()
    plt.savefig(path)  # Save as PDF
    plt.close()

# Function that finds all of the specific file names (index agnostic) in the folder.
# NOT FINISHED
'''def find_specific_file_names(path):
    specific_file_names = []
    for file in os.listdir(path):
        if file.endswith('.json'):
            whole_name = file.split('.')[0]
            if '_' not in whole_name:
                specific_file_names.append(whole_name)'''



# Main function
# Traverses all of the folders with results
# In each folder, it traverses all of the files
# For each specific file name (index agnostic) it finds the highest index of that file name
# It takes that file, finds the scores, averages them and stores them in a pandas dataframe
# The dataframe is the same for all of the results in a specific folder
for folder_name in MODEL_DICT.keys():
    results_dataframe = pd.DataFrame()
    # Check if a folder exists
    if os.path.isdir(os.path.join(RESULTS_PATH, folder_name)):
        # Find all of the specific file names (index agnostic) in the folder
        specific_file_names = MODEL_DICT[folder_name]
        for file_name in specific_file_names:
            index = find_highest_index(os.path.join(RESULTS_PATH, folder_name), file_name)
            full_file_name = concatenate_index_and_file_name(file_name, index) if index != -1 else file_name
            full_file_name = f'{full_file_name}.json'
            file_path = os.path.join(RESULTS_PATH, folder_name, full_file_name)
            with open(file_path, 'r') as f:
                data = json.load(f)
                r2_scores = data['r2_scores']
                r2_scores_avg = average_results(r2_scores)
                results_dataframe[file_name] = r2_scores_avg

        # Save the dataframe
        results_dataframe.to_csv(os.path.join(SAVE_PATH_TABLES, f'{folder_name}.csv'), index=False)
        
        # Save the dataframe as a figure
        save_pandas_as_figure(results_dataframe, os.path.join(SAVE_PATH_FIGURES, f'{folder_name}.pdf'))

        
    else:
        raise RuntimeError(f"Expected a directory but found a file in {os.path.join(RESULTS_PATH, folder_name)}.")
