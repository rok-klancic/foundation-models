# Imports
import pandas as pd
import os

# Constants and dictionaries
SAVE_PATH_FIGURES = '../../reports/all_models/figures'
RESULTS_PATH = '../../scripts/results'
SAVE_PATH_TABLES = '../../reports/all_models/tables'
MODEL_FOLDER_NAMES = ['deep_learning_models', 'time_moe_covariates', 'foundation_models', 'statistical_models', 'time_moe_covariate_models']

# Dictionary of the models based on their type (folder name)
MODEL_DICT = {
    'deep_learning_models': ['deepar', 'n_beats_x', 'n_hits_multivariate', 'n-beats', 'patch-tst'],
    'time_moe_covariates': 'Time MOE Covariates',
    'foundation_models': ['chronos', 'time_moe'],
    'statistical_models': ['gradient_boosting_multivariate', 'gradient_boosting_univariate', 
                           'linear_regression_multivariate', 'linear_regression_univariate',
                           'random_forest_multivariate', 'random_forest_univariate',
                           'ridge_regression_multivariate'],
    'time_moe_covariate_models': ['hidden_layer + MLP_mlp',
                                  'hidden_layer + statistical_gradient_boosting',
                                  'hidden_layer + statistical_random_forest',
                                  'hidden_layer + statistical_ridge_regression',
                                  'hidden_layer + statistical_linear_regression',
                                  'hidden_layer + deep_learning_n_beats_x',
                                  'hidden_layer + deep_learning_n_hits_multivariate',
                                  'hidden_layer + deep_learning_n-beats',
                                  'hidden_layer + deep_learning_patch-tst']
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

# Function that finds all of the specific file names (index agnostic) in the folder.
def find_specific_file_names(path):
    specific_file_names = []
    for file in os.listdir(path):
        if file.endswith('.json'):
            whole_name = file.split('.')[0]
            if '_' not in whole_name:
                specific_file_names.append(whole_name)



# Main function
# Traverses all of the folders with results
# In each folder, it traverses all of the files
# For each specific file name (index agnostic) it finds the highest index of that file name
# It takes that file, finds the scores, averages them and stores them in a pandas dataframe
# The dataframe is the same for all of the results in a specific folder
for folder_name in MODEL_FOLDER_NAMES:
    results = pd.DataFrame()
    # Check if a folder exists
    if os.path.isdir(os.path.join(RESULTS_PATH, folder_name)):
        # Find all of the specific file names (index agnostic) in the folder
        specific_file_names = find_specific_file_names(os.path.join(RESULTS_PATH, folder_name))
        
        
    else:
        raise RuntimeError("Expected a directory but found a file in RESULTS_PATH.")
