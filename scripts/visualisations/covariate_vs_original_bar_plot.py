# Imports
import pandas as pd
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Constants and dictionaries
# ---------------------------
SAVE_FIGURES_PATH = '../../reports/all_models/figures'
TABLES_PATH = '../../reports/all_models/tables'

# Dictionary that maps the original model names to the covariate model names
ORIGINAL_TO_COVARIATE_MODEL_NAMES = {'linear_regression_multivariate':'linear_regression', 
                                     'random_forest_multivariate':'random_forest', 
                                     'gradient_boosting_multivariate':'gradient_boosting', 
                                     'ridge_regression_multivariate':'ridge_regression', 
                                     'n_beats_x':'n_beats_x', 
                                     'n_hits_multivariate':'n_hits_multivariate'}

# Functions
# ----------------------------
def find_columns_that_contain_model_name(table, model_name):
    columns = []
    for column in table.columns:
        if model_name in column:
            columns.append(column)
    return columns


# Main part
# ----------------------------
# Concatenate deep_learning_models and 'statistical_models' tables
deep_learning_models_table = pd.read_csv(os.path.join(TABLES_PATH, 'deep_learning_models.csv'))
statistical_models_table = pd.read_csv(os.path.join(TABLES_PATH, 'statistical_models.csv'))

deep_learning_and_statistical_models_table = pd.concat([deep_learning_models_table, statistical_models_table], axis=1)

# Load the time_moe_covariate_models table
time_moe_covariate_models_table = pd.read_csv(os.path.join(TABLES_PATH, 'time_moe_covariate_models.csv'))

# Main loop
# Go through all of the models in original_to_covariate_model_names
# For each model find the appropriate column in the deep_learning_and_statistical_models_table
# and all of the appropriate columns in the time_moe_covariate_models_table
# For each model, combine the columns and save the bar plots

for original_model_name, covariate_model_name in ORIGINAL_TO_COVARIATE_MODEL_NAMES.items():
    result_df = pd.DataFrame()
    # Columns that contain the covariate model name
    covariate_model_columns = find_columns_that_contain_model_name(time_moe_covariate_models_table, covariate_model_name)

    # Add the original model column to the result_df
    result_df[original_model_name] = deep_learning_and_statistical_models_table[original_model_name]
    
    # Add the covariate model columns to the result_df
    for covariate_model_column in covariate_model_columns:
        result_df[covariate_model_column] = time_moe_covariate_models_table[covariate_model_column]
    
    # Save the barplot of the result_df
    result_df.plot.bar(figsize=(10, 5))
    plt.savefig(os.path.join(SAVE_FIGURES_PATH, f'{original_model_name}_vs_covariates.pdf'))








