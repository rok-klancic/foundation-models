# Imports
import pandas as pd
import os
import matplotlib.pyplot as plt

# Constants and dictionaries
# ---------------------------
SAVE_FIGURES_PATH = '../../reports/all_models/figures'
TABLES_PATH = '../../reports/all_models/tables'

UNIVARIATE_MODEL_NAMES = ['gradient_boosting_univariate', 
                          'linear_regression_univariate', 
                          'random_forest_univariate', 
                          'chronos', 
                          'time_moe', 
                          'deepar', 
                          'n-beats', 
                          'patch-tst']

MULTIVARIATE_MODEL_NAMES = ['gradient_boosting_multivariate',
                            'linear_regression_multivariate',
                            'random_forest_multivariate',
                            'ridge_regression_multivariate',
                            'n_beats_x',
                            'n_hits_multivariate']

TABLES_TO_CONCATENATE = ['deep_learning_models', 'foundation_models', 'statistical_models']




# Main part
# ----------------------------
# Get the deep_learning_models, foundation_models and statistical_models tables
# and concatenate them
results_df = pd.DataFrame()
for table_name in TABLES_TO_CONCATENATE:
    table = pd.read_csv(os.path.join(TABLES_PATH, f'{table_name}.csv'))
    results_df = pd.concat([results_df, table], axis=1)

# Make a table with only univariate models
univariate_models_df = results_df[UNIVARIATE_MODEL_NAMES]

# Create a bar plot of the univariate models
univariate_models_df.plot.bar(figsize=(10, 5))
plt.savefig(os.path.join(SAVE_FIGURES_PATH, 'univariate_models.pdf'))


# Make a table with only multivariate models
multivariate_models_df = results_df[MULTIVARIATE_MODEL_NAMES]
multivariate_models_df.plot.bar(figsize=(10, 5))
plt.savefig(os.path.join(SAVE_FIGURES_PATH, 'multivariate_models.pdf'))