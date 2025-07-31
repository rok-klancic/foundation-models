# Imports
import pandas as pd
import os
import matplotlib.pyplot as plt

# Constants and dictionaries
# ---------------------------
SAVE_FIGURES_PATH = '../../reports/all_models/figures'
TABLES_PATH = '../../reports/all_models/tables'


MULTIVARIATE_MODEL_NAMES = ['gradient_boosting_multivariate',
                            'linear_regression_multivariate',
                            'random_forest_multivariate',
                            'ridge_regression_multivariate',
                            'n_beats_x',
                            'n_hits_multivariate']

TABLES_TO_CONCATENATE = ['deep_learning_models', 'foundation_models', 'statistical_models']

# Functions
#-----------------------------
# Function that finds the 3 highest performing models for every horizon
# and returns a list of the names of the models
def find_highest_performing_models(df, num_of_models):
    highest_performing_models = []
    for horizon in df.index:
        highest_performing_models.append(df.loc[horizon].nlargest(num_of_models).index.tolist())
    
    # Flatten the list
    highest_performing_models = list({model for sublist in highest_performing_models for model in sublist})
    return highest_performing_models


# Main part
# ----------------------------
# Get the deep_learning_models, foundation_models and statistical_models tables
# and concatenate them
results_df = pd.DataFrame()
for table_name in TABLES_TO_CONCATENATE:
    table = pd.read_csv(os.path.join(TABLES_PATH, f'{table_name}.csv'))
    results_df = pd.concat([results_df, table], axis=1)

# Make a table with only multivariate models
multivariate_models_df = results_df[MULTIVARIATE_MODEL_NAMES]

# Load the time_moe_covariate_models table
time_moe_covariate_models_table = pd.read_csv(os.path.join(TABLES_PATH, 'time_moe_covariate_models.csv'))

# Find the top performing models for time_moe_covariate_models
top_models = find_highest_performing_models(time_moe_covariate_models_table, 3)

# Select only the top models
time_moe_covariate_models_table = time_moe_covariate_models_table[top_models]

# Add the top models to the multivariate_models_df
multivariate_models_df = pd.concat([multivariate_models_df, time_moe_covariate_models_table], axis=1)

# Create a bar plot of the multivariate models

# Specify the colors
num_models = len(multivariate_models_df.columns)
colors = plt.get_cmap('tab20').colors[:num_models]

ax = multivariate_models_df.plot.bar(figsize=(20, 8), color=colors)  # Wider and taller

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()  # Adjust layout to fit everything
plt.savefig(os.path.join(SAVE_FIGURES_PATH, 'top_multivariate_models.pdf'), bbox_inches='tight')