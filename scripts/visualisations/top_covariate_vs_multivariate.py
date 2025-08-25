# Imports
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

# Constants and dictionaries
# ---------------------------
SAVE_FIGURES_PATH = '../../reports/all_models/figures'
TABLES_PATH = '../../reports/all_models/tables'

TABLES_TO_CONCATENATE = ['deep_learning_models', 'foundation_models', 'statistical_models', 'mlp_models', 'time_moe_covariate_models']

MODELS_TO_PLOT = ['time_moe + statistical_linear_regression', 'statistical + time_moe_ridge_regression', 'linear_regression_multivariate', 'mlp_multivariate', 'timesfm_multivariate']

english_to_slovene = {
    'time_moe + statistical_linear_regression': 'ZP: TMoE + lin. reg.',
    'statistical + time_moe_ridge_regression': 'ZP: lin. reg. (L2) + TMoE', 
    'linear_regression_multivariate': 'Multivariatna linearna  regresija',
    'mlp_multivariate': 'Multivariatni MLP',
    'timesfm_multivariate': 'TimesFM'
}

# Define colors for the models
colors = ['#882255', '#CC6677', '#332288', '#44AA99', '#117733']

# Main part
# ----------------------------
# Get the deep_learning_models, foundation_models and statistical_models tables
# and concatenate them
results_df = pd.DataFrame()
for table_name in TABLES_TO_CONCATENATE:
    table = pd.read_csv(os.path.join(TABLES_PATH, f'{table_name}.csv'))
    results_df = pd.concat([results_df, table], axis=1)

# Extract the specified models from the dataframe
models_df = results_df[MODELS_TO_PLOT]

# Create horizons (assuming the index represents the horizon number)
horizons = range(1, len(models_df) + 1)

# Create a bar plot comparing the selected models
plt.figure(figsize=(10, 6))
width = 0.15
x = range(len(horizons))

for i, model in enumerate(MODELS_TO_PLOT):
    plt.bar([pos + i * width for pos in x], models_df[model], width, 
            label=english_to_slovene[model], color=colors[i % len(colors)])

plt.xlabel('Napovedno obzorje')
plt.ylabel('Vrednosti R²')
plt.title('Primerjava metod za razširitev Time-MoE in multivariatnih modelov')
plt.xticks([pos + width * 2 for pos in x], horizons)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(SAVE_FIGURES_PATH, 'top_covariate_vs_multivariate_bar_plot.pdf'), bbox_inches='tight')
plt.show()