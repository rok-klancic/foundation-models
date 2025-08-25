# Imports
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

# Constants and dictionaries
# ---------------------------
SAVE_FIGURES_PATH = '../../reports/all_models/figures'
TABLES_PATH = '../../reports/all_models/tables'

TABLES_TO_CONCATENATE = ['deep_learning_models', 'foundation_models', 'statistical_models', 'mlp_models']

MODELS_TO_PLOT = ['time_moe', 'mlp_univariate', 'linear_regression_multivariate', 'mlp_multivariate']

english_to_slovene = {
    'time_moe': 'Time-MoE',
    'mlp_univariate': 'Univariatni MLP',
    'linear_regression_multivariate': 'Multivariatna linearna regresija',
    'mlp_multivariate': 'Multivariatni MLP'
}

# Define colors for the models
colors = ['#332288', '#88CCEE', '#CC6677', '#882255']

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
width = 0.2
x = range(len(horizons))

for i, model in enumerate(MODELS_TO_PLOT):
    plt.bar([pos + i * width for pos in x], models_df[model], width, 
            label=english_to_slovene[model], color=colors[i % len(colors)])

plt.xlabel('Napovedno obzorje')
plt.ylabel('Vrednosti R²')
plt.title('Primerjava univariatnih in multivariatnih modelov')
plt.xticks([pos + width * 1.5 for pos in x], horizons)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(SAVE_FIGURES_PATH, 'univariate_vs_multivariate_models.pdf'), bbox_inches='tight')
plt.show()