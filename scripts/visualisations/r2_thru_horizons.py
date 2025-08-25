# Imports
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

# Constants and dictionaries
# ---------------------------
SAVE_FIGURES_PATH = '../../reports/all_models/figures'
TABLES_PATH = '../../reports/all_models/tables'

TABLES_TO_CONCATENATE = ['deep_learning_models', 'foundation_models', 'statistical_models']

# Main part
# ----------------------------
# Get the deep_learning_models, foundation_models and statistical_models tables
# and concatenate them
results_df = pd.DataFrame()
for table_name in TABLES_TO_CONCATENATE:
    table = pd.read_csv(os.path.join(TABLES_PATH, f'{table_name}.csv'))
    results_df = pd.concat([results_df, table], axis=1)

# Extract linear regression univariate and multivariate R2 scores
linear_univariate = results_df['linear_regression_univariate']
linear_multivariate = results_df['linear_regression_multivariate']

# Create horizons (assuming the index represents the horizon number)
horizons = range(1, len(linear_univariate) + 1)

# Create a line plot comparing univariate vs multivariate linear regression
plt.figure(figsize=(10, 6))
plt.plot(horizons, linear_univariate, marker='o', label='Univariatna linearna regresija', linewidth=2, color='#332288')
plt.plot(horizons, linear_multivariate, marker='o', label='Multivariatna linearna regresija', linewidth=2, color='#CC6677')

plt.xlabel('Napovedno obzorje')
plt.ylabel('Vrednosti R²')
plt.title('Vrednosti R² za univariatno in multivariatno linearno regresijo')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(SAVE_FIGURES_PATH, 'linear_regression_r2_through_horizons.pdf'), bbox_inches='tight')
plt.show()