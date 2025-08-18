import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
import joblib

aquifer_by_stations = joblib.load('../../data/interim/ground-water-and-weather-with-forecasts-and-additional-features.joblib')
aquifer = aquifer_by_stations[85065][-365:]

columns_to_plot = ['altitude', 'altitude_diff']
color = '#88CCEE'

# Create a figure with two subplots one on top of the other
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot altitude
ax1.plot(aquifer['date'], aquifer['altitude'], alpha=1, color=color)

# Plot altitude_diff
ax2.plot(aquifer['date'], aquifer['altitude_diff'], alpha=1, color=color)

# Configure the altitude plot
ax1.set_title('Višina gladine')
ax1.set_xlabel('Datum')
ax1.set_ylabel('Višina gladine')
#ax1.legend()
ax1.grid(True, alpha=0.3)

# Configure the altitude_diff plot
ax2.set_title('Razlika višin gladine')
ax2.set_xlabel('Datum')
ax2.set_ylabel('Razlika višin gladine')
#ax2.legend()
ax2.grid(True, alpha=0.3)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot
plt.savefig('../../reports/all_models/figures/altitude_vs_altitude_diff.pdf',  bbox_inches='tight')

# Show the plot
plt.show()





