import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('cohort_11_14.csv')

# Identify rows with NaN in MONTHS_SINCE_INSTALL and set them as censored
data['censored'] = pd.isnull(data['MONTHS_SINCE_INSTALL']).astype(int)

# For Kaplan-Meier, set NaNs in MONTHS_SINCE_INSTALL to a large number
# (indicating no event up to the end of the study)
max_time = data['MONTHS_SINCE_INSTALL'].max()
data['MONTHS_SINCE_INSTALL'].fillna(max_time + 1, inplace=True)

kmf = KaplanMeierFitter()

plt.figure(figsize=(10, 6))

for racking_type in data['RACKING_BUCKET'].unique():
    mask = data['RACKING_BUCKET'] == racking_type
    # Fit the model
    kmf.fit(data['MONTHS_SINCE_INSTALL'][mask],
            event_observed=(1 - data['censored'][mask]) *
            data['ROOF_LEAK_FLAG'][mask],
            label=racking_type)
    # Invert the survival function to get damage probability
    plt.step(kmf.cumulative_density_.index, kmf.cumulative_density_[
             racking_type], where="post", label=racking_type)

plt.title('Cumulative Damage Probability Curve - Roof Leak by Racking Type')
plt.xlabel('Months Since Installation')
plt.ylabel('Cumulative Damage Probability')
plt.legend()
plt.show()
