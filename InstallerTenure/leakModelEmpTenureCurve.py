import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('logit_data_12_11.csv')

# Select predictor variables and the target variable
predictor_vars = [
    'ROOF_PITCH_BUCKET',
    'YEARS_SINCE_INSTALL',
    'LIQUID_PRECIP_BUCKET',
    'ROOF_CONCRETE_S_TILE',
    'RACK_RL',
    'RACK_S',
    'RACK_UR',
    'RACK_TOPSPEED',
    'FOREMAN_TENURE_YRS',
    'AVG_EMP_TENURE_YRS'
]
target_var = 'ROOF_LEAK_FLAG'

# Drop rows with missing values
data.dropna(subset=predictor_vars + [target_var], inplace=True)

# Define the independent variables (add a constant term to represent the intercept)
X = sm.add_constant(data[predictor_vars])

# Define the dependent variable
y = data[target_var]

# Fit the logistic regression model
log_reg_model = sm.Logit(y, X).fit()

# Generate a range of values for AVG_EMP_TENURE_YRS to predict probabilities
tenure_range = np.linspace(data['AVG_EMP_TENURE_YRS'].min(
), data['AVG_EMP_TENURE_YRS'].max(), 10000)
X_new = pd.DataFrame({'AVG_EMP_TENURE_YRS': tenure_range})

# Add a constant and set all other variables to their mean values for control
X_new = sm.add_constant(X_new)
for var in predictor_vars:
    if var != 'AVG_EMP_TENURE_YRS':
        X_new[var] = data[var].mean()

# Predict probabilities
predicted_probabilities = log_reg_model.predict(X_new)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(tenure_range, predicted_probabilities, color='blue')
plt.xlabel('Average Employee Tenure Years')
plt.ylabel('Probability of Roof Leak')
plt.title('Probability of Roof Leak vs Average Employee Tenure Years')
plt.grid(True)
plt.show()
