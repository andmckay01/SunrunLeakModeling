import pandas as pd
import statsmodels.api as sm

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

# Print the summary of the model
print(log_reg_model.summary())
