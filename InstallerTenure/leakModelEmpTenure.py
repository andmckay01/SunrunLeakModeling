import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

data = pd.read_csv('logit_data_12_12.csv')

# Select predictor variables and the target variable
predictor_vars = [
    'ROOF_PITCH_BUCKET',
    'YEARS_SINCE_INSTALL',
    'LIQUID_PRECIP_BUCKET',
    # 'ROOF_CONCRETE_S_TILE',
    # 'RACK_RL',
    # 'RACK_S',
    # 'RACK_UR',
    # 'RACK_TOPSPEED',
    'FOREMAN_TENURE_YRS',
    'AVG_EMP_TENURE_YRS'
]

target_var = 'ROOF_LEAK_FLAG'

# Drop rows with missing values
# all data columns are fille dout except the "tenure" columns
# therefore, this will drop any row that is missing foreman tenure or employee tenure years
# insights derived from this model should be limited to foreman and employee tenure, with other variables merely used for controls
data.dropna(subset=predictor_vars + [target_var], inplace=True)


# Define the independent variables (add a constant term to represent the intercept)
X = sm.add_constant(data[predictor_vars])

# Define the dependent variable
y = data[target_var]

# Fit the logistic regression model
log_reg_model = sm.Logit(y, X).fit()

# Print the summary of the model
print(log_reg_model.summary())

# Calculate VIFs for each predictor variable
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(
    X.values, i) for i in range(X.shape[1])]

# Print the VIFs
print("\nVariance Inflation Factor (VIF) for each feature:")
print(vif_data)
