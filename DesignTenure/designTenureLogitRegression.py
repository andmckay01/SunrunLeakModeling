import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

data = pd.read_csv('designTenure.csv')

# Select predictor variables and the target variable
predictor_vars = [
    'SYSTEM_SIZE_DC',
    'STORAGE_FLAG',
    'SOW_FLAG',
    # 'HAS_AVOIDABLE_PRE_SUB_REJECTION',
    # 'HAS_REDLINES',
    'TENURE'
]

target_var = 'HAS_REDLINES'

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