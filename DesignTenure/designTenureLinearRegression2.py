import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

data = pd.read_csv('designTenure2.csv')

# data = data[(data['SIGNED_CAPP_GOOD'] == 1) & 
#                      (data['INSTALL_BRANCH_SUB_REGION'] == 'Hawaii')]

data = data[(data['JURISDICTION_IS_HARD'] == 1)]

# Select predictor variables and the target variable
predictor_vars = [
    'SYSTEM_SIZE_DC',
    'STORAGE_FLAG',
    'SOW_FLAG',
    'TENURE',
]

target_var = 'TOTAL_ADHOCS'

    # ,TOTAL_TIME_SIGNED_CAPP
    # ,TOTAL_TIME_CAPP_CAPA
    # ,TOTAL_TIME_CAPA_HO
    # ,TOTAL_TIME_HO_INSTALL


# Drop rows with missing values
data.dropna(subset=predictor_vars + [target_var], inplace=True)

# Define the independent variables (add a constant term to represent the intercept)
X = sm.add_constant(data[predictor_vars])

# Define the dependent variable
y = data[target_var]

# Fit the linear regression model
linear_reg_model = sm.OLS(y, X).fit()  # Use OLS here instead of Logit

# Print the summary of the model
print(linear_reg_model.summary())

# Calculate VIFs for each predictor variable
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(
    X.values, i) for i in range(X.shape[1])]

# Print the VIFs
print("\nVariance Inflation Factor (VIF) for each feature:")
print(vif_data)

# Calculate the predictions
predictions = linear_reg_model.predict(X)

# Calculate the residuals
residuals = y - predictions

# Plot the residuals
sns.residplot(x=predictions, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values')
plt.show()


