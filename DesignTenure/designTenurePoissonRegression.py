import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.discrete.discrete_model import NegativeBinomial

data = pd.read_csv('designTenure3.csv')

# Select predictor variables and the target variable
predictor_vars = [
    'SYSTEM_SIZE_DC',
    'STORAGE_FLAG',
    'SOW_FLAG',
    'TENURE',
    'JURISDICTION_IS_HARD',
]

target_var = 'TOTAL_ADHOCS'

# Drop rows with missing values
data.dropna(subset=predictor_vars + [target_var], inplace=True)

# Define the independent variables (add a constant term to represent the intercept)
X = sm.add_constant(data[predictor_vars])

# Define the dependent variable
y = data[target_var]

# Fit the negative binomial regression model
nb_model = NegativeBinomial(y, X).fit()

# Print the summary of the model
print(nb_model.summary())

# Note: For the Negative Binomial model, calculating Variance Inflation Factor (VIF) and plotting residuals 
# can still be useful for diagnostic purposes, but interpretation and diagnostics may differ from OLS due to 
# the nature of count data and the model itself.

# Assuming you still want to calculate VIF and plot residuals for exploratory analysis:

# Since the Negative Binomial model's predictions and residuals don't directly correspond to the same concepts 
# in linear regression, consider reviewing model diagnostics specific to count models for a more accurate analysis.

# For a simple residual plot (though interpret with caution):
predictions = nb_model.predict(X)
residuals = y - predictions

sns.residplot(x=predictions, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values')
plt.show()
