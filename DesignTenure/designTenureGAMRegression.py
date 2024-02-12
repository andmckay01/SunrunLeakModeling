import pandas as pd
from pygam import LinearGAM, s, f  # s is for splines, f is for factors
import seaborn as sns
import matplotlib.pyplot as plt

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

# Define the independent variables
X = data[predictor_vars].values

# Define the dependent variable
y = data[target_var].values

# Fit the GAM model
# s() specifies a spline term for continuous variables
# f() specifies a factor term for categorical variables
# The basis and number of splines may need to be adjusted based on your specific dataset
gam = LinearGAM(f(0) + f(1) + f(2) + f(3) + f(4)).fit(X, y)

# Print the summary of the model
print(gam.summary())

# Plot the partial dependency for each predictor variable
for i, term in enumerate(gam.terms):
    if term.isintercept:
        continue
    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
    
    plt.figure()
    plt.plot(XX[:, i], pdep)
    plt.plot(XX[:, i], confi, c='r', ls='--')
    plt.title(predictor_vars[i])
    plt.show()

# Since GAMs can handle non-linear relationships, you might not need to plot residuals the same way
# as for linear or negative binomial regression. However, if you still want to inspect residuals:

# Calculate the predictions
predictions = gam.predict(X)

# Calculate the residuals
residuals = y - predictions

# Plot the residuals
sns.residplot(x=predictions, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values')
plt.show()
