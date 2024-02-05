# importing libraries
import statsmodels.api as sm
import pandas as pd

# loading the training dataset
df = pd.read_csv('liquidPrecip_11_15.csv')
df.dropna(inplace=True)

# defining the dependent and independent variables
Xtrain = df['AVG_LIQUID_PRECIP']
ytrain = df['ROOF_LEAK_FLAG']

Xtrain = sm.add_constant(Xtrain)

# building the model and fitting the data
log_reg = sm.Logit(ytrain, Xtrain).fit()

print(log_reg.summary())
