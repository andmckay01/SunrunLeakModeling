import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import math


def log_odds_to_probability(log_odds):
    """Convert log odds to probability."""
    odds = math.exp(log_odds)
    probability = odds / (1 + odds)
    return probability


data = pd.read_csv('logit_data_12_11.csv')

predictor_vars = ['ROOF_PITCH_BUCKET',
                  'YEARS_SINCE_INSTALL',  # included to get a better view of foreman tenure
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

# Impute missing values
imputer = SimpleImputer(strategy='mean')
data[predictor_vars] = imputer.fit_transform(data[predictor_vars])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    data[predictor_vars], data[target_var], test_size=0.3, random_state=42)

# Create and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("Intercept (Bias):", model.intercept_)
print("Coefficients:")
for feature, coef in zip(predictor_vars, model.coef_[0]):
    print(f"{feature}: {coef}")
    print(f"{feature}_prob: {log_odds_to_probability(coef)}")
