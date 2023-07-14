import pandas as pd
import joblib

# Load the trained pipeline from file
loaded_pipeline = joblib.load('football_prediction_pipeline.joblib')

# Specify a new match to predict
new_data = pd.DataFrame({
    'Year': [2023],
    'Month': [7],
    'Day': [22],
    'Home': ['AIK'],
    'Away': ['Malmo FF'],
    "PH": ["3.76"],
    "PD": ["3.69"],
    "PA": ["1.91"]
})

# Make the prediction
prediction = loaded_pipeline.predict(new_data)

print('Predicted result:', prediction)
