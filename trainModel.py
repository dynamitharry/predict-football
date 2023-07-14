import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib

# Load the data
data = pd.read_csv('SWE.csv')

# Specify features and target
features = ['Home', 'Away', 'PH', 'PD', 'PA']
target = 'Res'

# Create dataframes for features and target
X = data[features].copy()  # Create a copy of the data to avoid SettingWithCopyWarning
X['Year'] = pd.DatetimeIndex(data['Date']).year
X['Month'] = pd.DatetimeIndex(data['Date']).month
X['Day'] = pd.DatetimeIndex(data['Date']).day
y = data[target]

# Specify categorical and numerical features
numerical_features = ['Year', 'Month', 'Day', 'PH', 'PD', 'PA']
categorical_features = ['Home', 'Away']

# Create transformers for preprocessing
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values by replacing with median
    ('scaler', StandardScaler())])  # Scale the numerical features

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Handle missing values by replacing with 'missing'
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])  # One-hot encode categorical features

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_features),
        ('cat', cat_transformer, categorical_features)])

# Define the model
model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)

# Create the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Use TimeSeriesSplit to split the data
tscv = TimeSeriesSplit(n_splits=7)

# Loop over all splits
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    print('Accuracy:', accuracy_score(y_test, y_pred))

# Save the trained pipeline to a file
joblib.dump(pipeline, 'football_prediction_pipeline.joblib')
