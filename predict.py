import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

file_path = 'Employee Sample Data.xlsx'
data = pd.read_excel(file_path)

features = data[['Job Title', 'Department', 'Business Unit', 'Age', 'Bonus %']]
target = data['Annual Salary']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Job Title', 'Department', 'Business Unit'])
    ], remainder='passthrough')

# Building the pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Training
model_pipeline.fit(X_train, y_train)

# Predict
y_pred = model_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'root_mean_squared_error = {rmse}')

#['Job Title', 'Department', 'Business Unit', 'Age', 'Bonus %']
#Returns: Predicted salaries
def predict_salary(new_data):
    return model_pipeline.predict(new_data)

example_data = pd.DataFrame({
    'Job Title': ['Sr. Manger', 'Technical Architect'],
    'Department': ['IT', 'IT'],
    'Business Unit': ['Research & Development', 'Manufacturing'],
    'Age': [45, 39],
    'Bonus %': [0.10, 0.12]
})

predicted_salaries = predict_salary(example_data)
print(predicted_salaries)

