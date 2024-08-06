import pandas as pd
from sklearn.model_selection import train_test_split
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Sample dataset
sample_data = {
    'age': [63, 67, 67, 37, 41, 56, 62, 57, 63, 53],
    'sex': [1, 1, 1, 1, 0, 1, 0, 0, 1, 1],
    'chol': [233, 286, 229, 250, 204, 236, 268, 354, 254, 203],
    'trestbps': [145, 160, 120, 130, 130, 120, 140, 140, 135, 140],
    'fbs': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'heart_disease': [1, 1, 1, 0, 0, 1, 0, 1, 1, 0]
}

# Convert the sample data into a DataFrame
heart_data = pd.DataFrame(sample_data)

# Discretize the 'age' variable into categories
age_bins = [20, 40, 60, 80]
age_labels = ['20-39', '40-59', '60-79']
heart_data['age'] = pd.cut(heart_data['age'], bins=age_bins, labels=age_labels)

# Convert columns to categorical types
for col in heart_data.columns:
    heart_data[col] = heart_data[col].astype('category')

# Display the first few rows of the dataset
print(heart_data.head())

# Split the data into training and testing sets
train_data, test_data = train_test_split(heart_data, test_size=0.2, random_state=42)

# Define the structure of the Bayesian Network
model = BayesianNetwork([('age', 'trestbps'),
                         ('age', 'fbs'),
                         ('sex', 'trestbps'),
                         ('trestbps', 'heart_disease'),
                         ('chol', 'heart_disease'),
                         ('fbs', 'heart_disease')])

# Fit the model using Maximum Likelihood Estimation
model.fit(train_data, estimator=MaximumLikelihoodEstimator)

# Perform inference
infer = VariableElimination(model)

# Query the model to calculate the probability of heart disease given new data
query_result = infer.query(variables=['heart_disease'], evidence={
    'age': '40-59',  # Use discrete age category
    'sex': 1,
    'chol': 250,
    'trestbps': 130,
    'fbs': 0
})

print(query_result)