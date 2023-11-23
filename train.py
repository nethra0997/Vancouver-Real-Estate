import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import joblib

def load_data(csv_path):
    data = pd.read_csv(csv_path)
    features = data.drop(columns=['Sold_price'])
    target = data['Sold_price']
    return features, target

def train_model(x_train, y_train, max_depth=None):
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
    model.fit(x_train, y_train)
    return model

# Path to the dataset
csv_path = 'final_data_test.csv'

# Load and preprocess the data
features, target = load_data(csv_path)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

# Train the model
model = train_model(x_train, y_train, max_depth=5)  # Adjust max_depth as needed

# Save the model
joblib.dump(model, 'vancouver_housing_model.joblib')

# Optionally, print out a message to indicate successful completion
print("Model trained and saved successfully.")
