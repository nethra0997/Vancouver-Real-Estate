import pandas as pd
from scipy.interpolate._ppoly import evaluate
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np


# Load the csv file
main_file_path = 'final_data_test.csv'
data = pd.read_csv(main_file_path)


# Prepare data for analysis
price_variable = data.Sold_price
data_predicator = list(data.dtypes[data.dtypes == "int64"].index)
data_predicator = data_predicator[:-1]
predicting_factors = data[data_predicator]


# Split data in half for training and testing
x_train, x_test, y_train, y_test = train_test_split(predicting_factors, price_variable, test_size=0.2, random_state = 42)


# Function to test the accuracy of the model with different max depth
def test_max_depth():
    for maximum_depth in [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50, 100]:
        # Fit the model with the data
        model = DecisionTreeRegressor(max_depth = maximum_depth, random_state = 42, splitter='best')
        model.fit(x_train, y_train)
        # Evaluate model accuracy
        predicted_price = model.predict(x_test)
        print('max_depth:', maximum_depth)        
        print('mean_absolute_error:', mean_absolute_error(y_test, predicted_price))
        print('mean_absolute_percentage_error: {:0.4f}%'.format(mean_absolute_percentage_error(y_test, predicted_price)*100))
        print('Accuracy: {:0.4f}%'.format(100 - mean_absolute_percentage_error(y_test, predicted_price)*100))

# Function to test the accuracy of the model with different minimum number of samples required to split an internal node.
def test_min_samples_split():
    for samples_split in [1, 2, 5, 10, 20, 30, 50, 100]:
        model=DecisionTreeRegressor(min_samples_split = samples_split, random_state = 42, splitter='best')
        model.fit(x_train, y_train)
        preds_val = model.predict(x_test)
        print('min_samples_split:', samples_split)    
        print('mean_absolute_error:', mean_absolute_error(y_test,preds_val))
        print('mean_absolute_percentage_error: {:0.2f}%'.format(mean_absolute_percentage_error(y_test,preds_val)*100))


# Run the tests to find the best data
test_max_depth()
test_min_samples_split()


# Function to draw the final decision tree
def draw_decision_tree(model_provided, data_predicator_provided):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize = (15, 15), dpi=500)
    tree.plot_tree(model_provided, feature_names = data_predicator_provided, filled = True)
    fig.savefig('imagename.png')


# Fit the model with the best data
model = DecisionTreeRegressor(max_depth=5, random_state = 42, splitter='best')
model.fit(x_train, y_train)

# Evaluate model accuracy
predicted_price = model.predict(x_test)
print("The final decision tree has max_depth of 5.")      
print('Accuracy: {:0.4f}%'.format(100 - mean_absolute_percentage_error(y_test, predicted_price)*100))

# draw the decision tree
draw_decision_tree(model, data_predicator)
