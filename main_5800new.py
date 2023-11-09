import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

csv_path = 'final_data_test.csv'

# Function to load and preprocess the data
@st.cache_resource
def load_data(csv_path):
    data = pd.read_csv(csv_path)
    features = data.drop(columns=['Sold_price'])
    target = data['Sold_price']
    return features, target

# Function to create and train the model
def train_model(x_train, y_train, max_depth=None):
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
    model.fit(x_train, y_train)
    return model

# Function to make predictions
def predict(model, input_data):
    return model.predict(input_data)

# Streamlit application
def main():
    st.title('Vancouver Home Price Prediction')

    # User input features
    st.sidebar.header('Input Features')
    def user_input_features():
        neighbourhood = st.sidebar.number_input('Neighbourhood (1 for Vancouver West, 2 for Downtown)', min_value=1, max_value=2, value=1)
        house_type = st.sidebar.number_input('Type of House (1 for Apartment, 2 for House, 3 for Townhouse)', min_value=1, max_value=3, value=1)
        bedrooms = st.sidebar.slider('Number of Bedrooms', min_value=1, max_value=10, value=2)
        bathrooms = st.sidebar.slider('Number of Bathrooms', min_value=1, max_value=10, value=2)
        sqft = st.sidebar.slider('Size of the house (sqft)', min_value=500, max_value=10000, value=1000)
        data = {'Neighbourhood': neighbourhood,
                'Type_of_house': house_type,
                'Bedrooms': bedrooms,
                'Bathrooms': bathrooms,
                'Sqft': sqft}
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # Load and preprocess the data
    features, target = load_data(csv_path)
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

    # Display the input features
    st.header('Specified Input features')
    st.write(input_df)

    # Train and predict
    max_depth = st.sidebar.slider('Max Depth', 1, 10, 5)
    model = train_model(x_train, y_train, max_depth)
    prediction = predict(model, input_df)

    # Display prediction
    st.header('Prediction of Home Price')
    st.write('Estimated Home Price in CAD: ', prediction[0])

    # Visualize the decision tree
    st.header('Decision Tree')
    fig, ax = plt.subplots()
    plot_tree(model, filled=True, feature_names=features.columns, ax=ax)
    st.pyplot(fig)

if __name__ == '__main__':
    main()
