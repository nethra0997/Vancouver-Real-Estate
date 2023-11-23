import streamlit as st
import pandas as pd
import joblib
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt

# Function to load the trained model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = joblib.load(model_path)
    return model

# Function to get user input features
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

def main():
    st.title('Vancouver Home Price Prediction')
    input_df = user_input_features()
    st.write('Specified Input features', input_df)

    # Load the trained model
    model = load_model('vancouver_housing_model.joblib')

    # Predict the price
    prediction = model.predict(input_df)
    st.write('Estimated Home Price in CAD: ', prediction[0])

    # Visualize the decision tree
    if st.checkbox("Show Decision Tree"):
        fig, ax = plt.subplots()
        plot_tree(model, filled=True, feature_names=input_df.columns, ax=ax)
        st.pyplot(fig)

if __name__ == '__main__':
    main()
