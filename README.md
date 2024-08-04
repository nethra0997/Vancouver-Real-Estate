# Decision-Tree-analyzing-Vancouver-real-estate-prices
# Project Title: Decision Tree Analysis of Vancouver's Real Estate Market

Objective: The main objective of this project was to analyze Vancouver's real estate market using decision tree algorithms to identify crucial factors affecting house prices and to effectively capture non-linear relationships between features and house prices.

Data Source: The dataset consists of information about various properties sold in Vancouver. The key features include neighborhood, type of house, number of bedrooms and bathrooms, size in square feet, and the sold price.

Key Steps:
# Data Preparation and Examination:
Dataset Columns:
Neighbourhood (Integer representation: 1 for Vancouver West, 2 for Downtown)
Type of house (Integer representation: 1 for Apartment, 2 for Detached House, 3 for Townhouse)
Bedrooms (Number of bedrooms)
Bathrooms (Number of bathrooms)
Sqft (Size of the house in square feet)
Sold_price (Sale price of the house)
The dataset was carefully prepared and examined to ensure accuracy and relevance for analysis.

# Data Processing and Cleaning:
Data Cleaning: Addressed missing values, inconsistencies, and ensured data integrity.
Data Visualization: Created histograms for each factor and a heatmap to explore the relationship between square footage and sold price.

# Decision Tree Algorithm Implementation:
Feature Interpretability: Clearly identified crucial factors affecting house prices, essential for stakeholders to understand valuation drivers.
Handling Non-linear Relationships: Effectively captured complex relationships between features and prices, ideal for real estate data.
Heterogeneous Data Handling: Seamlessly processed mixed data types (numerical and categorical).

Algorithm Process:
Started at the root node (square footage)
Checked for stopping conditions
Selected features
Split data
Iterative branching
Repeated until stopping conditions were met
Predicted outcomes
Model Training and Evaluation: Loaded data, trained the model, evaluated its performance, and performed tuning for optimal results.

Deployment: Deployed the model in a Streamlit app for interactive use.

# Performance Metrics:
Evaluated the decision tree model using relevant performance metrics to ensure accuracy and reliability.

# Conclusion:
Identified the need to expand the dataset, explore advanced modeling techniques, and integrate economic indicators for a more rigorous analysis.
Aimed to provide stakeholders with valuable insights into the factors affecting real estate prices in Vancouver.

# Tech Stack:
Programming Languages: Python
Libraries: Scikit-learn (for decision tree algorithm), Pandas (for data manipulation), NumPy (for numerical operations), Matplotlib and Seaborn (for data visualization)
Tools: Jupyter Notebook (for development and analysis), Streamlit (for deploying the interactive web app)
