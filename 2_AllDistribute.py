import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('D:/5800/Project/final_data_test.csv')

# Set the aesthetic style of the plots
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))

# Create the Histogram for the distribution of Neighbourhood
sns.histplot(df['Neighbourhood'], kde=True, color='lightseagreen') 
plt.title('Distribution of Neighbourhood')
plt.xlabel('Neighbourhood')
plt.ylabel('Count')
plt.show()

# Create the Histogram for the distribution of House Type
sns.histplot(df['Type_of_house'], kde=True, color='lightseagreen') 
plt.title('Distribution of House Type')
plt.xlabel('Type_of_house')
plt.ylabel('Count')
plt.show()

# Create the Histogram for the distribution of Bedrooms
sns.histplot(df['Bedrooms'], kde=True, color='lightseagreen') 
plt.title('Distribution of Bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
plt.show()

# Create the Histogram for the distribution of Bathrooms
sns.histplot(df['Bathrooms'], kde=True, color='lightseagreen') 
plt.title('Distribution of Bathrooms')
plt.xlabel('Bathrooms')
plt.ylabel('Count')
plt.show()

# Create the Histogram for the distribution of Sqft
sns.histplot(df['Sqft'], kde=True, color='lightseagreen') 
plt.title('Distribution of Sqft')
plt.xlabel('Sqft')
plt.ylabel('Count')
plt.show()

# Create the Histogram for the distribution of Sold Price
sns.histplot(df['Sold_price'], kde=True, color='lightseagreen') 
plt.title('Distribution of Sold Price')
plt.xlabel('Sold Price')
plt.ylabel('Count')
plt.show()