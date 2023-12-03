import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('D:/5800/Project/final_data_test.csv')

# Display the first 6 rows of the dataset
print(df.head(6))

# Set the display format
pd.set_option('display.float_format', '{:.2f}'.format)

# Generate the descriptive statistics of dataset
stats = df.describe()

# Display the statistical analysis result
print(stats)
