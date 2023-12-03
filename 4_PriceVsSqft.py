import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset
df = pd.read_csv('D:/5800/Project/final_data_test.csv')

# Perform the linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(df['Sqft'], df['Sold_price'])

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create the scatter plot for 'Sqft' vs 'Sold_price'
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Sqft', y='Sold_price', data=df, color='lightseagreen')

# Add the linear regression line
sns.regplot(x='Sqft', y='Sold_price', data=df, scatter=False, color='lightseagreen', 
            line_kws={"label":"y={0:.1f}x+{1:.1f}".format(slope,intercept)})

# Annotate with the linear regression equation
plt.legend()
plt.title('Sqft vs Sold Price')
plt.xlabel('Sqft')
plt.ylabel('Sold Price')

# Show plot
plt.show()

# Print the regression equation
print(f"The equation of the regression line is: y = {slope:.2f}x + {intercept:.2f}")

