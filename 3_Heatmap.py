import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('D:/5800/Project/final_data_test.csv')

# Calculate the correlation matrix
corr = df.corr()

# Set up the matplotlib figure with a larger size for better readability
plt.figure(figsize=(9, 7))

# Draw the heatmap
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'shrink': .5})

plt.title('Heatmap of Correlations Between Factors')
plt.xticks(rotation=30)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


