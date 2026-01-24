import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

master_table = pd.read_csv('master_macro_table.csv')

# Gets messed up if it picks up date, only need the second column
corr_matrix = master_table.select_dtypes(include='number').corr()

print(corr_matrix)


plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Macroeconomic Variables')
plt.show()
