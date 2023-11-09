import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('corr_data_11_8.csv')

print(df.head())

corr_matrix = df.corr()

print(corr_matrix)

plt.figure(figsize=(12, 12))

heatmap = sns.heatmap(corr_matrix, vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=14)

plt.show()
