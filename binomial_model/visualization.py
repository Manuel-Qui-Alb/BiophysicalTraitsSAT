import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

data = pd.read_excel('files/binomial_CN_sensitivity_analysis.xlsx', sheet_name='inputs')
# print(data.head())

# data = data[data.lai<5]
sns.scatterplot(data, x='Sn_V_bin', y='Sn_V_CN', hue='lai')
plt.plot([0, 1000], [0, 1000])
plt.show()
sns.scatterplot(data, x='Sn_S_bin', y='Sn_S_CN', hue='lai')
plt.plot([0, 1000], [0, 1000])
plt.show()