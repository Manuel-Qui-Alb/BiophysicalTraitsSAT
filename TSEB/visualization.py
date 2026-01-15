import matplotlib
# matplotlib.use("Agg")

import pandas as pd
from matplotlib import pyplot as plt

import seaborn as sns

data = pd.read_csv(rf'files/outputs_sensitivity_analysis.csv')
# data_melt = data.melt(var_name="params",
#                  value_name="value")
# sns.boxplot(data=data_melt, x='params', y='value')
# plt.show()
#
# plt.hist(data.LE_V, density=True, bins=100)
sns.scatterplot(data=data, x='Rn_V', y='LE_V')
plt.show()
# import matplotlib.pyplot as plt

# plt.plot([1, 2, 3], [4, 5, 6])
# plt.show()
