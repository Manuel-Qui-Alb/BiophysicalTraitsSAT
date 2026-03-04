import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

data_CN = pd.read_excel('files/sensitivity_analysis_TSEB_CN.xlsx', sheet_name='Sn_V')
print(data_CN.head())
data_CN.sort_values(by=['ST'], inplace=True, ascending=False)

data_CN = data_CN[['params', 'S1', 'ST']].melt(id_vars=["params"], var_name="index", value_name="value")

sns.barplot(data=data_CN, y='params', x='value', hue='index')
plt.tight_layout()
plt.show()
