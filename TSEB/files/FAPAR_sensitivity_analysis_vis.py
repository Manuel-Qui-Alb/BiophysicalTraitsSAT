import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

data_iso = pd.read_csv('FAPAR_iso_sensitivity_analysis.csv')
data_row = pd.read_csv('FAPAR_row_sensitivity_analysis.csv')

def rename_columns(data):
    data = data[['params', 'S1', 'ST']]
    data = data.sort_values('ST', ascending=False)
    data = data.melt(id_vars=["params"], var_name="stats", value_name="Sobol Index")

    con = data.params == 'row_azimuth'
    data.loc[con, 'params'] = r'$\phi_{r}$'
    con = data.params == 'saa'
    data.loc[con, 'params'] = r'$\phi_{s}$'
    con = data.params == 'sza'
    data.loc[con, 'params'] = r'$\theta_{s}$'
    con = data.params == 'x_LAD'
    data.loc[con, 'params'] = r'X$_{E}$'
    con = data.params == 'fv'
    data.loc[con, 'params'] = r'f$_{V}$'
    con = data.params == 'h_V'
    data.loc[con, 'params'] = r'h$_{V}$'
    con = data.params == 'w_V'
    data.loc[con, 'params'] = r'w$_{V}$'
    return data

data_iso = rename_columns(data_iso)
data_row = rename_columns(data_row)

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False)
sns.catplot(data=data_iso, x="Sobol Index", y="params", hue="stats", kind="bar", ax=ax[0])
sns.catplot(data=data_row, x="Sobol Index", y="params", hue="stats", kind="bar", ax=ax[1])
# fig, ax = plt.subplots(2, 1, figsize=(10, 5))
# sns.barplot(data, x='params', y='S1', ax=ax[0])
plt.show()