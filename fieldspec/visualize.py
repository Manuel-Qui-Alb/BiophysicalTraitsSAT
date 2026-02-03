import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

path_output = rf'C:\Users\mqalborn\Box\DATA_CUBBIES\Manuel\BLS\FieldSpec/processed/rho_280126.csv'
data = pd.read_csv(path_output)
# data['sample'] = data['sample'].astype(str)
data.drop('Unnamed: 0', axis=1)
data = data.astype({'wl': int, 'rho': float, 'sample': str, 'date': int})

# data = data.sample(frac=1).reset_index(drop=True)
con = data['sample'].str.contains('SOIL')
data.loc[con, 'sample'] = 'SOIL'
data.loc[~con, 'sample'] = data[~con]['sample'].str[:3]

data = data[(data.wl <= 1300) | (data.wl >= 1500)] # first atmospheric window
data = data[(data.wl <= 1800) | (data.wl >= 2000)] # second atmospheric window
data = data[(data.wl <= 2400)]

data = data.groupby(['wl', 'sample', 'date']).agg(rho=('rho', 'mean')).reset_index()
sns.lineplot(data=data, x='wl', y='rho', hue='sample', palette='rainbow', markers='')
plt.legend(loc=1)
plt.show()
print(data.shape)