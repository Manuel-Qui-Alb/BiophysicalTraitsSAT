import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import glob

from uri_template import expand

# Satellite data
# folder_dir = rf'C:\Users\mqalborn\Desktop\ET_3SEB\results\LAI/'
# paths = glob.glob(folder_dir + '*.csv')\
path = rf'C:\Users\mqalborn\Desktop\ET_3SEB\results\LAI/LAI2200_B_RIP720.csv'
data_sat = pd.read_csv(path).drop('Unnamed: 0', axis=1)
# data_sat.loc[:, 'date'] = pd.to_datetime(data_sat.id.str[-8:])
data_sat.loc[:, 'date'] = pd.to_datetime(data_sat.id.str.split('_', expand=True)[1].str.split('T', expand=True)[0])
data_sat = data_sat[['date', 'block', 'transect', 'LAI']]
data_sat = data_sat.groupby(['date', 'block', 'transect']).agg(LAI_sat=('LAI', 'mean'),
                                                               LAI_sat_std=('LAI', 'mean')).reset_index()
data_sat.rename(columns={'date': 'date_sat'}, inplace=True)
data_sat = data_sat.astype({'block': str, 'transect': str})
data_sat.block = '72' + data_sat.block
# print(data_sat.columns)

# LAI 2200
path = rf'C:/Users\mqalborn\Desktop\ET_3SEB\LAI\RIP720/20190504_RIP_720_LAI_summary.csv'
data_lai2200 = pd.read_csv(path).drop('Unnamed: 0', axis=1)
data_lai2200.loc[data_lai2200.timestamp.isna(), 'timestamp'] = np.unique(data_lai2200.timestamp)[0]
data_lai2200.timestamp = data_lai2200.timestamp.astype('int').astype('str')
data_lai2200.loc[:, 'date'] = pd.to_datetime(data_lai2200.timestamp, format='%y%m%d')
# print(data_lai.columns)
data_lai2200 = data_lai2200[['date', 'measurement', 'block', 'transect', 'tree', 'LAI_ov', 'LAI_un']]
data_lai2200.loc[data_lai2200.LAI_un.isna(), 'LAI_un'] = 0
data_lai2200.loc[:, 'LAI_eco'] = data_lai2200.LAI_ov + data_lai2200.LAI_un

data_lai2200.rename(columns={'date': 'date_lai2200'}, inplace=True)
data_lai2200.block = '72' + data_lai2200.block.astype(str)
data_lai2200 = (data_lai2200.groupby(['block', 'transect', 'date_lai2200'])
                .agg(LAI_ov=('LAI_ov', 'mean'),
                     LAI_ov_std=('LAI_ov', 'std'),
                     LAI_eco=('LAI_eco', 'mean'),
                     LAI_eco_std=('LAI_eco', 'std')
                     ).reset_index())
data_lai2200 = data_lai2200.astype({'block': str, 'transect': str})

date_lai = np.unique(data_lai2200.date_lai2200)
deltaTime = np.timedelta64(1, 'W')
con = (data_sat.date_sat < date_lai[0] + deltaTime) & (data_sat.date_sat > date_lai[0] - deltaTime)
data_sat = data_sat[con]
data_lai = pd.merge(data_lai2200, data_sat, on=['block', 'transect'])
data_lai = data_lai[['date_lai2200', 'transect', 'date_sat', 'block', 'LAI_eco', 'LAI_eco_std',
                     'LAI_ov', 'LAI_ov_std', 'LAI_sat', 'LAI_sat_std']]
data_lai = data_lai[data_lai.date_sat == '2019-05-07']
con = data_lai.block.isin(['721', '722'])
data_lai.loc[con, 'Treatment'] = 'NCC'
data_lai.loc[-con, 'Treatment'] = 'CC'

data_lai = data_lai[-data_lai.transect.isin(['333', '433'])]

sns.set_context('notebook')
sns.set_palette(['#C9A77B', '#658A6E'])
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
g0 = sns.boxplot(data=data_lai, x='block', y='LAI_ov', hue='Treatment', ax=axs[0])
g0.set_ylabel('Ecosystemic LAI')

g1 = sns.boxplot(data=data_lai, x='block', y='LAI_sat', hue='Treatment', ax=axs[1])
g1.set_ylabel('S2 PROSAIL LAI')

g2 = sns.scatterplot(data=data_lai, x='LAI_sat', y='LAI_ov', hue='Treatment', ax=axs[2])
g2.set_ylabel('Ecosystemic LAI')
g2.set_xlabel('S2 PROSAIL LAI')

# [x.set(ylim=[1, 1.6]) for x in [g0, g1, g2]]
# g2.set(xlim=[1, 3])
plt.tight_layout()
plt.show()
# sns.relplot(data=data_lai, x='LAI_sat', y='LAI', hue='block')
# plt.show()

