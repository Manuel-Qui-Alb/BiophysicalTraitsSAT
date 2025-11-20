import numpy as np
import pandas as pd
import os
import glob
import seaborn as sns
from matplotlib import pyplot as plt

dir = rf'C:\Users\mqalborn\Desktop\ET_3SEB\LAI\BLS\LAI\2025/*.TXT'
paths = glob.glob(dir)

docker = []
for x in paths:
    data = pd.read_csv(x, sep="\t", header=None, on_bad_lines="skip")
    data = data[data[0].isin(['LAI_FILE', 'GPSLAT', 'GPSLONG','DATE', 'LAI'])]
    data = data.set_index(data[0])
    data = data.T.drop(0)
    docker.append(data)

data_all = pd.concat(docker)
# data_all.to_csv('C:/Users/mqalborn/Desktop/PISTACHIO/LAI/2025/resume.csv')

data_all.loc[:, 'DATE'] = pd.to_datetime(data_all.DATE).dt.date
data_all = data_all.astype({'LAI': 'float'})
data_all.loc[:, 'code'] = data_all.LAI_FILE.str[3:5]    
data_all = data_all[data_all.LAI>0]
data_all.loc[:, 'block'] = data_all.LAI_FILE.str[0:2]

con = data_all.code.isin(['AC', 'A5'])
data_all.loc[con, 'source'] = 'LAI_ov'
con = data_all.code.isin(['B5', 'UC', '01', '10', '35'])
data_all.loc[con, 'source'] = 'LAI_un'

data_all = data_all[['block', 'DATE', 'GPSLAT', 'GPSLONG', 'LAI', 'source']]
data_all_pivot = pd.pivot_table(
    data=data_all,
    values='LAI',
    columns='source',
    index=['DATE', 'GPSLAT', 'GPSLONG', 'block'],
    aggfunc=np.mean)

# data_all_pivot.to_csv('C:/Users/mqalborn/Desktop/PISTACHIO/LAI/2025/resume_v2.csv')

sns.scatterplot(data=data_all, x='DATE', y='LAI', hue='source')
plt.show()

