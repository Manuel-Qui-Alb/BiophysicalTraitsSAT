import os
import pandas as pd
import glob
from matplotlib import pyplot as plt
import seaborn as sns

dir_file = os.path.normpath(rf'C:\Users\mqalborn\Desktop\ET_3SEB\LAI\BLS\LAI\2025\expanded/')

path_list = glob.glob(os.path.join(dir_file, '*.txt'))

df = [pd.read_csv(x, sep='\t') for x in path_list]
df = pd.concat(df)
df.loc[:, 'block'] = df.LAI_File.str[0:3]
df.loc[:, 'cover'] = df.LAI_File.str[3:5]

con = (df.cover == 'B5')
df.loc[con, 'cover'] = 'UC'
con = (df.cover == 'A5') | (df.cover == '01') | (df.cover == '10') | (df.cover == '35')
df.loc[con, 'cover'] = 'AC'

df = df.rename(columns={'LAI_File': 'id', 'Mean(RawTime)': 'timestamp', 'GpsLat': 'lat', 'GpsLong': 'lon'})

import numpy as np
df.loc[:, 'B'] = df.id.str.split('.', expand=True)[1]
df.loc[df.B.isna(), 'B'] = 0
df.B = df.B.astype(int)
df_B = pd.DataFrame({'B':np.arange(0, 21), 'B_vector': [0, 11, 12, 13, 14, 15,
                                                 21, 22, 23, 24, 25,
                                                 31, 32, 33, 34, 35,
                                                 41, 42, 43, 44, 45]})
df = pd.merge(df, df_B, on='B')
df.B = df.B_vector
df.drop('B_vector', axis=1, inplace=True)

df.timestamp = pd.to_datetime(df.timestamp)
# df.to_csv(rf'C:\Users\mqalborn\Desktop\ET_3SEB\LAI\BLS\LAI\2025\expanded/extended.csv', index=False)

import geopandas as gpd
vector = gpd.read_file(rf'C:\Users\mqalborn\Desktop\ET_3SEB\LAI\BLS\LAI\measurement_location/LAI_2025.geojson')
vector = pd.merge(vector, df, on=['block', 'B'])

vector.to_file(rf'C:\Users\mqalborn\Desktop\ET_3SEB\LAI\BLS\LAI\measurement_location/LAI_2025_data.geojson', driver='GeoJSON')

sns.scatterplot(x='timestamp', y='LAI', hue='cover', data=df)
plt.show()

# path = rf'C:\Users\mqalborn\Desktop\ET_3SEB\LAI\BLS\LAI\2025\expanded/AC210_AC827.txt'



