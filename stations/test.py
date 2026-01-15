import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from pathlib import PurePath

# Define base directory
base_dir = rf'C:\Users\mqalborn\Box\DATA_CUBBIES\Manuel'
ndvi_folder = os.path.normpath(base_dir + '\BLS\stations/NDVI/*.dat')

ndvi_files = glob.glob(ndvi_folder)

block_code = {'7055': 'NCC', '12331': 'CC'}
# Assuming data is space-separated and has no header
def ndvi_table_preprocessing(path):
    df = pd.read_csv(path, sep=',', skiprows=1)
    site = PurePath(path).parts[-4]
    dl_code = os.path.basename(path).split('_')[0]
    df.insert(0, 'site', site, allow_duplicates=False)
    df.insert(1, 'block', block_code[dl_code], allow_duplicates=False)

    df = df.iloc[2:, :]
    df = df.astype({'LowWaveDn_TOTAL_Avg': float, 'LowWaveUp_TOTAL_Avg': float,
                    'LowWaveDn_UNDERCANOPY_Avg': float, 'LowWaveUp_UNDERCANOPY_Avg': float,
                    'LowWaveDn_CANOPY_Avg': float,
                    'HighWaveDn_TOTAL_Avg': float, 'HighWaveUp_TOTAL_Avg': float,
                    'HighWaveDn_UNDERCANOPY_Avg': float, 'HighWaveUp_UNDERCANOPY_Avg': float,
                    'HighWaveDn_CANOPY_Avg': float
                    })
    df.TIMESTAMP = pd.to_datetime(df.TIMESTAMP, format='%Y-%m-%d %H:%M:%S')
    df = df[df.TIMESTAMP>'2025-12-16 11:00:00']

    # reflectance calculation
    df.loc[:, 'LowWaveRef_TOTAL'] = df.LowWaveDn_TOTAL_Avg / df.LowWaveUp_TOTAL_Avg
    df.loc[:, 'LowWaveRef_UNDERCANOPY'] = df.LowWaveDn_UNDERCANOPY_Avg / df.LowWaveUp_UNDERCANOPY_Avg
    df.loc[:, 'LowWaveRef_CANOPY'] = df.LowWaveDn_CANOPY_Avg / df.LowWaveUp_TOTAL_Avg

    df.loc[:, 'HighWaveRef_TOTAL'] = df.HighWaveDn_TOTAL_Avg / df.HighWaveUp_TOTAL_Avg
    df.loc[:, 'HighWaveRef_UNDERCANOPY'] = df.HighWaveDn_UNDERCANOPY_Avg / df.HighWaveUp_UNDERCANOPY_Avg
    df.loc[:, 'HighWaveRef_CANOPY'] = df.HighWaveDn_CANOPY_Avg / df.HighWaveUp_TOTAL_Avg

    ndvi = lambda red, nir: (nir - red) / (nir + red)

    ndvi = [ndvi(df['LowWaveRef_{}'.format(x)], df['HighWaveRef_{}'.format(x)]) for x in ['TOTAL', 'UNDERCANOPY', 'CANOPY']]
    df[['NDVI_TOTAL', 'NDVI_UNDERCANOPY', 'NDVI_CANOPY']] = np.array(ndvi).T
    return df

df = list(map(ndvi_table_preprocessing, ndvi_files))
ndvi_data = pd.concat(df)

# fig, axes = plt.subplots(2, 1, figsize=(10, 10))
sns.lineplot(data=ndvi_data, x='TIMESTAMP', y='HighWaveUp_TOTAL_Avg', hue='block')
sns.lineplot(data=ndvi_data, x='TIMESTAMP', y='LowWaveUp_UNDERCANOPY_Avg', hue='block', linestyle='--')
plt.show()
# print(ndvi_data.columns)
# print(np.unique(ndvi_data.site))

#fig, axes = plt.subplots(3, 1, figsize=(10, 10))
# sns.lineplot(data=ndvi_data, x='TIMESTAMP', y='NDVI_TOTAL', hue='block', ax=axes[0])
# sns.lineplot(data=ndvi_data, x='TIMESTAMP', y='NDVI_CANOPY', hue='block', ax=axes[1])
# sns.lineplot(data=ndvi_data, x='TIMESTAMP', y='NDVI_UNDERCANOPY', hue='block', ax=axes[2])

# [ax.set_ylim([0.1, 0.8]) for ax in axes]
# plt.show()

# ndvi_TOTAL = ndvi(df.LowWaveRef_TOTAL, df.HighWaveRef_TOTAL)
# print(ndvi_TOTAL)