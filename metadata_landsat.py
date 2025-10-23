import pandas as pd
import glob
import os


MCD19_folder = rf'C:\Users\mqalborn\Desktop\ET_3SEB\results\MCD19_MODIS/'
MCD19_paths = glob.glob(MCD19_folder + '*.csv')
def read_csv(path):
    df = pd.read_csv(path)
    loc = os.path.basename(path).split('_')[1]
    df.loc[:, 'farm'] = loc
    return df

farm = 'BLS'
aot_wvp = pd.concat(map(read_csv, MCD19_paths)).drop('Unnamed: 0', axis=1)
aot_wvp.overpass_solar_time = pd.to_datetime(aot_wvp.overpass_solar_time)
aot_wvp = aot_wvp[aot_wvp.overpass_solar_time.dt.hour<12]
aot_wvp = aot_wvp[aot_wvp.farm == farm]
aot_wvp.loc[:, 'date'] = aot_wvp.overpass_solar_time.dt.date
aot_wvp.rename(columns={'overpass_solar_time':'overpass_solar_time_MOD'}, inplace=True)

out_metadata = rf"C:\Users\mqalborn\Desktop\ET_3SEB\satellite\{farm}/L08/RHO/METADATA.csv"
pd_metadata = pd.read_csv(out_metadata).drop('Unnamed: 0', axis=1)
pd_metadata.loc[:, 'date'] = pd.to_datetime(pd_metadata.overpass_solar_time).dt.date
pd_metadata = pd.merge(pd_metadata, aot_wvp, on='date', how='left').interpolate()
out_metadata = rf"C:\Users\mqalborn\Desktop\ET_3SEB\satellite\{farm}/L08/RHO/METADATA.csv"
pd_metadata.to_csv(out_metadata)
# print(aot_wvp.head())
print(pd_metadata.head())