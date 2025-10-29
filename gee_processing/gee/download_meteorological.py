import ee
# import geemap
# import pandas as pd
# import gee.gee_objects as gee
# import datetime as dt
# import math
# from functions import utc_to_local_solar_time
import numpy as np
import glob
# import get_AOT_WVP_MODIS
import datetime as dt
import geopandas as gpd

ee.Initialize(project='saw-ucdavis')

def e_T(T):
    """
    Calculates Saturation Vapor Pressure Enviromental biophysics (page 42).

    Parameters
    ----------
    T : float
        air temperature in Kelvin

    Returns
    ---------
    es : Saturation vapor pressure [kPa]
    """
    T = T- 273.15
    A = 0.611
    B = 17.502
    C = 240.97
    es = A * np.exp((B * T) / (T + C))
    return es


farm = 'RIP720'

vector_dic = {
    'RIP720': {
        'vector': ee.FeatureCollection('projects/saw-ucdavis/assets/RIP_720'),
        'folder_id': 'projects/saw-ucdavis/assets/LAI_LANDSAT_RIP720',
        'PATH': 43,
        'ROW': 34,
        'first_date': '2019-05-01',
        'last_date': '2019-08-31',
        'out_dir': rf'C:\Users\mqalborn\Desktop\ET_3SEB\satellite\RIP720\L08/LAI_TRAD'
    },
    'BLS': {
        'vector': ee.FeatureCollection('projects/saw-ucdavis/assets/BLS'),
        'folder_id': 'projects/saw-ucdavis/assets/LAI_LANDSAT_BLS',
        'PATH': 44,
        'ROW': 33,
        'first_date': '2024-01-01',
        'last_date': '2025-12-31',
        'out_dir': rf'C:\Users\mqalborn\Desktop\ET_3SEB\satellite\BLS\L08/LAI_TRAD'}
}

first_date = vector_dic[farm]['first_date']
last_date = vector_dic[farm]['last_date']

vector = vector_dic[farm]['vector']
meteo_var = ['temperature_2m', 'dewpoint_temperature_2m', 'u_component_of_wind_10m',
             'surface_solar_radiation_downwards', 'surface_thermal_radiation_downwards',
             'v_component_of_wind_10m', 'surface_pressure']
ERA5col = ((ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
           .select(meteo_var)
           .filterDate(first_date, last_date)))
           # .filter(ee.Filter.calendarRange(10, 12, 'hour')))


def calculate_stats(img):
    reducer = ee.Reducer.mean()

    stats = img.select(meteo_var).reduceRegion(
        reducer=reducer,
        geometry=vector.geometry(),
        scale=1000,  # MAIAC native resolution
        bestEffort=True,
        tileScale=4,
        maxPixels=1e13,
        crs='EPSG:32610'
    )

    # attach stats + a readable date to the image (no getInfo here!)
    return img.set(stats)


def to_feature(img):
    props = img.toDictionary(['system:time_start'] + meteo_var)
    return ee.Feature(None, props)

# test = calculate_stats(ERA5col.first())

feature_stats = ERA5col.map(calculate_stats)
table = ee.FeatureCollection(feature_stats.map(to_feature))
rows = table.getInfo()
data_stats = gpd.GeoDataFrame.from_features(rows).drop('geometry', axis=1)
data_stats.rename(columns={'system:time_start': 'timestamp'}, inplace=True)
data_stats.loc[:, 'timestamp'] = [dt.datetime.fromtimestamp(x/1000)
                                  for x in data_stats['timestamp']]
data_stats.loc[:, 'wapor_pressure'] = data_stats.dewpoint_temperature_2m.map(e_T)
data_stats.loc[:, 'wind_speed_10m'] = np.sqrt(
    data_stats.u_component_of_wind_10m**2 +
    data_stats.v_component_of_wind_10m**2)

data_stats_11 = data_stats[(data_stats.timestamp.dt.hour == 11)]
data_stats_10 = data_stats[(data_stats.timestamp.dt.hour == 10)]

data_stats_11.loc[:, 'surface_solar_radiation_downwards'] = ((data_stats_11.surface_solar_radiation_downwards.values -
                                                             data_stats_10.surface_solar_radiation_downwards.values) /
                                                             3600)
data_stats_11.loc[:, 'surface_thermal_radiation_downwards'] = ((data_stats_11.surface_thermal_radiation_downwards.values -
                                                             data_stats_10.surface_thermal_radiation_downwards.values) /
                                                             3600)
data_stats_11.loc[:, 'surface_pressure'] = data_stats_11.surface_pressure * 0.01
data_stats_11.loc[:, 'wapor_pressure'] = data_stats_11.wapor_pressure * 10

data_stats_11 = data_stats_11[['timestamp', 'temperature_2m', 'wind_speed_10m', 'wapor_pressure',
                         'surface_pressure', 'surface_solar_radiation_downwards',
                         'surface_thermal_radiation_downwards']]

import seaborn as sns
from matplotlib import pyplot as plt

sns.lineplot(x=data_stats.timestamp.dt.hour, y=data_stats.temperature_2m)
plt.show()

data_stats_11.to_csv(rf'C:\Users\mqalborn\Desktop\ET_3SEB\results\ERA5/ERA5_{farm}.csv')
print(data_stats.columns)
print(data_stats.shape)
print(data_stats.head())


