import ee
import geemap
import pandas as pd
import gee.gee_objects as gee
import datetime as dt
import math
from functions import utc_to_local_solar_time
import numpy as np
import glob
import get_AOT_WVP_MODIS


farm = 'RIP720'
crs = 'EPSG:32610'

vector_dic = {
    'RIP720': {
        'vector': ee.FeatureCollection('projects/saw-ucdavis/assets/RIP_720'),
        'first_date': '2018-01-01',
        'last_date': '2020-12-31'
    },
    'BLS': {
        'vector': ee.FeatureCollection('projects/saw-ucdavis/assets/BLS'),
        'first_date': '2024-01-01',
        'last_date': '2025-12-31'}
}

vector = vector_dic[farm]['vector']
first_date = vector_dic[farm]['first_date']
last_date = vector_dic[farm]['last_date']

out_images = rf"C:\Users\mqalborn\Desktop\ET_3SEB\satellite\{farm}/L08/RHO"
out_metadata = rf"C:\Users\mqalborn\Desktop\ET_3SEB\satellite\{farm}/L08/RHO/METADATA.csv"

centroid = vector.geometry().centroid().getInfo()['coordinates']
lon = centroid[0]
lat = centroid[1]

# Load ImageCollection from Google Earth Engine
LX = (gee.Landsat(farm=farm, aoi=vector)
      .filter_date(first_date, last_date))

LX.percentage_pixel_free_clouds(band='SR_B4', scale=30, crs='EPSG:4326')

LX.filter_by_feature(filter='gte',
                     name='percentage_pixel_free_clouds',
                     value=99)
SAA = LX.gee_image_collection.aggregate_array('SUN_AZIMUTH').getInfo()
SEA = LX.gee_image_collection.aggregate_array('SUN_ELEVATION').getInfo()
# SZA = 90 - SEA
VZA = LX.gee_image_collection.aggregate_array('ROLL_ANGLE').getInfo()
overpass_time_utc = LX.gee_image_collection.aggregate_array('system:time_start').getInfo()

overpass_solar_time = list(map(utc_to_local_solar_time, overpass_time_utc, np.repeat(lon, len(overpass_time_utc))))

pd_metadata = pd.DataFrame({'overpass_solar_time': overpass_solar_time,
                            'SAA': SAA,
                            'SZA': 90 - np.array(SEA),
                            'VZA': 0})

pd_metadata.to_csv(out_metadata)


# data_stats['overpass_solar_time'] = data_stats['system:time_start'].map(
#     lambda x: utc_to_local_solar_time(x, longitude_deg=lon)
# )
# utc_to_local_solar_time
# SUN_ZENITH = 90 - SUN_ELEVATION
# ZENITH_ANGLE

geemap.ee_export_image_collection(
    LX.gee_image_collection.select('SR.*'),
    scale=30,
    region=LX.aoi.geometry(),
    out_dir=out_images,
    crs=crs)
#

print('Processing AOT and WVP data from MCD19A2')
get_AOT_WVP_MODIS.run_process(farm)

