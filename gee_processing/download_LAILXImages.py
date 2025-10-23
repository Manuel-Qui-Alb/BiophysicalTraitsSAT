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
        'folder_id': 'projects/saw-ucdavis/assets/LAI_L0X_PATH44_ROW33',
        'first_date': '2018-01-01',
        'last_date': '2020-12-31'
    },
    'BLS': {
        'vector': ee.FeatureCollection('projects/saw-ucdavis/assets/BLS'),
        'folder_id': 'projects/saw-ucdavis/assets/LAI_L0X_PATH43_ROW34',
        'first_date': '2024-01-01',
        'last_date': '2025-12-31'}
}

vector = vector_dic[farm]['vector']
# first_date = vector_dic[farm]['first_date']
# last_date = vector_dic[farm]['last_date']
folder = vector_dic[farm]['folder_id']

out_images = rf"C:\Users\mqalborn\Desktop\ET_3SEB\satellite\{farm}/L08/LAI_MODIS"
# out_metadata = rf"C:\Users\mqalborn\Desktop\ET_3SEB\satellite\{farm}/L08/RHO/METADATA.csv"

centroid = vector.geometry().centroid().getInfo()['coordinates']
lon = centroid[0]
lat = centroid[1]

# Load ImageCollection from Google Earth Engine
images = ee.data.listAssets({'parent': folder})['assets']
collection_id = [ee.Image(img['id']) for img in images if img['type'] == 'IMAGE']
collection = ee.ImageCollection(collection_id)

# data_stats['overpass_solar_time'] = data_stats['system:time_start'].map(
#     lambda x: utc_to_local_solar_time(x, longitude_deg=lon)
# )
# utc_to_local_solar_time
# SUN_ZENITH = 90 - SUN_ELEVATION
# ZENITH_ANGLE

geemap.ee_export_image_collection(
    collection,
    scale=30,
    region=vector.geometry(),
    out_dir=out_images,
    crs=crs)
#

