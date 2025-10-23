import ee
import geemap
import pandas as pd
import gee.gee_objects as gee
from functions import utc_to_local_solar_time


def reduceRegion(image, band, vector, reducer='mean', scale=10, crs='EPSG:32610'):
    reducer = (image.select(band)
               .reduceRegion(reducer=reducer, geometry=vector, scale=scale, crs=crs)
               .set('system:time_start', image.get('system:time_start'))
               .set('MEAN_SOLAR_AZIMUTH_ANGLE', image.get('MEAN_SOLAR_AZIMUTH_ANGLE'))
               .set('MEAN_SOLAR_ZENITH_ANGLE', image.get('MEAN_SOLAR_ZENITH_ANGLE'))
               .set('MEAN_INCIDENCE_ZENITH_ANGLE_B8', image.get('MEAN_INCIDENCE_ZENITH_ANGLE_B8'))
               )
    reducer = reducer.getInfo()
    return reducer


### Define study site and dates
vector_BLS = ee.FeatureCollection('projects/saw-ucdavis/assets/BLS')
# vector_RIP720 = ee.FeatureCollection('projects/saw-ucdavis/assets/RIP_720')

# out_images = rf'C:\Users\mqalborn\Desktop\ET_3SEB\GRAPEX\Sentinel2'
out_images = rf'C:\Users\mqalborn\Desktop\ET_3SEB\PISTACHIO\Sentinel2'
# out_metadata = rf'C:\Users\mqalborn\Desktop\ET_3SEB\GRAPEX\Sentinel2/METADATA.csv'
out_metadata = rf'C:\Users\mqalborn\Desktop\ET_3SEB\PISTACHIO\Sentinel2/METADATA.csv'
vector = vector_BLS
farm = 'BLS'
first_date = '2024-01-01'
last_date = '2025-12-31'
crs = 'EPSG:32610'

centroid = vector.geometry().centroid().getInfo()['coordinates']
lon = centroid[0]
lat = centroid[1]

# Load ImageCollection from Google Earth Engine
S2 = (gee.Sentinel2(farm=farm, aoi=vector)
      .filter_date(first_date, last_date)
      .cloud_mask())

S2 = S2.clip()
if farm == 'RIP720':
    S2.gee_image_collection = S2.gee_image_collection.filter(ee.Filter.eq('MGRS_TILE', '10SGF'))

S2.percentage_pixel_free_clouds(band='B4', scale=10, crs=crs)
S2.filter_by_feature(filter='gte',
                     name='percentage_pixel_free_clouds',
                     value=99)

# Download images

geemap.ee_export_image_collection(
    S2.gee_image_collection.select('B.*'),
    scale=10,
    region=S2.aoi.geometry(),
    out_dir=out_images,
    crs=crs)

# Get AOT, WVP, azimuth, zenith and solar time
dates = S2.get_feature('date', unique=True)
data_stats = [reduceRegion(S2.get_image(i), band=['AOT', 'WVP'], vector=vector, scale=20, crs=crs)
                      for i in dates]
data_stats = pd.DataFrame(data_stats)
data_stats['overpass_solar_time'] = data_stats['system:time_start'].map(
    lambda x: utc_to_local_solar_time(x, longitude_deg=lon)
)
data_stats = data_stats[['overpass_solar_time', 'MEAN_SOLAR_AZIMUTH_ANGLE', 'MEAN_SOLAR_ZENITH_ANGLE', 'MEAN_INCIDENCE_ZENITH_ANGLE_B8', 'AOT', 'WVP']]
data_stats.WVP = data_stats.WVP * 0.001
data_stats.AOT = data_stats.AOT * 0.001
data_stats.to_csv(out_metadata, index=False)