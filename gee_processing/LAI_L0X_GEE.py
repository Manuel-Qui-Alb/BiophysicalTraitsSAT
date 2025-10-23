import ee
ee.Initialize(project='saw-ucdavis')
import geopandas as gpd
from functions import utc_to_local_solar_time
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


farm = 'BLS'
crs='EPSG:32610'
dict_info = {'RIP720':
                 {'vector': ee.FeatureCollection('projects/saw-ucdavis/assets/RIP_720_blocks'),
                  'folder_id': 'projects/saw-ucdavis/assets/LAI_L0X_PATH43_ROW34'},
             'BLS':
                 {'vector': ee.FeatureCollection('projects/saw-ucdavis/assets/BLS_blocks'),
                  'folder_id': 'projects/saw-ucdavis/assets/LAI_L0X_PATH44_ROW33'}
             }

folder = dict_info[farm]['folder_id']
vector = dict_info[farm]['vector']
images = ee.data.listAssets({'parent': folder})['assets']
centroid = vector.geometry().centroid().getInfo()['coordinates']
lon = centroid[0]
lat = centroid[1]

collection_id = [ee.Image(img['id']) for img in images if img['type'] == 'IMAGE']
collection = ee.ImageCollection(collection_id)

docker = []
for x in vector.aggregate_array('block').getInfo():
    def calculate_stats(img):
        reducer = ee.Reducer.mean().combine(
            ee.Reducer.stdDev(), sharedInputs=True).combine(
            ee.Reducer.count(), sharedInputs=True)

        stats = img.select('LAI').reduceRegion(
            reducer=reducer,
            geometry=vector.filter(ee.Filter.eq('block', x)).geometry(),
            scale=30,  # MAIAC native resolution
            bestEffort=True,
            tileScale=4,
            maxPixels=1e13,
            crs='EPSG:32610'
        )

        # attach stats + a readable date to the image (no getInfo here!)
        return img.set(stats)  # (img.set(stats).set({'system:time_start': img.get('system:time_start')}))


    def to_feature(img):
        props = img.toDictionary(['system:time_start', 'LAI_mean', 'LAI_stdDev', 'LAI_count'])
        return ee.Feature(None, props)

    feature_stats = collection.map(calculate_stats)
    table = ee.FeatureCollection(feature_stats.map(to_feature))

    # OPTION A) Quick look in Python (small results)
    rows = table.getInfo()
    data_stats = gpd.GeoDataFrame.from_features(rows)
    data_stats.loc[:, 'block'] = x

    docker.append(data_stats)

data = pd.concat(docker)
data.LAI_mean = data.LAI_mean * 0.01
data['solar_time'] = data['system:time_start'].map(
    lambda x: utc_to_local_solar_time(x, longitude_deg=lon)
)
data = data.reset_index(drop=True)
folder = rf'C:\Users\mqalborn\Desktop\ET_3SEB\results\LAI/LAI_L0X_MODIS_{farm}.csv'
data.to_csv(folder, index=False)

sns.lineplot(data=data, x='solar_time', y='LAI_mean', hue='block', marker='o')
plt.show()