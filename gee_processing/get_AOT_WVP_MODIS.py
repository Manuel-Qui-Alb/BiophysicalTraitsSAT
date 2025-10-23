# pip install earthengine-api
import ee
import pandas as pd
import geopandas as gpd
from functions import utc_to_local_solar_time

ee.Initialize(project='saw-ucdavis')

farm = 'RIP720'
satellite = 'L08'
crs = 'EPSG:32610'

dict_info = {'BLS':
               {'vector': ee.FeatureCollection('projects/saw-ucdavis/assets/BLS'),
                'first_dates': ['2024-01-01', '2024-07-01','2025-01-01','2025-07-01'],
                'last_dates': ['2024-07-01', '2025-01-01','2025-07-01','2025-12-01']},
           'RIP720':
               {'vector': ee.FeatureCollection('projects/saw-ucdavis/assets/RIP_720'),
                'first_dates': ['2018-01-01', '2018-07-01', '2019-01-01', '2019-07-01', '2020-01-01', '2020-07-01'],
                'last_dates': ['2018-07-01', '2019-01-01', '2019-07-01', '2020-01-01', '2020-07-01', '2021-01-01']}}

def run_process(farm):
    vector = dict_info[farm]['vector']
    first_dates = dict_info[farm]['first_dates']
    last_dates = dict_info[farm]['last_dates']

    # --- per-image stats (SERVER-SIDE) ---
    def add_stats(img):
        # scale to physical units
        aot = img.select('Optical_Depth_055').multiply(0.001)  # unitless @ 550 nm
        wv = img.select('Column_WV').multiply(0.001)  # cm (== g cm^-2)

        # optional light QA/range masks
        aot = aot.updateMask(aot.gte(0).And(aot.lte(5)))
        wv = wv.updateMask(wv.gte(0).And(wv.lte(10)))

        stack = aot.rename('AOT550').addBands(wv.rename('PWV_cm'))

        stats = stack.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=geometry,
            scale=1000,  # MAIAC native resolution
            bestEffort=True,
            tileScale=4,
            maxPixels=1e13
        )

        # attach stats + a readable date to the image (no getInfo here!)
        return (img.set(stats)
                .set({'system:time_start': img.get('system:time_start')}))

        # --- turn into a feature table so you can download/inspect ---
    def to_feature(img):
        props = img.toDictionary(['system:time_start',
                                      'AOT550_mean','AOT550_stdDev',
                                      'PWV_cm_mean','PWV_cm_stdDev'])
        return ee.Feature(None, props)


    atm_data_list = []
    for x in range(0, len(first_dates)):
        first_date = first_dates[x]
        last_date = last_dates[x]
        print(first_date)
        output = rf'C:\Users\mqalborn\Desktop\ET_3SEB\results\MCD19_MODIS/MCD19_{farm}_{first_date.replace("-", "")}_{last_date.replace("-", "")}.csv'

        centroid = vector.geometry().centroid().getInfo()['coordinates']
        lon = centroid[0]
        lat = centroid[1]
        geometry = vector.geometry()

        # --- 3) Pull MAIAC granules near that time, over your AOI ---
        MCD19 = (ee.ImageCollection('MODIS/061/MCD19A2_GRANULES')
                 .filterBounds(geometry)
                 .filterDate(first_date, last_date))



        imgs_with_stats = MCD19.map(add_stats)

        # keep only images that actually have data over your orchard
        imgs_with_stats = imgs_with_stats.filter(
            ee.Filter.notNull(['PWV_cm_mean', 'AOT550_mean'])
        )

        table = ee.FeatureCollection(imgs_with_stats.map(to_feature))

        # OPTION A) Quick look in Python (small results)
        rows = table.getInfo()
        data_stats = gpd.GeoDataFrame.from_features(rows)

        data_stats = data_stats.rename(columns={'AOT550_mean': 'AOT', 'PWV_cm_mean': 'WVP'})
        data_stats['overpass_solar_time'] = data_stats['system:time_start'].map(
            lambda x: utc_to_local_solar_time(x, longitude_deg=lon)
        )
        data_stats = data_stats[['overpass_solar_time', 'AOT', 'WVP']]
        data_stats.to_csv(output)
        atm_data_list.append(data_stats)

    atm_data = pd.concat(atm_data_list)
    atm_data.loc[:, 'date'] = atm_data.overpass_solar_time.dt.date
    atm_data = atm_data.rename(columns={'overpass_solar_time': 'overpass_solar_time_MOD'})
    atm_data = atm_data[atm_data.overpass_solar_time_MOD.dt.hour < 12]

    path = rf'C:\Users\mqalborn\Desktop\ET_3SEB\satellite\{farm}\{satellite}\RHO/METADATA.csv'
    metadata = pd.read_csv(path)
    metadata = metadata[['overpass_solar_time', 'SAA', 'SZA', 'VZA']]
    metadata.loc[:, 'date'] = pd.to_datetime(metadata.overpass_solar_time).dt.date
    metadata_tot = pd.merge(metadata, atm_data, on='date', how='left')
    metadata_tot = metadata_tot.interpolate()
    metadata_tot.to_csv(path)
