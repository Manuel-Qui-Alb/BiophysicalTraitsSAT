import ee
import numpy as np
import geopandas as gpd
import pandas as pd

ee.Initialize(project='saw-ucdavis')


class ImageCollection:
    def __init__(self, farm, aoi, collection_id):
        self.available_dates = None
        self.first_image = None
        self.farm = farm
        self.aoi = aoi
        self.gee_image_collection = (ee.ImageCollection(collection_id)
                                     .filterBounds(aoi)
                                     .map(lambda img: img.set('timestamp', img.date().format('YYYY-MM-dd HH:mm:ss')))
                                     .map(lambda img: img.set('date', img.date().format('YYYY-MM-dd'))))

    def clip(self):
        self.gee_image_collection = self.gee_image_collection.map(lambda img: img.clip(self.aoi)
                                                                  .copyProperties(img,
                                                                                  ['system:time_start', 'timestamp',
                                                                                   'date']))
        return self

    def count_pixels_tile(self, band='B4', scale=10, crs='EPSG:4326'):
        aoi = self.aoi
        col = self.gee_image_collection

        col = col.map(
            lambda image: image.set('count_pixels',
                                    (ee.Number(image.select(band)
                                               .reduceRegion(
                                        reducer=ee.Reducer.count(),
                                        geometry=aoi,
                                        scale=scale,
                                        crs=crs,
                                        tileScale=4,
                                        bestEffort=True).get(band))
                                     )
                                    )
        )

        self.gee_image_collection = col

    def count_pixels_features(self, index='B4', scale=10, crs='EPSG:4326'):
        dates = self.get_feature('date', unique=True)
        # calculate_stats_geeimage(image, aoi, indexes, scale, crs, reducer)
        data_stats = [calculate_stats_geeimage(self.get_image(i), self.aoi, index, scale, crs, reducer='count')
                      for i in dates]
        data_stats = pd.concat(data_stats)

        return data_stats

    def filter_date(self, first_date, last_date):
        self.gee_image_collection = self.gee_image_collection.filterDate(first_date, last_date)
        return self

    def filter_by_feature(self, filter, name, value):
        if filter == 'gt':
            self.gee_image_collection = self.gee_image_collection.filter(ee.Filter.gt(name, value))
        elif filter == 'gte':
            self.gee_image_collection = self.gee_image_collection.filter(ee.Filter.gte(name, value))
        elif filter == 'lt':
            self.gee_image_collection = self.gee_image_collection.filter(ee.Filter.lt(name, value))
        elif filter == 'lte':
            self.gee_image_collection = self.gee_image_collection.filter(ee.Filter.lte(name, value))

    def get_feature(self, feature, unique=True):
        output = self.gee_image_collection.aggregate_array(feature).getInfo()
        if unique:
            output = np.unique(output)
        return np.array(output)

    def get_first_image(self):
        first_image = self.gee_image_collection.first()
        return first_image

    def get_image(self, date):
        image = (self.gee_image_collection
                 .filter(ee.Filter.eq('date', date)))

        image = (image.first())
        return image

    # 32630 españa, #32631 cataluña
    def percentage_pixel_free_clouds(self, band='B4', scale=10, crs='EPSG:4326'):
        aoi = self.aoi
        col = self.gee_image_collection
        ideal_image = ee.Image(1).reproject(
            crs=crs,
            scale=scale
        )

        ideal_pixel = ee.Number(ideal_image.reduceRegion(
            geometry=aoi,
            reducer=ee.Reducer.count(),
            scale=scale,
            crs=crs,
            tileScale=4,
            bestEffort=True).get('constant'))

        col = col.map(
            lambda image: image.set('percentage_pixel_free_clouds',
                                    (ee.Number(image.select(band)
                                               .reduceRegion(
                                        reducer=ee.Reducer.count(),
                                        geometry=aoi,
                                        scale=scale,
                                        crs=crs,
                                        tileScale=4, maxPixels=1000000000,
                                        bestEffort=True).get(band)).divide(ideal_pixel).multiply(100).round()
                                     )
                                    )
        )

        self.gee_image_collection = col

    def stats_calculator(self, indexes, scale=10, crs='EPSG:4326', reducer=None):
        dates = self.get_feature('date', unique=True)

        data_stats = [calculate_stats_geeimage(self.get_image(i), self.aoi, indexes, scale, crs, reducer)
                      for i in dates]
        data_stats = pd.concat(data_stats)

        return data_stats



class ERA5Daily(ImageCollection):
    def __init__(self, farm, aoi, collection_id='ECMWF/ERA5_LAND/DAILY_AGGR'):
        super().__init__(self, farm, aoi, collection_id)


class Sentinel2(ImageCollection):
    def __init__(self, farm, aoi, collection_id='COPERNICUS/S2_SR_HARMONIZED'):
        super().__init__(self, farm, aoi, collection_id)

    def cloud_mask(self):
        csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
        self.gee_image_collection = (self.gee_image_collection.linkCollection(csPlus, ['cs_cdf'])
                                     .map(lambda img: img.updateMask(img.select('cs_cdf').gte(0.6))))
        return self

    def get_spectral_index(self):
        self.gee_image_collection = self.gee_image_collection.map(index_s2)


def maskLXClouds(image):
    qa = image.select('QA_PIXEL')
    # Use bitwiseAnd for the QA_PIXEL band.
    cloudState = qa.bitwiseAnd(4).eq(0)  # Cloud bit (2), so 2^2 = 4
    cloudShadowState = qa.bitwiseAnd(8).eq(0)  # Cloud Shadow bit (3), so 2^3 = 8
    waterState = qa.bitwiseAnd(32).eq(0)  # Water bit (5), so 2^5 = 32

    # Return the masked image.
    return image.updateMask(cloudState.And(cloudShadowState).And(waterState))


class Landsat(ImageCollection):
    def __init__(self, farm, aoi):
        self.available_dates = None
        self.first_image = None
        self.farm = farm
        self.aoi = aoi

        # Get the Landsat 8 and Landsat 9 Collection 2, Level 2 collections.
        l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
        l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')

        # Combine the two collections using the `merge()` function.
        lx = l8.merge(l9)

        col = (lx.filterBounds(aoi)
               # .mosaic()
               .set('SENSOR_ID', 'OLI_TIRS')
               .map(lambda img: img.set('timestamp', img.date().format('YYYY-MM-dd HH:mm:ss')))
               .map(lambda img: img.set('date', img.date().format('YYYY-MM-dd'))))
        col = col.map(maskLXClouds)

        self.gee_image_collection = col


class MCD19A2(ImageCollection):
    def __init__(self, farm, aoi, collection_id='MODIS/061/MCD19A2_GRANULES'):
        super().__init__(self, farm, aoi, collection_id)


######################################################### FUNCIONES ####################################################
def calculate_stats_geeimage(image, aoi, indexes, scale, crs, reducer):
    reducer = (image.select(indexes).reduceRegions(
        collection=aoi,
        reducer=reducer,
        scale=scale,
        crs=crs)
               # .select(propertySelectors=['region', 'city', 'parcela', 'poligono', 'mean'] + indexes)
               .set('PRODUCT_ID', image.get('LANDSAT_PRODUCT_ID'))
               )

    reducer = reducer.getInfo()
    data = gpd.GeoDataFrame.from_features(reducer)
    # PRODUCT_ID = reducer['properties']['PRODUCT_ID']
    if len(indexes) == 1:
        data = data.rename(columns={'mean': indexes[0]})

    # data.loc[:, 'PRODUCT_ID'] = PRODUCT_ID

    return data


def index_s2(image):
    image_scale = image.select('B.*')#.divide(10000)

    # Normalized difference Vegetation Index
    ndvi = ee.Image(image_scale.expression('(NIR - Red) / (NIR + Red)', {
        'NIR': image_scale.select('B8'),
        'Red': image_scale.select('B4')})).rename('NDVI')

    # Soil Adjusted Vegetation index
    savi = ee.Image(image_scale.expression('((NIR - Red) / (NIR + Red + L)) * (1 + 0.5)', {
        'NIR': image_scale.select('B8'),
        'Red': image_scale.select('B4'),
        'L': 5000}
                                           )).rename('SAVI')

    # Enhanced Vegetation Index
    evi = ee.Image(image_scale.expression('2.5 * (NIR - Red) / ((NIR + C1 * Red - C2 * Blue)1 + 10000)', {
        'NIR': image_scale.select('B8'),
        'Red': image_scale.select('B4'),
        'C1': 6,
        'C2': 7.5,
        'Blue': image_scale.select('B2'),
        'L': 10000}
                                          )).rename('EVI')

    # Green Normalized Difference Vegetation
    gndvi = ee.Image(image_scale.expression('(NIR - green) / (NIR + green)', {
        'NIR': image_scale.select('B8'),
        'green': image_scale.select('B3')}
                                            )).rename('GNDVI')

    seli = ee.Image(image_scale.expression('(Red_Edge1 -Red_Edge4) / (Red_Edge1 + Red_Edge4)', {
        'Red_Edge1': image_scale.select('B8A'),
        'Red_Edge4': image_scale.select('B5')}
                                           )).rename('SELI')

    # Normalized Difference Moisture Index
    ndmi = ee.Image(image_scale.expression('(NIR - SWIR) / (NIR + SWIR)', {
        'NIR': image_scale.select('B8'),
        'SWIR': image_scale.select('B11')}
                                           )).rename('NDMI')

    # Normalizaed Difference Water Index
    ndwi = ee.Image(image_scale.expression('(GREEN - NIR) / (GREEN + NIR)', {
        'GREEN': image_scale.select('B3'),
        'NIR': image_scale.select('B8')})).rename('NDWI')

    # Green-Red Vegetation Index
    grvi = image_scale.expression('(GREEN - RED) / (GREEN + RED)', {
        'GREEN': image_scale.select('B3'),
        'RED': image_scale.select('B4')
    }).rename('GRVI')

    # normalized Difference Snow Index
    ndsi = ee.Image(image_scale.expression('(GREEN - SWIR1) / (GREEN + SWIR1)', {
        'GREEN': image_scale.select('B3'),
        'SWIR1': image_scale.select('B11')
    })).rename('NDSI')

    # Enhanced Bloom Index
    ebi = ee.Image(image_scale.expression('(RED + GREEN + BLUE) / ( (GREEN / BLUE) * (RED - BLUE + epsilon) )', {
        'BLUE': image_scale.select('B2'),
        'GREEN': image_scale.select('B3'),
        'RED': image_scale.select('B4'),
        'epsilon': 10000
    })).rename('EBI')

    return (image_scale.addBands([ndvi, savi, evi, gndvi, seli, ndmi, ndwi, grvi, ndsi, ebi])
            .copyProperties(image, ['PRODUCT_ID', 'system:time_start', 'timestamp', 'date',
                                    'percentage_pixel_free_clouds', 'count_pixels']))
