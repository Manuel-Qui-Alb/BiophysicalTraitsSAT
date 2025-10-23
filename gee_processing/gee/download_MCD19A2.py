import ee
import geemap
from gee_objects import calculate_stats_geeimage, MCD19A2

ee.Initialize(project='saw-ucdavis')

vector_BLS = ee.FeatureCollection('projects/saw-ucdavis/assets/BLS')
# vector_RIP720 = ee.FeatureCollection('projects/saw-ucdavis/assets/RIP_720')

vector = vector_BLS

crs = 'EPSG:32610'
lambert = ee.Projection(crs)

# Define a function to reproject each feature.
def reproject_feature(feature):
    """Reprojects a single feature's geometry."""
    return feature.transform(lambert, 0.001)

vector = vector.map(reproject_feature)

first_date = '2025-05-01'
last_date = '2025-07-31'

MCD19A2col = (MCD19A2(farm='BLS', aoi=vector, crs=crs)
              .filter_date(first_date, last_date))
data_stats = MCD19A2col.stats_calculator(['Column_WV', 'Optical_Depth_047', 'Optical_Depth_055'],
                                         scale=1000, crs=crs, reducer='mean')
data_stats.to_csv('C:/Users/mqalborn/Desktop/ET_3SEB/PISTACHIO/results/MCD19A2.csv')
print(data_stats.head())

# out_images = rf'C:\Users\mqalborn\Desktop\ET_3SEB\PISTACHIO\MODIS/'


# def reduceRegion(image, band, vector, reducer='mean', crs='EPSG:32610'):
#     reducer = (image.select(band)
#                .reduceRegion(reducer=reducer, geometry=vector, crs=crs)
#                .set('system:time_start', image.get('system:time_start')))
#     reducer = reducer.getInfo()
#     return reducer
#
# MCD = (ee.ImageCollection('MODIS/061/MCD19A2_GRANULES')
#        .select(['Optical_Depth_055', 'Column_WV'])
#        .filterBounds(vector)
#        .filterDate(first_date, last_date))

# MCD.map(lambda img: calculate_stats_geeimage(img,
#                                              aoi=vector,
#                                              indexes=['Optical_Depth_055'],
#                                              scale=1000,
#                                              crs='EPSG:32610',
#                                              reducer='mean'))



# print(MCD.getInfo())