from functions.raster_utils import ImageCollection
import os

farm = ['BLS']
satellite = 'L08'
product = 'LAI_TRAD'
band_image = ['LAI', 'TRAD']
band_stats = ['LAI']

dir_dataset = os.path.normpath(rf'C:/Users/mqalborn/Desktop/ET_3SEB/')
vector = os.path.join(dir_dataset, 'LAI/BLS/LAI/measurement_location/LAI_2025.geojson')
output = os.path.join(dir_dataset, 'results/LAI_SATELLITE/LAI_L0X_MODIS_BLS_2025.csv')

for x in farm:
    folder = os.path.join(dir_dataset, rf'satellite/{x}/{satellite}/{product}/')

    # outpath = rf'C:/Users\mqalborn\Desktop\ET_3SEB/results/LAI/LAI_{satellite}_{product}_{x}.csv'

    Col = ImageCollection(folder=folder, vector=vector, farm=x, satellite=satellite, band=band_image)
    Col.load_collection()
    pd_stats = Col.reduce_regions(band=band_stats, outpath=output)
    # pd_stats.to_csv(output)
