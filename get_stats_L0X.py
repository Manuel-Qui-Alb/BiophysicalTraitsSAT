from functions.raster_utils import L08ImageCollection, S02ImageCollection, ImageCollection


dir_dataset = rf'C:/Users/mqalborn/Desktop/ET_3SEB/'

satellite = 'L08'
product = 'LAI_MODIS'
farm = ['RIP720']
band_image = ['LAI', 'TRAD']
band_stats = ['LAI', 'TRAD']

for x in farm:
    folder = dir_dataset + rf'satellite/{x}/{satellite}/{product}/'
    vector = dir_dataset + rf"vectors\{x}_blocks\{x}_blocks.shp"
    outpath = rf'C:/Users\mqalborn\Desktop\ET_3SEB/results/LAI_TRAD_{satellite}_{product}_{x}.csv'

    Col = ImageCollection(folder=folder, vector=vector, farm=x, satellite=satellite, band=band_image)
    Col.load_collection()
    pd_stats = Col.reduce_regions(band=band_stats, outpath=outpath)