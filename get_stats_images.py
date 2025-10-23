from functions.raster_utils import L08ImageCollection, S02ImageCollection, ImageCollection


dir_dataset = rf'C:/Users/mqalborn/Desktop/ET_3SEB/'
farm = ['BLS', 'RIP720']

"""
satellite = 'L08'
band_stats = ["B01", "B02", "B03", "B04", "B05", "B06", "B07"]
for x in farm:
    folder = dir_dataset + rf'satellite/{x}/{satellite}/RHO/'
    vector = dir_dataset + rf"vectors\{x}_blocks\{x}_blocks.shp"
    outpath = rf'C:/Users\mqalborn\Desktop\ET_3SEB/results/spectral_signature/SS_{satellite}_{x}.csv'

    Col = L08ImageCollection(folder=folder, vector=vector, farm=x)
    Col.load_collection()
    pd_stats = Col.reduce_regions(band=band_stats, outpath=outpath)

satellite = 'S2'
band = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
for x in farm:
    folder = dir_dataset + rf'satellite/{x}/{satellite}/RHO/'
    vector = dir_dataset + rf"vectors\{x}_blocks\{x}_blocks.shp"
    outpath = rf'C:/Users\mqalborn\Desktop\ET_3SEB/results/spectral_signature/SS_{satellite}_{x}.csv'

    Col = S02ImageCollection(folder=folder, vector=vector, farm=x)
    Col.load_collection()
    pd_stats = Col.reduce_regions(band=band, outpath=outpath)
"""

satellite = 'S2'
band = ["LAI"]
product = 'PROSAIL'
for x in farm:
    folder = dir_dataset + rf'satellite/{x}/{satellite}/PROSAIL/'
    vector = dir_dataset + rf"vectors\{x}_blocks\{x}_blocks.shp"
    outpath = rf'C:/Users\mqalborn\Desktop\ET_3SEB/results/LAI/LAI_{satellite}_{product}_{x}.csv'

    Col = ImageCollection(folder=folder, vector=vector, farm=x, satellite=satellite, band=band)
    Col.load_collection()
    pd_stats = Col.reduce_regions(band=band, outpath=outpath)


satellite = 'L08'
product = 'PROSAIL'
band_image = ['Cab', 'Car', 'Cm', 'Cw', 'Ant', 'Cbrown', 'LAI', 'leaf_angle']
band_stats = ['LAI']
for x in farm:
    folder = dir_dataset + rf'satellite/{x}/{satellite}/{product}/'
    vector = dir_dataset + rf"vectors\{x}_blocks\{x}_blocks.shp"
    outpath = rf'C:/Users\mqalborn\Desktop\ET_3SEB/results/LAI/LAI_{satellite}_{product}_{x}.csv'

    Col = ImageCollection(folder=folder, vector=vector, farm=x, satellite=satellite, band=band_image)
    Col.load_collection()
    pd_stats = Col.reduce_regions(band=band_stats, outpath=outpath)
