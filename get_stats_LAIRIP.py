from functions.raster_utils import L08ImageCollection, S02ImageCollection, ImageCollection
import geopandas as gpd
import pandas as pd


farm = 'RIP720'
satellite = 'S2'
band = ["LAI"]
product = 'PROSAIL'

dir_dataset = rf'C:/Users/mqalborn/Desktop/ET_3SEB/'
folder = dir_dataset + rf'satellite/{farm}/{satellite}/PROSAIL/'
vector_path = rf'C:\Users\mqalborn\Desktop\ET_3SEB\vectors\RIP720_uavs/LAI2200_B.shp'

outpath = rf'C:/Users\mqalborn\Desktop\ET_3SEB/results/LAI/LAI2200_B_RIP720.csv'

Col = ImageCollection(folder=folder, vector=vector_path, farm=farm, satellite=satellite, band=band)
Col.load_collection()
pd_stats = Col.reduce_regions(band=band, outpath=outpath)
vector = gpd.read_file(vector_path)
vector = vector.reset_index().rename(columns={'index': 'id_vector'})

pd_stats = pd.merge(vector, pd_stats).drop(['geometry'], axis=1)
pd_stats.to_csv(outpath)




