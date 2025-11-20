import os
import rioxarray
import pandas as pd
import xarray as xr
import geopandas as gpd
import glob
from functools import reduce


class ImageCollection:
    def __init__(self, folder, vector, farm, satellite, band):
        self.folder = folder
        self.paths = sorted(glob.glob(os.path.join(folder, "*.tif")))
        self.vector = vector
        self.farm = farm
        self.satellite = satellite
        self.band = band

    def load_collection(self):
        dataarrays = []
        # dates = []
        id_list = []

        for f in self.paths:
            # open raster
            da = rioxarray.open_rasterio(f)  # dims: (band, y, x)
            id = os.path.basename(f).split('.')[0]
            # try:
            #     extract date from filename (example: "image_20210101.tif")
                # date_str = os.path.basename(f).split("_")[0].split("T")[0]
                # date = pd.to_datetime(date_str, format="%Y%m%d").date()
            # except ValueError:
            #     date_str = os.path.basename(f).split("_")[1].split("T")[0]
            #     date = pd.to_datetime(date_str, format="%Y%m%d").date()

            # add new coordinate for date
            da = da.expand_dims(id=[id])

            # MGRS_TILE = os.path.basename(f).split("_")[2].split('.')[0]# dims: (date, band, y, x)
            # da = da.expand_dims(MGRS_TILE=[MGRS_TILE])

            dataarrays.append(da)
            # dates.append(date)
            id_list.append(id)
        col = xr.concat(dataarrays, dim="id")

        if self.band:
            col = col.assign_coords(band=self.band)

        self.images = col

    def reduce_regions(self, band, outpath=None):
        images = self.images
        vector = gpd.read_file(self.vector).to_crs(images.rio.crs)

        band_stats = [reduce_regions(col=images, band=XX, vector=vector)
                      .rename(columns={'mean': XX, 'count': '{}_count'.format(XX)})
                      for XX in band]

        band_stats = reduce(lambda left, right: pd.merge(left, right, on=['block', 'id']), band_stats)
        if outpath:
            band_stats.to_csv(outpath)
        return band_stats


class L08ImageCollection(ImageCollection):
    def __init__(self, folder, vector, farm, band=None):
        satellite = 'L08'
        if band is None:
            band = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B_AEROSOL"]
        super().__init__(folder, vector, farm, satellite, band)

class S02ImageCollection(ImageCollection):
    def __init__(self, folder, vector, farm, band=None):
        satellite = 'S02'
        if band is None:
            band = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
        super().__init__(folder, vector, farm, satellite, band)



def reduce_regions(col, band, vector):
    col = col.sel(band=band)
    results = []
    for i, row in vector.iterrows():
        geom = gpd.GeoDataFrame(geometry=[row.geometry], crs=vector.crs)
        clip_i = col.rio.clip(geom.geometry, geom.crs, drop=True, all_touched=False)
        clip_i = clip_i.where(clip_i != 9999)
        stats_i = {
            "block": row.block,
            "B": row.B,
            'id': clip_i["id"].values, # or row["your_id_field"]
            'id_vector': i,
            "mean": clip_i.mean(("y", "x"), skipna=True).values,
            "count": clip_i.count(("y", "x")).values,
        }
        results.append(stats_i)

    # Tidy as a long DataFrame
    outcome = (
        pd.concat(
            [pd.DataFrame(r).assign(block=r["block"]) for r in results],
            ignore_index=True
        )
    )
    return outcome


def normalized_diff(col, band1, band2, band_name):
    band1 = col.sel(band=band1)
    band2 = col.sel(band=band2)

    normdiff = (band1 - band2) / (band1 + band2)
    normdiff = normdiff.expand_dims(dim={"band": [band_name]})

    return normdiff