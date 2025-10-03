import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
import glob
import os
from functions.biophysical import get_diffuse_radiation_6S, build_soil_database, SRF_LIBRARY, S2_BANDS
from pypro4sail import machine_learning_regression as inv
from sklearn.ensemble import RandomForestRegressor

# Simulate Sentinel-2 spectra
### bands to use in generating LUT and inversion
### Satellite options: Landsat5, Landsat7, Landsat8, Landsat9, PRISMA
#                      Sentinel2A, Sentinel2B, Sentinel2C, Sentinel3A_SYN, SentinelBA_SYN
S2_BANDS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
### Stack spectral bands
srf = []
srf_file = SRF_LIBRARY / f'Sentinel2A.txt'
srfs = np.genfromtxt(srf_file, dtype=None, names=True)
for band in S2_BANDS:
    srf.append(srfs[band])

# open as pandas dataframe
srf_df = pd.read_csv(srf_file, sep = '\t')

band_names = srf_df[S2_BANDS].columns

# Forward RTM simulations to build LUT
n_simulations = 40000

# parameter names
OBJ_PARAM_NAMES = ["Cab", "Car", "Cm", "Cw", "Ant", "Cbrown",
                   "LAI", "leaf_angle"]
# parameter info
PARAM_PROPS = {"Cab": ["Chlorophyll a+b", r"$\mu g\,cm^{-2}$", 1],
               "Car": ["Carotenoids", r"$\mu g\,cm^{-2}$", 1],
               "Cm": ["Dry matter", r"$g\,cm^{-2}$", 3],
               "Cw": ["Water content", r"$g\,cm^{-2}$", 3],
               "Ant": ["Antocyanins", r"$\mu g\,cm^{-2}$", 1],
               "Cbrown": ["Brown pigments", r"$-$", 1],
               "LAI": ["Leaf Area Index", r"$m^{2}\,m^{-2}$", 2],
               "leaf_angle": ["Mean leaf inclination angle", r"º", 1]}

# specify range of variable values
## minimum
MIN_N_LEAF = 1.0  # From LOPEX + ANGERS average
MIN_CAB = 0.0  # From LOPEX + ANGERS average
MIN_CAR = 0.0  # From LOPEX + ANGERS average
MIN_CBROWN = 0.0  # from S2 L2B ATBD
MIN_CM = 0.0017  # From LOPEX + ANGERS average
MIN_CW = 0.000  # From LOPEX + ANGERS average
MIN_ANT = 0.0
MIN_LAI = 0.0
MIN_LEAF_ANGLE = 30.0  # from S2 L2B ATBD
MIN_HOTSPOT = 0.1  # from S2 L2B ATBD
MIN_BS = 0.50  # from S2 L2B ATBD

## maximum
MAX_N_LEAF = 3.0  # From LOPEX + ANGERS average
MAX_CAB = 100.0  # From LOPEX + ANGERS average
MAX_CAR = 30.0  # From LOPEX + ANGERS average
MAX_CBROWN = 2.00  # from S2 L2B ATBD
MAX_CM = 0.0331  # From LOPEX + ANGERS average
MAX_CW = 0.0525  # From LOPEX + ANGERS average
MAX_ANT = 40.0
MAX_LAI = 5  # from S2 L2B ATBD
MAX_LEAF_ANGLE = 80.0  # from S2 L2B ATBD
MAX_HOTSPOT = 0.5  # from S2 L2B ATBD
MAX_BS = 3.5  # from S2 L2B ATBD

prosail_bounds = {'N_leaf': (MIN_N_LEAF, MAX_N_LEAF),
                  'Cab': (MIN_CAB, MAX_CAB),
                  'Car': (MIN_CAR, MAX_CAR),
                  'Cbrown': (MIN_CBROWN, MAX_CBROWN),
                  'Cw': (MIN_CW, MAX_CW),
                  'Cm': (MIN_CM, MAX_CM),
                  'Ant': (MIN_ANT, MAX_ANT),
                  'LAI': (MIN_LAI, MAX_LAI),
                  'leaf_angle': (MIN_LEAF_ANGLE, MAX_LEAF_ANGLE),
                  'hotspot': (MIN_HOTSPOT, MAX_HOTSPOT),
                  'bs': (MIN_BS, MAX_BS)}
df_bounds = pd.DataFrame(prosail_bounds, index=['min', 'max'])

print(f'Setting up {n_simulations} simulations with inputs bounds:\n\n {df_bounds[OBJ_PARAM_NAMES]}')
params_orig = inv.build_prosail_database(n_simulations,
                                         param_bounds=prosail_bounds,
                                         distribution=inv.SALTELLI_DIST)

print(f"Building {np.size(params_orig['bs'])} PROSPECTD+4SAIL simulations")
soil_spectrum = build_soil_database(params_orig["bs"])
print('Done!')

path_dir = rf'C:\Users\mqalborn\Desktop\ET_3SEB\PISTACHIO\Sentinel2'
files = sorted(glob.glob(os.path.join(path_dir, "*.tif")))
f = files[0]

image = rioxarray.open_rasterio(f)  # dims: (band, y, x)
# extract date from filename (example: "image_20210101.tif")
date_str = os.path.basename(f).split("_")[0].split("T")[0]
date = pd.to_datetime(date_str, format="%Y%m%d").date()

# add new coordinate for date
image = image.expand_dims(date=[date])  # dims: (date, band, y, x)

# Define new band names in the same order
band_names = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9" ,"B11", "B12"]

# Assign these names to the 'band' coordinate
image = image.assign_coords(band=band_names)
image = image.sel(band=band_names) * 0.0001

path_dir = rf'C:\Users\mqalborn\Desktop\ET_3SEB\PISTACHIO\Sentinel2/METADATA.csv'
metadata = pd.read_csv(path_dir)

metadata.loc[:, 'date'] =pd.to_datetime(metadata.overpass_solar_time).dt.date
metadata_image = metadata[metadata.date == date]
# for col in metadata.columns:
#     image = image.assign_coords({col: ("date", metadata[col].values)})

date = pd.to_datetime(metadata_image.overpass_solar_time[0])
SAA = np.array(metadata_image.MEAN_SOLAR_AZIMUTH_ANGLE)
SZA = np.array(metadata_image.MEAN_SOLAR_ZENITH_ANGLE)
VZA = np.array(metadata_image.MEAN_INCIDENCE_ZENITH_ANGLE_B8)
AOT = np.array(metadata_image.AOT)
WVP = np.array(metadata_image.WVP)# precipitable water vapour, The total amount of water in a vertical path through the atmosphere (in g/cm^2 = cm)
print(f"Running 6S for estimation of diffuse/direct irradiance")

skyl = get_diffuse_radiation_6S(AOT, WVP, SZA, SAA, date, altitude=0.1)

BandsInputs = image.sel(band=['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])

# if you want to save the LUT generating you can can specify a directory for lut_outfile
lut_outfile = None

# spectral range
wls_sim = np.arange(400, 2501)
njobs = 8

# generate LUT
best_params_cv = {
        'bootstrap': False,
        'ccp_alpha': 1.5575316820286568e-05,
        'max_depth': 16,
        'max_features': 'sqrt',
        'max_samples': None,
        'min_samples_leaf': 12,
        'min_samples_split': 13, 'n_estimators': 1156
}

RANDOM_STATE = 42
N_JOBS = -1

# Option A: use the CV-best directly on the validation set
rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS, **best_params_cv)

def main():
    # import INSIDE the guard so the package isn’t imported at top level
    from pypro4sail import machine_learning_regression as inv  # replace with real module

    # print(njobs)
    # print(params_orig)
    # print(skyl)
    # print(SZA)
    rho_canopy_vec, params = inv.simulate_prosail_lut_parallel(
        n_jobs=njobs,
        input_dict=params_orig,
        wls_sim=wls_sim,  # Change to estimate with Landsat, Planet and UAVs
        rsoil_vec=soil_spectrum,
        skyl=skyl,
        sza=SZA,
        vza=VZA,  # View(sensor) Zenith Angle (degrees).
        psi=0,  # Relative Sensor-Sun Azimuth Angle (degrees).
        srf=srf,  # Change to estimate with Landsat, Planet and UAVs
        outfile=lut_outfile,
        calc_FAPAR=False,
        reduce_4sail=True)
    # lut = pd.DataFrame(params)
    # lut.loc[:, 'date'] = date
    return rho_canopy_vec, params


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()  # harmless on Windows; useful if you ever freeze to exe
    rho_canopy_vec, params = main()
    print(rho_canopy_vec)