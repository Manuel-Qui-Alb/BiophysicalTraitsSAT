
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
import glob
import os
from functions.biophysical import get_diffuse_radiation_6S, build_soil_database, SRF_LIBRARY, S2_BANDS
from pypro4sail import machine_learning_regression as inv
from sklearn.ensemble import RandomForestRegressor
import sys

original_stdout = sys.stdout
# Simulate Sentinel-2 spectra
### bands to use in generating LUT and inversion
### Satellite options: Landsat5, Landsat7, Landsat8, Landsat9, PRISMA
#                      Sentinel2A, Sentinel2B, Sentinel2C, Sentinel3A_SYN, SentinelBA_SYN
S2_BANDS_model = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
### Stack spectral bands
srf = []
srf_file = SRF_LIBRARY / f'Sentinel2A.txt'
srfs = np.genfromtxt(srf_file, dtype=None, names=True)
for band in S2_BANDS_model:
    srf.append(srfs[band])

# open as pandas dataframe
srf_df = pd.read_csv(srf_file, sep = '\t')

band_names = srf_df[S2_BANDS_model].columns

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
MAX_LAI = 3 # from S2 L2B ATBD
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

# Define new band names in the same order
path_dir = rf'C:\Users\mqalborn\Desktop\ET_3SEB\PISTACHIO\Sentinel2/METADATA.csv'
metadata = pd.read_csv(path_dir)

metadata.loc[:, 'date'] =pd.to_datetime(metadata.overpass_solar_time).dt.date

# spectral range
wls_sim = np.arange(400, 2501)
njobs = 8

def main(path, out_dir):
    image = rioxarray.open_rasterio(path)  # dims: (band, y, x)
    # extract date from filename (example: "image_20210101.tif")
    date_str = os.path.basename(path).split("_")[0].split("T")[0]
    date = pd.to_datetime(date_str, format="%Y%m%d").date()

    image = image.assign_coords(band=["B01", "B02", "B03", "B04", "B05", "B06",
                                      "B07", "B08", "B8A", "B09", "B11", "B12"])
    BandsInputs = image.sel(band=S2_BANDS_model) * 0.0001

    metadata_image = metadata[metadata.date == date]

    date = pd.to_datetime(metadata_image.overpass_solar_time.values)
    SAA = float(metadata_image.MEAN_SOLAR_AZIMUTH_ANGLE)
    SZA = float(metadata_image.MEAN_SOLAR_ZENITH_ANGLE)
    VZA = float(metadata_image.MEAN_INCIDENCE_ZENITH_ANGLE_B8)
    AOT = float(metadata_image.AOT)
    WVP = float(metadata_image.WVP) # precipitable water vapour, The total amount of water in a vertical path through the atmosphere (in g/cm^2 = cm)

    # import INSIDE the guard so the package isn’t imported at top level
    from pypro4sail import machine_learning_regression as inv  # replace with real module

    print(f"Running 6S for estimation of diffuse/direct irradiance")
    skyl = get_diffuse_radiation_6S(AOT, WVP, SZA, SAA, date, altitude=0.1)
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
            outfile=None,
            calc_FAPAR=False,
            reduce_4sail=True)

    # RF paramters
    scikit_regressor_opts = {"n_estimators": 100,
                             "min_samples_leaf": 1,
                             "n_jobs": -1}

    input_scalers = {}
    output_scalers = {}
    regs = {}
    fis = {}

    for i, param in enumerate(OBJ_PARAM_NAMES):
        reg, input_gauss_scaler, output_gauss_scaler, _ = \
            inv.train_reg(rho_canopy_vec,
                          params[param].reshape(-1, 1),
                          scaling_input='normalize',
                          scaling_output='normalize',
                          regressor_opts=scikit_regressor_opts,
                          reg_method="random_forest")

        input_scalers[param] = input_gauss_scaler
        output_scalers[param] = output_gauss_scaler
        regs[param] = reg
        fis[param] = reg.feature_importances_
    print("\nProcessing time (Training):")
    # print(f"\t{len(OBJ_PARAM_NAMES)} Random forests: {end_time_standard}")

    print('Done!')

    fis = pd.DataFrame(fis)
    fis.loc[:, 'bands'] = S2_BANDS_model

    print(date[0])
    fis.loc[:, 'date'] = date[0]

    arr = BandsInputs.values
    y_coords = BandsInputs.y
    x_coords = BandsInputs.x

    # print(type(BandsInputs))
    bio_dict = {}
    dataarrays = []
    for i, param in enumerate(OBJ_PARAM_NAMES):
        # output = np.full(arr.size, np.nan)

        print(f"Appliying {param} model to S2 image reflectance array")
        if np.any(arr):
            print(arr.shape)
            arr_reshape = arr.reshape((arr.shape[0], -1)).T
            arr_scaled = input_scalers[param].transform(arr_reshape)
            arr_scaled_pred = regs[param].predict(arr_scaled)
            out = output_scalers[param].inverse_transform(arr_scaled_pred.reshape(-1, 1)).reshape(-1)

        output = out.reshape(arr[0,:,:].shape)
        pred_xarray = xr.DataArray(
            output,
            dims=("y", "x"),
            coords={"y": y_coords, "x": x_coords},
            name=param
        )
        pred_xarray = pred_xarray.assign_coords(band=param)
        dataarrays.append(pred_xarray)

        # bio_dict[param] = output
    col = xr.concat(dataarrays, dim="band")
    col = col.sel(band='LAI')
    col = col.rio.write_crs("EPSG:32630")

    # out_file = out_dir + 'LAI_' + os.path.basename(path)
    # col.rio.to_raster(out_file, dtype="float32", nodata=np.nan)
    return fis

if __name__ == "__main__":
    path_dir = rf'C:\Users\mqalborn\Desktop\ET_3SEB\PISTACHIO\Sentinel2\march_2025'
    out_dir = rf"C:\Users\mqalborn\Desktop\ET_3SEB\PISTACHIO\PRO4SAIL/test4/"

    files = sorted(glob.glob(os.path.join(path_dir, "*.tif")))
    fis = main(files[0], out_dir)

    from matplotlib import pyplot as plt
    import seaborn as sns

    fis = [main(x, out_dir) for x in files]
    fis = pd.concat(fis)
    sns.barplot(x="bands", y="LAI", data=fis)
    plt.show()
    # for x in files:
    #     print(x)
    #     main(x, out_dir)



# Approach 1: Train individual random forest model for each variable
print(f"Training {len(OBJ_PARAM_NAMES)} Random forests for "
      f"{','.join(OBJ_PARAM_NAMES)}")

















