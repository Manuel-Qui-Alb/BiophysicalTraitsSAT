import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
from matplotlib import pyplot as plt
from functions.biophysical import get_diffuse_radiation_6S, build_soil_database, SRF_LIBRARY, S2_BANDS
import os
import rioxarray
import xarray as xr
import glob


location = 'RIP720'
satellite = 'L08'

folder_images = rf'C:\Users\mqalborn\Desktop\ET_3SEB\satellite\{location}/{satellite}/RHO/'
out_dir = rf'C:\Users\mqalborn\Desktop\ET_3SEB\satellite\{location}/{satellite}/PROSAIL/'
metadata = pd.read_csv(folder_images + 'METADATA.csv')
metadata.loc[:, 'date'] =pd.to_datetime(metadata.overpass_solar_time).dt.date

srf_info = {'L08': {'SAT_LIBRARY' : f'Landsat8.txt',
                    'bands_names' : ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
                    'bands_coords': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'QA_AEROSOL'],
                    'scale':2.75e-05,
                    'offset':-0.2},
            'S2': {'SAT_LIBRARY' : f'Sentinel2A.txt',
                   'bands_names' : ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'],
                   'bands_coords': ["B01", "B02", "B03", "B04", "B05", "B06",
                                    "B07", "B08", "B8A", "B09", "B11", "B12"],
                   'scale': 0.0001,
                   'offset': 0}}

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
MIN_N_LEAF = 1.2  # From LOPEX + ANGERS average
MIN_CAB = 20.0  # From LOPEX + ANGERS average
MIN_CAR = 0.0  # From LOPEX + ANGERS average
MIN_CBROWN = 0.0  # from S2 L2B ATBD
MIN_CM = 0.003  # From LOPEX + ANGERS average
MIN_CW = 0.008  # From LOPEX + ANGERS average
MIN_ANT = 0.0
MIN_LAI = 0.0
MIN_LEAF_ANGLE = 30.0  # from S2 L2B ATBD
MIN_HOTSPOT = 0.1  # from S2 L2B ATBD
MIN_BS = 0.50  # from S2 L2B ATBD

## maximum
MAX_N_LEAF = 1.8  # From LOPEX + ANGERS average
MAX_CAB = 90.0  # From LOPEX + ANGERS average
MAX_CAR = 30.0  # From LOPEX + ANGERS average
MAX_CBROWN = 2.00  # from S2 L2B ATBD
MAX_CM = 0.0110  # From LOPEX + ANGERS average
MAX_CW = 0.02  # From LOPEX + ANGERS average
MAX_ANT = 40.0
MAX_LAI = 7  # from S2 L2B ATBD
MAX_LEAF_ANGLE = 80.0  # from S2 L2B ATBD
MAX_HOTSPOT = 0.5  # from S2 L2B ATBD
MAX_BS = 2  # from S2 L2B ATBD

MEAN_N_LEAF = 1.50  # From LOPEX + ANGERS average
MEAN_CAB = 45.00  # From LOPEX + ANGERS average
MEAN_CAR = 9.55  # From LOPEX + ANGERS average
MEAN_CBROWN = 0.0  # from S2 L2B ATBD
MEAN_CM = 0.005  # From LOPEX + ANGERS average
MEAN_CW = 0.012 # From LOPEX + ANGERS average
MEAN_ANT = 1.0
MEAN_LAI = 2.0  # from S2 L2B ATBD
MEAN_LEAF_ANGLE = 60.0  # from S2 L2B ATBD
MEAN_HOTSPOT = 0.2  # from S2 L2B ATBD
MEAN_BS = 1.2  # from S2 L2B ATBD

STD_N_LEAF = 0.30  # From LOPEX + ANGERS average
STD_CAB = 30.00  # From LOPEX + ANGERS average
STD_CAR = 4.69  # From LOPEX + ANGERS average
STD_CBROWN = 0.3  # from S2 L2B ATBD
STD_CM = 0.005  # From LOPEX + ANGERS average
STD_CW = 0.005  # From LOPEX + ANGERS average
STD_ANT = 10
STD_LAI = 3.0  # from S2 L2B ATBD
STD_LEAF_ANGLE = 30  # from S2 L2B ATBD
STD_HOTSPOT = 0.2  # from S2 L2B ATBD
STD_BS = 2.00  # from S2 L2B ATBD

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

prosail_moments = {'N_leaf': (MEAN_N_LEAF, STD_N_LEAF),
                   'Cab': (MEAN_CAB, STD_CAB),
                   'Car': (MEAN_CAR, STD_CAR),
                   'Cbrown': (MEAN_CBROWN, STD_CBROWN),
                   'Cw': (MEAN_CW, STD_CW),
                   'Cm': (MEAN_CM, STD_CM),
                   'Ant': (MEAN_ANT, STD_ANT),
                   'LAI': (MEAN_LAI, STD_LAI),
                   'leaf_angle': (MEAN_LEAF_ANGLE, STD_LEAF_ANGLE),
                   'hotspot': (MEAN_HOTSPOT, STD_HOTSPOT),
                   'bs': (MEAN_BS, STD_BS)}

UNIFORM_DIST = 1
GAUSSIAN_DIST = 2
GAMMA_DIST = 3
SALTELLI_DIST = 4

prosail_distribution = {'N_leaf': UNIFORM_DIST,
                        'Cab': GAUSSIAN_DIST,
                        'Car': GAUSSIAN_DIST,
                        'Cbrown': GAUSSIAN_DIST,
                        'Cw': GAUSSIAN_DIST,
                        'Cm': GAUSSIAN_DIST,
                        'Ant': GAUSSIAN_DIST,
                        'LAI': GAUSSIAN_DIST,
                        'leaf_angle': GAUSSIAN_DIST,
                        'hotspot': GAUSSIAN_DIST,
                        'bs': GAMMA_DIST}

df_bounds = pd.DataFrame(prosail_bounds, index=['min', 'max'])
n_simulations = 10000

# Stack spectral bands
try:
    SAT_LIBRARY = srf_info[satellite]['SAT_LIBRARY']
    band_names = srf_info[satellite]['bands_names']
    band_coords = srf_info[satellite]['bands_coords']
    scale = srf_info[satellite]['scale']
    offset = srf_info[satellite]['offset']
except KeyError:
    print('SAT library not available')
# Stack spectral bands

srf_file = SRF_LIBRARY / SAT_LIBRARY
srf = []
srfs = np.genfromtxt(srf_file, dtype=None, names=True)
for band in band_names:
    srf.append(srfs[band])

# open as pandas dataframe
srf_df = pd.read_csv(srf_file, sep='\t')


def main(path, out_dir):
    print(path)
    from pypro4sail import machine_learning_regression as inv
    # print(f'Setting up {n_simulations} simulations with inputs bounds:\n\n {df_bounds[OBJ_PARAM_NAMES]}')
    params_orig = inv.build_prosail_database(n_simulations,
                                             param_bounds=prosail_bounds,
                                             distribution=prosail_distribution,
                                             moments=prosail_moments)
    # print(params_orig)
    # params_orig_graph = pd.DataFrame(params_orig)
    #
    # numeric_cols = params_orig_graph.select_dtypes(include=['float64', 'int64']).columns
    # fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    # axes = axes.flatten()
    #
    # Loop through columns and plot
    # for i, col in enumerate(numeric_cols):
    #     sns.kdeplot(params_orig_graph[col], ax=axes[i], color='black', linewidth=2)
    #     axes[i].set_title(f'Distribution of {col}')
    #     axes[i].set_xlabel(col)
    #     axes[i].set_ylabel('Frequency')
    #     axes[i].grid(False)
    #
    # Turn off any remaining empty subplots
    # for j in range(len(numeric_cols), len(axes)):
    #     axes[j].axis('off')
    #
    # plt.tight_layout()
    # plt.savefig(rf'C:\Users\mqalborn\Desktop\ET_3SEB\figures/prosail_distribution.jpg', dpi=300)
    # plt.show()
    image = rioxarray.open_rasterio(path)  # dims: (band, y, x)
    image = image.assign_coords(band=band_coords)
    BandsInputs = image.sel(band=band_names) * scale + offset

    try:
        date_str = os.path.basename(path).split("_")[0].split("T")[0]
        date = pd.to_datetime(date_str, format="%Y%m%d").date()
    except ValueError:
        date_str = os.path.basename(path).split("_")[3].split('.')[0]
        date = pd.to_datetime(date_str, format="%Y%m%d").date()

    metadata_image = metadata[metadata.date == date]
    date = pd.to_datetime(metadata_image.overpass_solar_time.values).to_pydatetime()[0]

    SAA = float(metadata_image.SAA.mean())
    SZA = float(metadata_image.SZA.mean())
    VZA = float(metadata_image.VZA.mean())
    AOT = float(metadata_image.AOT.mean()) # The AOT at 550nm at the sensor
    WVP = float(metadata_image.WVP.mean()) # precipitable water vapour,
    # The total amount of water in a vertical path through the atmosphere (in g/cm^2 = cm)

    soil_spectrum = build_soil_database(params_orig["bs"])

    # print(f"Running 6S for estimation of diffuse/direct irradiance")
    skyl = get_diffuse_radiation_6S(AOT, WVP, SZA, SAA, date, altitude=0.1)

    # spectral range
    wls_sim = np.arange(400, 2501)

    # number of CPUs to use to perform simulations
    # (can change depending on number of CPUs in your computer)
    njobs = 4
    # generate LUT
    rho_canopy_vec, params = inv.simulate_prosail_lut_parallel(
            njobs,
            params_orig,
            wls_sim,
            soil_spectrum,
            skyl=skyl,
            sza=SZA,
            vza=VZA,
            psi=0,
            srf=srf,
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

    fis = pd.DataFrame(fis)
    fis.loc[:, 'bands'] = band_names

    # print(date[0])
    fis.loc[:, 'date'] = date
    arr = BandsInputs.values
    y_coords = BandsInputs.y
    x_coords = BandsInputs.x

    # print(type(BandsInputs))
    # bio_dict = {}
    dataarrays = []
    for i, param in enumerate(OBJ_PARAM_NAMES):
        # output = np.full(arr.size, np.nan)

        # print(f"Appliying {param} model to S2 image reflectance array")
        if np.any(arr):
            print(arr.shape)
            arr_reshape = arr.reshape((arr.shape[0], -1)).T
            arr_scaled = input_scalers[param].transform(arr_reshape)
            arr_scaled_pred = regs[param].predict(arr_scaled)
            out = output_scalers[param].inverse_transform(arr_scaled_pred.reshape(-1, 1)).reshape(-1)

        output = out.reshape(arr[0, :, :].shape)
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
    # print(col.band.values)
    # col = col.assign_coords(band=['Cab', 'Car', 'Cm', 'Cw', 'Ant', 'Cbrown', 'LAI', 'leaf_angle'])
    # col = col.sel(band='LAI')
    col = col.rio.write_crs("EPSG:32610")

    out_file = out_dir + 'PROSAIL_' + os.path.basename(path)
    col.rio.to_raster(out_file, dtype="float32", nodata=np.nan)
    return fis

#
    # data_example.loc[:, 'LAI_class'] = data_example.LAI.astype(int)
    # sns.set_context('notebook')
    # sns.set_palette("Spectral")
    # g = sns.lineplot(x='band', y='rho', marker='o', data=data_example, hue='LAI_class')
    # g.set_xlabel('Sentinel-2 Band')
    # g.set_ylabel('PROSAIL Reflectance')
    # plt.show()
    # print(data_example.head())
    # print('Done!')
#

if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(folder_images, "*.tif")))
    # fis = main(files[0], out_dir)

    from matplotlib import pyplot as plt
    import seaborn as sns

    fis = [main(x, out_dir) for x in files]
    # fis = pd.concat(fis)
    # sns.barplot(x="bands", y="LAI", data=fis)
    # plt.show()
    # for x in files:
    #     print(x)
    #     main(x, out_dir)
