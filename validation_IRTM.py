import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
from matplotlib import pyplot as plt
from functions.biophysical import get_diffuse_radiation_6S, build_soil_database, SRF_LIBRARY
import os
import rioxarray
import xarray as xr
import glob
import sys


location = 'BLS'
satellite = 'L08' #['S2', 'L08']

folder_images = rf'C:\Users\mqalborn\Desktop\ET_3SEB\satellite\{location}/{satellite}/RHO/'
metadata = pd.read_csv(folder_images + 'METADATA.csv')
metadata.loc[:, 'date'] =pd.to_datetime(metadata.overpass_solar_time).dt.date

# path_dir = rf'C:\Users\mqalborn\Desktop\ET_3SEB\satellite\{location}/{satellite}/RHO/'
files = sorted(glob.glob(os.path.join(folder_images, "*.tif")))

srf_info = {'L08': {'SAT_LIBRARY' : f'Landsat8.txt',
                    'bands_names' : ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']},
            'S2': {'SAT_LIBRARY' : f'Sentinel2A.txt',
                    'bands_names' : ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']}}

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

try:
    SAT_LIBRARY = srf_info[satellite]['SAT_LIBRARY']
    band_names = srf_info[satellite]['bands_names']
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

# Show spectral response function (SRF) curves
# plot spectral response function
# colormap = plt.cm.rainbow # Choose a colormap
# get color within colormap range for each band (depends on number of bands)
# colors = [colormap(x / (len(band_names) - 1)) for x in range(len(band_names))]
# plt.figure(figsize=(9, 5))
# plt.title(f'Spectral Response Function (SRF) - Sentinel-2', fontsize=14)
# plt.xlabel('Wavelength (nm)', fontsize=12)
# plt.xlim(400, 2500)
# plt.ylabel('Relative Response (-)', fontsize=12)
# plt.ylim(0, 1)
# plt.grid(True)
# i = 0
# for band in band_names:
#     plt.plot(srf_df['SR_WL'], srf_df[band], color=colors[i], label = f'{str(band)}')
#     i += 1
#
# plt.legend(loc='lower right', ncol=5)
# plt.show()

def main(path):
    print(path)
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

    from pypro4sail import machine_learning_regression as inv
    print(f'Setting up {n_simulations} simulations with inputs bounds:\n\n {df_bounds[OBJ_PARAM_NAMES]}')
    params_orig = inv.build_prosail_database(n_simulations,
                                             param_bounds=prosail_bounds,
                                             distribution=prosail_distribution,
                                             moments=prosail_moments)
    print(f"Running 6S for estimation of diffuse/direct irradiance")

    soil_spectrum = build_soil_database(params_orig["bs"])
    skyl = get_diffuse_radiation_6S(AOT, WVP, SZA, SAA, date, altitude=0.1)

    # spectral range
    wls_sim = np.arange(400, 2501)
    # number of CPUs to use to perform simulations
    # (can change depending on number of CPUs in your computer)
    njobs = 4
    # generate LUT
    rho_train, params_train = inv.simulate_prosail_lut_parallel(
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
            inv.train_reg(rho_train,
                          params_train[param].reshape(-1, 1),
                          scaling_input='normalize',
                          scaling_output='normalize',
                          regressor_opts=scikit_regressor_opts,
                          reg_method="random_forest")

        input_scalers[param] = input_gauss_scaler
        output_scalers[param] = output_gauss_scaler
        regs[param] = reg
        fis[param] = reg.feature_importances_

    pd_fis = pd.DataFrame(fis)
    pd_fis.loc[:, 'band'] = band_names
    pd_fis.loc[:, 'date'] = date.date()

    out_dir = rf"C:\Users\mqalborn\Desktop\ET_3SEB\results\feature_importance_prosail/{satellite}/FI_{satellite}_{date.date()}.csv"
    pd_fis.to_csv(out_dir)

    rho_test, params_test = inv.simulate_prosail_lut_parallel(
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

    # param = 'LAI'
    output_est = {}
    for i, param in enumerate(OBJ_PARAM_NAMES):
        output = (output_scalers[param].inverse_transform(
            regs[param].predict(
                input_scalers[param].transform(
                    rho_test #1
                )
            ).reshape(-1, 1)).reshape(-1)
        )
        output_est[param + '_est'] = output

    param_test = pd.DataFrame(params_test)
    param_est = pd.DataFrame(output_est)
    # print(param_est.shape)
    param_val = pd.concat([param_test, param_est], axis=1).drop('index', axis=1)
    param_val.loc[:, 'date'] = date.date()
    # print(param_val.shape)
    # y = param_val['LAI_est']
    # x = param_val['LAI']
    # plt.hexbin(x, y, gridsize=120, bins='log', cmap='viridis', mincnt=1)
    # plt.scatter(x, y,  s=4, alpha=0.15)
    # plt.plot([0, 7], [0, 7],  linestyle=':', color='k', lw=2)
    # plt.show()
    # out_file = out_dir + 'LAI_' + os.path.basename(path)
    out_dir = rf"C:\Users\mqalborn\Desktop\ET_3SEB\results\validation_prosail/{satellite}/VAL_{date.date()}.csv"
    param_val.to_csv(out_dir)

    #
    # return param_val


if __name__ == "__main__":
    _ = [main(x) for x in files[6:]]
    # param_val = pd.concat(param_val)
    # out_dir = rf"C:\Users\mqalborn\Desktop\ET_3SEB\results/PRO4SAIL_val.csv"
    # _ = main(files)

    # param_val.to_csv(out_dir)
    # print(param_val)
