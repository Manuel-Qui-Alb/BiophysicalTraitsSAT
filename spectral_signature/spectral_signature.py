import os
import glob
import pandas as pd
import numpy as np
from functions.biophysical import get_diffuse_radiation_6S, build_soil_database, SRF_LIBRARY

path_dir = rf'C:\Users\mqalborn\Desktop\ET_3SEB\PISTACHIO\satellite_images\Sentinel2/METADATA.csv'
metadata = pd.read_csv(path_dir)
metadata.loc[:, 'date'] =pd.to_datetime(metadata.overpass_solar_time).dt.date

LAI = np.arange(0, 6)
params_orig = {'N_leaf': np.repeat(1.4, len(LAI)),
               'Cab': np.repeat(50, len(LAI)),
               'Car': np.repeat(9.55, len(LAI)),
               'Cbrown': np.repeat(0, len(LAI)),
               'Cw': np.repeat(0.01, len(LAI)),
               'Cm': np.repeat(0.012, len(LAI)),
               'Ant': np.repeat(1, len(LAI)),
               'LAI': LAI,
               'leaf_angle': np.repeat(60, len(LAI)),
               'hotspot': np.repeat(0.2, len(LAI)),
               'bs': np.repeat(1.2, len(LAI))}

# spectral range
srf = np.identity(2101)
srf = [row for row in srf]
wls_sim = np.arange(400, 2501)

def main(path, out_dir):
    date_str = os.path.basename(path).split("_")[0].split("T")[0]
    date = pd.to_datetime(date_str, format="%Y%m%d").date()

    metadata_image = metadata[metadata.date == date]
    date = pd.to_datetime(metadata_image.overpass_solar_time.values).to_pydatetime()[0]

    SAA = float(metadata_image.MEAN_SOLAR_AZIMUTH_ANGLE)
    SZA = float(metadata_image.MEAN_SOLAR_ZENITH_ANGLE)
    VZA = float(metadata_image.MEAN_INCIDENCE_ZENITH_ANGLE_B8)
    AOT = float(metadata_image.AOT) # The AOT at 550nm at the sensor
    WVP = float(metadata_image.WVP) # precipitable water vapour,
    # The total amount of water in a vertical path through the atmosphere (in g/cm^2 = cm)

    skyl = get_diffuse_radiation_6S(AOT, WVP, SZA, SAA, date, altitude=0.1)
    soil_spectrum = build_soil_database(params_orig["bs"])
    # import INSIDE the guard so the package isn’t imported at top level
    from pypro4sail import machine_learning_regression as inv  # replace with real module

    print(f"Running 6S for estimation of diffuse/direct irradiance")
    skyl = get_diffuse_radiation_6S(AOT, WVP, SZA, SAA, date, altitude=0.1)


    from pypro4sail import machine_learning_regression as inv
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
            outfile=False,
            calc_FAPAR=False,
            reduce_4sail=True)
    rho_canopy_vec = pd.DataFrame(rho_canopy_vec)
    rho_canopy_vec = rho_canopy_vec.T
    rho_canopy_vec.loc[:, 'band'] = wls_sim
    rho_canopy_vec.to_csv(out_dir + "{}_rho_canopy_vec.csv".format(date_str))
    print(rho_canopy_vec.head())

if __name__ == "__main__":
    path_dir = rf'C:\Users\mqalborn\Desktop\ET_3SEB\PISTACHIO\satellite_images\Sentinel2\march_2025'
    out_dir = rf"C:\Users\mqalborn\Desktop\ET_3SEB\PISTACHIO\RTM\prosail/"

    files = sorted(glob.glob(os.path.join(path_dir, "*.tif")))
    # main(files[0], out_dir)
    for x in files:
        print(x)
        main(x, out_dir)