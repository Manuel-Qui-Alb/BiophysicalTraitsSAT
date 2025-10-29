import pyTSEB.TSEB as TSEB
from pyTSEB import meteo_utils as met
from pyTSEB import net_radiation as rad
import numpy as np
import glob
import rioxarray
import os
import pandas as pd
import datetime
import xarray as xr

dir_path = rf'C:\Users\mqalborn\Desktop\ET_3SEB\satellite\RIP720\L08\LAI_TRAD'
paths = glob.glob(dir_path + '/*.tif')
f = paths[0]
id = os.path.basename(f).split('.')[0]
date_img = pd.to_datetime(id.split('_')[-1])

metadata = pd.read_csv(dir_path + '/METADATA.csv')
metadata = metadata[pd.to_datetime(metadata.OST).dt.date == date_img.date()]
farm = 'RIP720'

def tseb_raster(f):
    raster = rioxarray.open_rasterio(f)
    raster = raster.assign_coords(band=['LAI', 'TRAD'])

    # Orchard Characteristics
    row_width = 3.35
    tree_dist = 1.52

    lon = -120.17574403015209
    lat = 36.84941819235097
    stdlon = -120

    # Satellite
    Tr_K = raster[raster.band=='TRAD'].values[0] # Radiometric Temperature
    date_tr = pd.to_datetime(metadata.OST.values[0])
    LAI = raster[raster.band=='LAI'].values[0] # Leaf Area Index
    fcov = np.full(LAI.shape, 0.3)
    h_C = np.full(LAI.shape, 1.5) # Canopy height
    w_c = np.full(LAI.shape, np.mean([row_width, tree_dist]) * fcov) # Canopy Height
    vza = 0 # View Zenith Angle
    sza = np.full(LAI.shape, metadata.SZA.mean())

    # Meteorological
    data_meteo = (pd.read_csv(rf'C:\Users\mqalborn\Desktop\ET_3SEB\results\ERA5/ERA5_RIP720.csv')
                  .drop('Unnamed: 0', axis=1))
    data_meteo.timestamp = pd.to_datetime(data_meteo.timestamp)
    data_meteo.loc[:, 'date'] = data_meteo.timestamp.dt.date

    data_meteo = data_meteo[data_meteo.timestamp.dt.hour.isin([11])]
    # data_meteo = data_meteo.groupby('date').mean().reset_index()
    data_meteo = data_meteo[data_meteo.date == date_img.date()]

    T_A_K = np.full(LAI.shape, data_meteo.temperature_2m) # Air temperature
    u = np.full(LAI.shape, data_meteo.wind_speed_10m) # Wind speed at 10 m
    ea = np.full(LAI.shape, data_meteo.wapor_pressure) #Water vapour pressure above the canopy (mb)
    p = np.full(LAI.shape, data_meteo.surface_pressure) # Atmospheric pressure (mb), use 1013 mb by default.
    S_dn = np.full(LAI.shape, data_meteo.surface_solar_radiation_downwards) # Incoming shortwave radiation (W m-2)
    L_dn = np.full(LAI.shape, data_meteo.surface_thermal_radiation_downwards) # Downwelling longwave radiation (W m-2).
    z_u = np.full(LAI.shape,10) # Height of measurement of windspeed (m).
    z_T = np.full(LAI.shape, 2) # Height of measurement of air temperature (m).

    # Sun Zenith Angle
    saa = np.full(LAI.shape, metadata.SAA.mean())
    ftime = np.full(LAI.shape, date_tr.hour + date_tr.minute/60 + date_tr.second/3600)

    # Set up inputs
    X_LAD = np.full(LAI.shape,1)

    emis_C = np.full(LAI.shape,0.97) # Canopy emissivity
    emis_S = np.full(LAI.shape, 0.95) # Soil emissivity
    KB_1_DEFAULT = np.full(LAI.shape,0.0)

    RHO_VIS_C = np.full(LAI.shape,0.07)
    TAU_VIS_C = np.full(LAI.shape,0.08)
    RHO_NIR_C = np.full(LAI.shape,0.32)
    TAU_NIR_C = np.full(LAI.shape,0.33)
    RHO_VIS_S = np.full(LAI.shape,0.15)
    RHO_NIR_S = np.full(LAI.shape,0.25)
    Z0_SOIL = np.full(LAI.shape,0.05)

    ### Net Radiation and clumping
    # calculate diffuse/direct ratio

    difvis, difnir, fvis, fnir = TSEB.rad.calc_difuse_ratio(
        S_dn=S_dn,
        sza=sza,
        press=p
    )
    skyl = fvis * difvis + fnir * difnir
    Sdn_dir = (1. - skyl) * S_dn
    Sdn_dif = skyl * S_dn

    # incoming long wave radiation
    emisAtm = rad.calc_emiss_atm(
        ea=ea,
        t_a_k=T_A_K
    )
    Lsky = emisAtm * met.calc_stephan_boltzmann(T_K=T_A_K)

    # We need to compute SAA
    # doy = date_tr.dayofyear
    # dec_time = date_tr.hour + date_tr.minute / 60.
    # sza, saa = met.calc_sun_angles(lat=lat, lon=lon, stdlon=stdlon, doy=doy, ftime=ftime)

    # to take into account row strucure on vegetation clumping
    row_direction = 90
    psi = row_direction - saa
    psi = np.full(LAI.shape, psi)

    Omega0 = TSEB.CI.calc_omega0_Kustas(LAI=LAI, f_C=fcov, x_LAD=1, isLAIeff=True)

    Omega = TSEB.CI.calc_omega_rows(lai=LAI, f_c0=fcov, theta=sza, psi=psi, w_c=w_c, x_lad=X_LAD, is_lai_eff=True)

    F = LAI / fcov
    # effective LAI (tree crop)
    LAI_EFF =  F * Omega
    Sn_C, Sn_S = TSEB.rad.calc_Sn_Campbell(
        lai=LAI,
        sza=sza,
        S_dn_dir=Sdn_dir,
        S_dn_dif=Sdn_dif,
        fvis=fvis,
        fnir=fnir,
        rho_leaf_vis=RHO_VIS_C,
        tau_leaf_vis=TAU_VIS_C,
        rho_leaf_nir=RHO_NIR_C,
        tau_leaf_nir=TAU_NIR_C,
        rsoilv=RHO_VIS_S,
        rsoiln=RHO_NIR_S ,
        x_LAD=1,
        LAI_eff=None
    )


    Sn_C[~np.isfinite(Sn_C)] = 0
    Sn_S[~np.isfinite(Sn_S)] = 0

    z_0m, d_0 = TSEB.res.calc_roughness(LAI=LAI,
                                        h_C=h_C,
                                        w_C=w_c,
                                        landcover=np.full_like(LAI, TSEB.res.CROP), #11?,
                                        f_c=fcov)

    d_0[d_0 < 0] = 0
    z_0m[z_0m < 0.05] = 0.05

    print('Done!')

    flag, T_S, T_C, T_AC, L_nS, L_nC, LE_C, H_C, LE_S, H_S, G, R_S, R_x, R_A, u_friction, L, n_iterations = TSEB.TSEB_PT(
        Tr_K=Tr_K,
        vza=vza,
        T_A_K=T_A_K,
        u=u,
        ea=ea,
        p=p,
        Sn_C=Sn_C,
        Sn_S=Sn_S,
        L_dn=L_dn,
        LAI=LAI,
        h_C=h_C,
        emis_C=emis_C,
        emis_S=emis_S,
        z_0M=z_0m,
        d_0=d_0,
        z_u=z_u,
        z_T=z_T,
        leaf_width=0.1,
        z0_soil=0.01,
        alpha_PT=1.26,
        x_LAD=1,
        f_c=1.0,
        f_g=1.0,
        w_C=1.0,
        resistance_form=None,
        calcG_params=None,
        const_L=None,
        kB=KB_1_DEFAULT,
        massman_profile=None,
        verbose=True
    )

    y_coords = raster.y
    x_coords = raster.x
    ds = xr.Dataset(
        {
            "LE_C": (("y", "x"), LE_C),
            "H_C":  (("y", "x"), H_C),
            "LE_S": (("y", "x"), LE_S),
            "H_S":  (("y", "x"), H_S),
            "T_S":  (("y", "x"), T_S),
            "T_C":  (("y", "x"), T_C),
        },
        coords={
            "y": y_coords,
            "x": x_coords
        }
    )

    ds = ds.rio.write_crs(raster.rio.crs)
    basename = os.path.basename(f).split('_')[-1]
    filename = rf'TSEB_{farm}_{basename}'
    ds.rio.to_raster(rf"C:\Users\mqalborn\Desktop\ET_3SEB\satellite\RIP720\L08\TSEB/{filename}")


_ = [tseb_raster(f) for f in paths]