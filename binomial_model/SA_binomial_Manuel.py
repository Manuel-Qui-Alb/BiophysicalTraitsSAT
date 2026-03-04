import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BinomialModelPrism_DirectandDiffuseRadiation_Manuel import compute_binomial_prism_manuel
import pyTSEB.TSEB as TSEB
import pyTSEB.meteo_utils as mo
import GeneralFunctions as GF
from SALib.sample import saltelli
import time
import pvlib

site = pvlib.location.Location(
    latitude=38.5449,
    longitude=-121.7405,
    tz='US/Pacific',
    altitude=18,      # meters; small effect, 15–25 m is fine
    name='Davis_CA'
)


def solar_doy_hour_to_times(doys, solar_hours, year=2024,
                            lon_deg=-121.7405,
                            tz='US/Pacific',
                            lon_std_deg=-120.0):
    """
    doys: array-like, 1..365
    solar_hours: array-like, local solar time in decimal hours (e.g., 12.5 = 12:30 solar)
    Returns: tz-aware DatetimeIndex in civil time that corresponds to those solar hours.
    """
    doys = np.asarray(doys).astype(int).ravel()
    solar_hours = np.asarray(solar_hours).astype(float).ravel()

    # equation of time (minutes), Spencer 1971 (pvlib returns minutes)
    # pvlib expects dayofyear array
    eot_min = pvlib.solarposition.equation_of_time_spencer71(doys)

    # time correction (minutes): EoT + 4*(lon - lon_std)
    tc_min = eot_min + 4.0 * (lon_deg - lon_std_deg)

    # convert solar hour -> civil hour
    civil_hours = solar_hours - tc_min / 60.0

    base = pd.Timestamp(f'{year}-01-01', tz=tz)
    times = base + pd.to_timedelta(doys - 1, unit='D') + pd.to_timedelta(civil_hours, unit='h')
    return pd.DatetimeIndex(times)



def binomial_model(lai, fv_var, h_V, row_sep, plant_sep_var,
                   doy, hour, row_radians,
                   abs_vis_leaf,
                   abs_nir_leaf,
                   rho_vis_soil, rho_nir_soil,
                   Nz_diff=16, Nphi_diff=32, Nbins=40, nrays=5000, shape="prism"):

    # abs_vis_leaf = 1 - rho_vis_leaf - tau_vis_leaf
    # abs_nir_leaf = 1 - rho_nir_leaf - tau_nir_leaf

    # vza_rad = np.radians(vza_degrees)
    # Campbell and Norman 1998. Page 172.
    times = solar_doy_hour_to_times(doy, hour, year=2024, lon_deg=site.longitude, tz=site.tz)

    solpos = site.get_solarposition(times)
    sza_degrees = solpos.zenith
    saa_degrees = solpos.azimuth

    irradiance_cs = site.get_clearsky(times)
    St = irradiance_cs.ghi
    St, sza_degrees, saa_degrees = [np.reshape(x, lai.shape) for x in [St, sza_degrees, saa_degrees]]

    sza_radians = np.radians(sza_degrees)
    saa_radians = np.radians(saa_degrees)

    difvis, difnir, fvis, fnir = TSEB.rad.calc_difuse_ratio(
        S_dn=St,
        sza=sza_degrees,
        press=np.full_like(lai, 1013)
    )

    skyl = fvis * difvis + fnir * difnir
    Srad_dir = (1. - skyl) * St
    Srad_diff = skyl * St

    psi_radians = saa_radians - row_radians

    # saltelli.sample create random values that sometime don't make any sense
    # LAI, FV, wc, sp, sr are intrinsically related, so they can not be analysed independently
    # For example FV = 0 with LAI = 8. wc = 8, sp and sr = 5
    # They'll be estimated mainly based on the LAI
    K_be = GF.estimate_Kbe(x_LAD=1, sza=0)
    fv0 = (1 - np.exp(-K_be * fv_var  * lai))
    w_V = fv0 * row_sep
    plant_sep = row_sep * plant_sep_var # sp [0.5, 1]

    start_time_man = time.perf_counter()
    # psi_radians = np.arccos(np.cos(saa_radians - row_radians))
    # phi_radians = np.arccos(np.cos(saa_radians - row_azimuth_radians))

    Rc_binomial, Rs_binomial = compute_binomial_prism_manuel(
        sr=row_sep, #Row spacing (meters)
        sza=sza_radians, # sun zenith angle (radians)
        psi=psi_radians, # sun azimuth relative to row orientation (radians)
        lai=lai, # Leaf Area Index
        ameanv=abs_vis_leaf, # Leaf absorptivity in the visible (PAR) band
        ameann=abs_nir_leaf, # Leaf absorptivity in the near infra-red band (NIR) band
        rsoilv=rho_vis_soil, # Soil absorptivity in the visible (PAR) band
        rsoiln=rho_nir_soil,  # Soil absorptivity in the near infra-red band (NIR) band
        Srad_dir=Srad_dir,  # Direct-beam incoming radiation (W m-2)
        Srad_diff=Srad_diff,  # Diffuse incoming radiation (W m-2)
        fvis=fvis, # Fraction incoming radiation in the visible part of the spectrum
        fnir=fnir,  # Fraction incoming radiation in the near infra-red part of the spectrum
        CanopyHeight=h_V,# Canopy heigth (meters)
        wc=w_V, # Canopy width (meters)
        sp=plant_sep, #Plant spacing (meters)
        Gtheta=0.5, # fraction of leaf area projected in the direction of the sun
        nrays=nrays, #
        Nbins=Nbins,
        shape=shape,   # <-- shape of the canopy
        Nz_diff=Nz_diff,
        Nphi_diff=Nphi_diff,  # sampling for diffuse hemisphere
    )

    end_time_man = time.perf_counter()

    elapsed_time_man = end_time_man - start_time_man
    print(f"Execution time Manuel: {elapsed_time_man} seconds")

    return St, Rc_binomial, Rs_binomial

names = ['lai', 'fv_var', 'h_V', 'row_sep', 'plant_sep_var',
         'doy', 'hour', 'row_radians',
         'abs_vis_leaf', 'abs_nir_leaf',
         'rho_vis_soil', 'rho_nir_soil']

problem = {
    "num_vars": 12,
    "names": names,
    "bounds": [
        [0.2, 5],  # lai: orchards/crops
        [0.25, 1],  # fv_var
        [1, 5],  # CanopyHeight
        [2.0, 8.0],  # row_sep: typical orchard rows
        [0.5, 1],  # plant_sep_var: plant space in relation to sr
        [1, 365],  # doy: avoid near 0 and near 90
        [11, 16], # hour: relative azimuth 0..pi
        [np.deg2rad(0), np.deg2rad(180)], # row_orientation: relative azimuth 0..pi
        [0.80, 0.90],  # 18. abs_vis_leaf
        [0.05, 0.15],  # 19. abs_nir_leaf
        [0.05, 0.30],  # 22. rho_vis_soil
        [0.20, 0.45],  # 23. rho_nir_soil
    ],
}

N = 100
second_order = False
X = saltelli.sample(problem, N, calc_second_order=second_order)

inputs_dict = {f"{names[i]}": X[:, i:i+1] for i in range(X.shape[1])}
data = pd.DataFrame(X)
data.columns = names

# Testing with 1d array
St, Rc_binomial, Rs_binomial = binomial_model(
    lai=inputs_dict['lai'],
    fv_var=inputs_dict['fv_var'],
    h_V=inputs_dict['h_V'],
    row_sep=inputs_dict['row_sep'],
    plant_sep_var=inputs_dict['plant_sep_var'],
    doy=inputs_dict['doy'],
    hour=inputs_dict['hour'],
    row_radians=inputs_dict['row_radians'],
    abs_vis_leaf=inputs_dict['abs_vis_leaf'],
    abs_nir_leaf=inputs_dict['abs_nir_leaf'],
    rho_vis_soil=inputs_dict['rho_vis_soil'],
    rho_nir_soil=inputs_dict['rho_nir_soil'],
    Nz_diff=16, Nphi_diff=32, Nbins=40, nrays=5000, shape="prism"
)

data.loc[:, 'Rc_binomial'] = Rc_binomial
data.loc[:, 'Rs_binomial'] = Rs_binomial
data.loc[:, 'St'] = St
data.to_csv("files/SA_Binomial.csv", index=False)

Sn = Rc_binomial + Rs_binomial
Sr = St - Sn
alb = Sr / St

plt.hist(Sr, bins=100, density=True)
plt.show()

# plt.hist(alb, bins=100, density=True)
# plt.show()
