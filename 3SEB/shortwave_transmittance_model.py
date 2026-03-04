# Cover crop Shortwave Tranmittance Models
import numpy as np
import TSEB.functions as myTSEB
import pvlib
import pandas as pd
import pyTSEB.TSEB as TSEB
import GeneralFunctions as GF
from pyTSEB import net_radiation as rad
from matplotlib import pyplot as plt


site = pvlib.location.Location(
    latitude=38.5449,
    longitude=-121.7405,
    tz='US/Pacific',
    altitude=18,      # meters; small effect, 15–25 m is fine
    name='Davis_CA'
)

def set_legend(loc=(0.02, 0.3)):
    try:
        # Get handles and labels from both axes
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        #
        # # Combine the handles and labels lists
        all_handles = handles1 + handles2
        all_labels = labels1 + labels2
        #
        # # Create a single legend on ax1 (or ax2, or fig)
        ax1.legend(all_handles, all_labels, loc=loc)
    except:
        plt.legend(loc='best')

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


LAI_CC = np.arange(0, 10, 1).astype(float)
LAI_T = np.full_like(LAI_CC, 2)
# LAI_T = np.random.uniform(2.9, 3, 1000)
Tair = np.full_like(LAI_CC, 301)
# Tair = np.random.uniform(301, 302, 1000)
Trad_T_var = np.full_like(LAI_CC, 1)
# Trad_T_var = np.random.uniform(0, 0.1, 1000)
Trad_G_var = np.arange(0, 4, 0.4)[::-1]
# Trad_G_var = np.random.uniform(1, 1.01, 1000)
# Trad_S_var = np.random.uniform(, 2, 1000)

Trad_T = Tair + Trad_T_var
Trad_G = Tair + Trad_G_var
Trad_S = Tair + Trad_G_var + 2

ea = np.full_like(LAI_CC, 20)
# ea = np.random.uniform(20, 21, 1000)

doy = np.full_like(LAI_CC, 179)
# doy = np.random.uniform(179, 180, 1000)
hour = np.full_like(LAI_CC, 12)
# hour = np.random.uniform(12, 13, 1000)

x_LAD = np.full_like(LAI_CC, 1)
omega_un = np.full_like(LAI_CC, 1)
# abs_vis_leaf = np.full_like(LAI_un, 0.85)
# abs_nir_leaf = np.full_like(LAI_un, 0.15)
rho_vis_leaf = np.full_like(LAI_CC, 0.10)
rho_nir_leaf = np.full_like(LAI_CC, 0.45)
tau_vis_leaf = np.full_like(LAI_CC, 0.05)
tau_nir_leaf =np.full_like(LAI_CC, 0.45)

emis_leaf = np.full_like(LAI_CC, 0.97)
# emiss_leaf =np.full_like(LAI_CC, 0.45)

rho_vis_soil =  np.full_like(LAI_CC, 0.15)
rho_nir_soil =  np.full_like(LAI_CC,  0.40)
emis_soil = np.full_like(LAI_CC, 0.98)
rho_soil = np.array((rho_vis_soil, rho_nir_soil))

times = solar_doy_hour_to_times(doy, hour, year=2024, lon_deg=site.longitude, tz=site.tz)

solpos = site.get_solarposition(times)
sza_degrees = solpos.zenith.values
saa_degrees = solpos.azimuth.values

irradiance_cs = site.get_clearsky(times)
St = irradiance_cs.ghi.values

difvis, difnir, fvis, fnir = TSEB.rad.calc_difuse_ratio(
    S_dn=St,
    sza=sza_degrees,
    press=np.full_like(LAI_T, 1013)
)

skyl = fvis * difvis + fnir * difnir
St_dir = (1. - skyl) * St
St_dif = skyl * St

K_be = GF.estimate_Kbe(x_LAD=1, sza=0)
fv_CC = (1 - np.exp(-K_be * LAI_CC))
fv_T = (1 - np.exp(-K_be * LAI_T))
L_inc = rad.calc_longwave_irradiance(
    ea,
    Tair,
    p=1013,
    z_T=2,
    h_C=2
)

Ln_T, Ln_G = rad.calc_L_n_Campbell(
    T_C=Trad_T,
    T_S=Trad_G,
    L_dn=L_inc,
    lai=LAI_T,
    emisVeg=emis_leaf,
    emisGrd=emis_soil,
    x_LAD=x_LAD
)

albb_ground, albd_ground, taubt_CC, taudt_CC = rad.calc_spectra_Cambpell(
    lai=LAI_CC,
    sza=sza_degrees,
    rho_leaf=np.array((rho_vis_leaf, rho_nir_leaf)),
    tau_leaf=np.array((tau_vis_leaf, tau_nir_leaf)),
    rho_soil=rho_soil,
    x_lad=x_LAD,
    lai_eff=None)


_, _, taubt_T, taudt_T = rad.calc_spectra_Cambpell(LAI_T,
                                                   sza_degrees,
                                                   np.array((rho_vis_leaf, rho_nir_leaf)),
                                                   np.array((tau_vis_leaf, tau_nir_leaf)),
                                                   albb_ground,
                                                   x_lad=x_LAD,
                                                   lai_eff=None)

Lt_T = np.nan_to_num(LAI_T / fv_T)
# Lt_T = np.full_like(Lt_T, 5)
albb_T, albd_T, _, _ = rad.calc_spectra_Cambpell(Lt_T,
                                                 sza_degrees,
                                                 np.array((rho_vis_leaf, rho_nir_leaf)),
                                                 np.array((tau_vis_leaf, tau_nir_leaf)),
                                                 albb_ground,
                                                 x_lad=x_LAD,
                                                 lai_eff=None)

Lt_CC = np.nan_to_num(LAI_CC / fv_CC)
albb_CC, albd_CC, _, _ = rad.calc_spectra_Cambpell(Lt_CC,
                                                   sza_degrees,
                                                   np.array((rho_vis_leaf, rho_nir_leaf)),
                                                   np.array((tau_vis_leaf, tau_nir_leaf)),
                                                   rho_soil,
                                                   x_lad=x_LAD,
                                                   lai_eff=None)

# plt.scatter(albb_CC[0], albb_T1[0])
# plt.show()
Sn_T = ((1.0 - taubt_T[0]) * (1.0 - albb_T[0]) * St_dir * fvis
        + (1.0 - taubt_T[1]) * (1.0 - albb_T[1]) * St_dir * fnir
        + (1.0 - taudt_T[0]) * (1.0 - albd_T[0]) * St_dif * fvis
        + (1.0 - taudt_T[1]) * (1.0 - albd_T[1]) * St_dif * fnir)

# Sn_t = Sn_ov * taubt_ov[0]
St_dir_vis_ground = taubt_T[0] * St_dir * fvis
St_dir_nir_ground = taubt_T[1] * St_dir * fnir
St_dif_vis_ground = taudt_T[0] * St_dif * fvis
St_dif_nir_ground = taudt_T[1] * St_dif * fnir

Sn_CC = ((1.0 - taubt_CC[0]) * (1.0 - albb_CC[0]) * St_dir_vis_ground
         + (1.0 - taubt_CC[1]) * (1.0 - albb_CC[1]) * St_dir_nir_ground
         + (1.0 - taudt_CC[0]) * (1.0 - albd_CC[0]) * St_dif_vis_ground
         + (1.0 - taudt_CC[1]) * (1.0 - albd_CC[1]) * St_dif_nir_ground)

Sn_S = (taubt_CC[0] * (1.0 - rho_vis_soil) * St_dir_vis_ground
        + taubt_CC[1] * (1.0 - rho_nir_soil) * St_dir_nir_ground
        + taudt_CC[0] * (1.0 - rho_vis_soil) * St_dif_vis_ground
        + taudt_CC[1] * (1.0 - rho_nir_soil) * St_dif_nir_ground)

Sn_CC = np.asarray(Sn_CC)
Sn_S = np.asarray(Sn_S)
Sn_G = Sn_CC + Sn_S
G = Sn_S * 0.35
AE = Sn_G + Ln_G

Rn_T = Sn_T + Ln_T

canopy_alb_vis = (1 - difvis) * albb_T[0] + difvis * albd_T[0]
canopy_alb_nir = (1 - difnir) * albb_T[1] + difnir * albd_T[1]
grd_alb_vis = (1 - difvis) * albb_ground[0] + difvis * albd_ground[0]
grd_alb_nir = (1 - difnir) * albb_ground[1] + difnir * albd_ground[1]

import seaborn as sns
sns.set_context('paper')
color = '0.3'
fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
ax00 = axs[0]
ax01 = ax00.twinx()

ax00.plot(LAI_CC, grd_alb_vis, label='$ρ_{vis, G}$', color=color, linestyle='-', marker='^',
          markerfacecolor='white'
          )
ax00.plot(LAI_CC, canopy_alb_vis, label='$ρ_{vis, T}$', color=color, linestyle='-', marker='s',
          markerfacecolor='white'
          )
ax01.plot(LAI_CC, grd_alb_nir, label='$ρ_{NIR, G}$', color=color, linestyle='-', marker='^',
         # markerfacecolor='white'
         )
ax01.plot(LAI_CC, canopy_alb_nir, label='$ρ_{NIR, T}$', color=color, linestyle='-', marker='s',
         # markerfacecolor='white'
         )
ax00.set_ylabel('$ρ_{vis}$')
ax01.set_ylabel('$ρ_{NIR}$')
ax00.set_xlabel('LAI$_{CC}$')
ax00.set_ylim(0, 0.175)
ax01.set_ylim(0.36, 0.405)

# Get handles and labels from both axes
handles1, labels1 = ax00.get_legend_handles_labels()
handles2, labels2 = ax01.get_legend_handles_labels()
#
# # Combine the handles and labels lists
all_handles = handles1 + handles2
all_labels = labels1 + labels2
#
# # Create a single legend on ax1 (or ax2, or fig)
ax00.legend(all_handles, all_labels, loc=0)

canopy_tau_vis = (1 - difvis) * taubt_T[0] + difvis * taudt_T[0]
canopy_tau_nir = (1 - difnir) * taubt_T[1] + difnir * taudt_T[1]
grd_tau_vis = (1 - difvis) * taubt_CC[0] + difvis * taudt_CC[0]
grd_tau_nir = (1 - difnir) * taubt_CC[1] + difnir * taubt_CC[1]

color = '0.3'
ax10 = axs[1]
ax11 = ax10.twinx()

ax10.plot(LAI_CC, grd_tau_vis, label='$τ_{vis, G}$', color=color, linestyle='-', marker='^',
          markerfacecolor='white'
          )
ax10.plot(LAI_CC, canopy_tau_vis, label='$τ_{vis, T}$', color=color, linestyle='-', marker='s',
          markerfacecolor='white'
          )
ax11.plot(LAI_CC, grd_tau_nir, label='$τ_{NIR, G}$', color=color, linestyle='-', marker='^',
         # markerfacecolor='white'
         )
ax11.plot(LAI_CC, canopy_tau_nir, label='$τ_{NIR, T}$', color=color, linestyle='-', marker='s',
         # markerfacecolor='white'
         )

handles1, labels1 = ax10.get_legend_handles_labels()
handles2, labels2 = ax11.get_legend_handles_labels()
#
# # Combine the handles and labels lists
all_handles = handles1 + handles2
all_labels = labels1 + labels2
#
# # Create a single legend on ax1 (or ax2, or fig)
ax10.legend(all_handles, all_labels, loc=0)
ax10.set_ylabel('$τ_{vis}$')
ax11.set_ylabel('$τ_{NIR}$')
ax10.set_xlabel('LAI$_{CC}$')
plt.tight_layout()
plt.savefig('files/tau_alb_LAIcc.png', dpi=300)

color = '0.3'
fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
ax1 = axs[0]
ax2 = axs[1]
# ax2 = ax1.twinx()

# ax1.plot(LAI_CC, Sn_S1, label='S$_{n,S}$', color=color, linestyle='-',  marker='^',
         # markerfacecolor='white'
         # )
ax1.plot(LAI_CC, Sn_S, label='S$_{n,S}$', color=color, linestyle='-', marker='^',
         markerfacecolor='white'
         )
ax1.plot(LAI_CC, Sn_CC, label='S$_{n,CC}$', color=color, linestyle='-',  marker='s',
         markerfacecolor='white'
         )
ax2.plot(LAI_CC, Sn_G, label='S$_{n,G}$', color=color, linestyle='-', marker='o',
         # markerfacecolor='white'
         )
ax2.plot(LAI_CC, Sn_T, label='S$_{n,T}$', color=color, linestyle='-', marker='D',
         # markerfacecolor='white'
         )
# _ = set_legend()
# handles1, labels1 = ax1.get_legend_handles_labels()
# handles2, labels2 = ax2.get_legend_handles_labels()
#
# Combine the handles and labels lists
# all_handles = handles1 + handles2
# all_labels = labels1 + labels2
# ax1.legend(all_handles, all_labels, loc=0)
ax1.set_ylabel('S$_{n} \ (W \ m^{-2})$')
ax2.set_ylabel('S$_{n} \ (W \ m^{-2})$')
ax1.set_xlabel('LAI$_{CC}$')
ax2.set_xlabel('LAI$_{CC}$')
ax1.legend(loc=0)
ax2.legend(loc=0)
plt.tight_layout()
# plt.savefig('files/Sn_LAIcc.png', dpi=300)
plt.show()

