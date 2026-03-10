from py3seb import py3seb
import numpy as np
import pandas as pd
import pvlib
import pyTSEB.TSEB as TSEB
from pyTSEB import net_radiation as rad
import GeneralFunctions as GF
from pyTSEB import resistances as res


Trad_var = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
LAI_CC = np.full_like(Trad_var, 4)

# LAI_CC = np.array([0.1, 0.5, 1, 1.5, 2, 2.5, 3])
# Trad_var = np.full_like(LAI_CC, 2)

# LAI_CC = np.full_like(Trad_var, 3)
LAI_T = np.full_like(LAI_CC, 3)
# LAI_T = np.random.uniform(2.9, 3, 1000)
K_be = GF.estimate_Kbe(x_LAD=1, sza=0)
fv_CC = (1 - np.exp(-K_be * 0.8 * LAI_CC))
fv_T = (1 - np.exp(-K_be * 0.8 * LAI_T))
h_T = np.full_like(LAI_T, 2)
w_T = np.full_like(LAI_T, 1)

h_CC = np.full_like(LAI_T, 0.2)

Tair = np.full_like(LAI_CC, 301)
# Tair = np.random.uniform(301, 302, 1000)

Trad = Tair + Trad_var

u = np.full_like(LAI_CC, 2)
ea = np.full_like(LAI_CC, 20)
P_atm = np.full_like(LAI_T, 1013)

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

x_LAD = np.full_like(LAI_CC, 1)

doy = np.full_like(LAI_CC, 179)
hour = np.full_like(LAI_CC, 12)
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


times = solar_doy_hour_to_times(doy, hour, year=2024, lon_deg=site.longitude, tz=site.tz)
solpos = site.get_solarposition(times)
sza_degrees = solpos.zenith.values
saa_degrees = solpos.azimuth.values

irradiance_cs = site.get_clearsky(times)
St = irradiance_cs.ghi.values

########################################################################################################
##################################### Add clumping index #####################################
########################################################################################################
omega0_T = TSEB.CI.calc_omega0_Kustas(
    LAI_T,
    fv_T,
    x_LAD=x_LAD,
    isLAIeff=False
)

omega_SZA_T = TSEB.CI.calc_omega_Kustas(
    omega0_T,
    theta=sza_degrees,
    w_C=w_T / h_T
)

LAI_eff_T = LAI_T * omega_SZA_T
# omega0_un = CI.calc_omega0_Kustas(LAI_ov, fv_ov, x_LAD=x_LAD, isLAIeff=True)
omega0_CC = np.full_like(LAI_CC, 0.8)
LAI_eff_CC = LAI_CC * omega0_CC

difvis, difnir, fvis, fnir = TSEB.rad.calc_difuse_ratio(
    S_dn=St,
    sza=sza_degrees,
    press=np.full_like(LAI_T, 1013)
)

skyl = fvis * difvis + fnir * difnir
St_dir = (1. - skyl) * St
St_dif = skyl * St

albb_ground, albd_ground, taubt_CC, taudt_CC = rad.calc_spectra_Cambpell(
    lai=LAI_CC,
    sza=sza_degrees,
    rho_leaf=np.array((rho_vis_leaf, rho_nir_leaf)),
    tau_leaf=np.array((tau_vis_leaf, tau_nir_leaf)),
    rho_soil=rho_soil,
    x_lad=x_LAD,
    lai_eff=LAI_eff_CC)


_, _, taubt_T, taudt_T = rad.calc_spectra_Cambpell(LAI_T,
                                                   sza_degrees,
                                                   np.array((rho_vis_leaf, rho_nir_leaf)),
                                                   np.array((tau_vis_leaf, tau_nir_leaf)),
                                                   albb_ground,
                                                   x_lad=x_LAD,
                                                   lai_eff=LAI_eff_T)

Lt_T = np.nan_to_num(LAI_eff_T / fv_T)
# Lt_T = np.full_like(Lt_T, 5)
albb_T, albd_T, _, _ = rad.calc_spectra_Cambpell(Lt_T,
                                                 sza_degrees,
                                                 np.array((rho_vis_leaf, rho_nir_leaf)),
                                                 np.array((tau_vis_leaf, tau_nir_leaf)),
                                                 albb_ground,
                                                 x_lad=x_LAD,
                                                 lai_eff=Lt_T)

Lt_CC = np.nan_to_num(LAI_eff_CC / fv_CC)
albb_CC, albd_CC, _, _ = rad.calc_spectra_Cambpell(Lt_CC,
                                                   sza_degrees,
                                                   np.array((rho_vis_leaf, rho_nir_leaf)),
                                                   np.array((tau_vis_leaf, tau_nir_leaf)),
                                                   rho_soil,
                                                   x_lad=x_LAD,
                                                   lai_eff=Lt_CC)

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
#
L_inc = rad.calc_longwave_irradiance(
    ea,
    Tair,
    p=1013,
    z_T=2,
    h_C=2
)

z0_soil = np.full(LAI_T.shape, 0.01)
L = np.zeros(Trad.shape) + np.inf
z_0m_T, d_0_T = TSEB.res.calc_roughness(
    LAI_T,
    h_T,
    w_T,
    np.full_like(LAI_T, TSEB.res.CROP),
    f_c=fv_T
)
d_0_T[d_0_T < 0] = 0
z_0m_T[z_0m_T < np.min(z0_soil)] = np.mean(z0_soil)

z_0m_CC, d_0_CC = TSEB.res.calc_roughness(
    LAI_CC,
    h_C=h_CC,
    w_C=np.full_like(LAI_CC, 1),
    landcover=np.full_like(LAI_T, TSEB.res.GRASS),
    f_c=fv_T
)
d_0_CC[d_0_CC < 0] = 0
z_0m_CC[z_0m_CC < np.min(z0_soil)] = np.mean(z0_soil)

# KB_1_DEFAULTC = 0
# z_0H_T = res.calc_z_0H(
#     z_0m_T,
#     kB=KB_1_DEFAULTC)


#
# Ln_T, Ln_G = rad.calc_L_n_Campbell(
#     T_C=Trad_T,
#     T_S=Trad_G,
#     L_dn=L_inc,
#     lai=LAI_T,
#     emisVeg=emis_leaf,
#     emisGrd=emis_soil,
#     x_LAD=x_LAD
# )

[flag_3seb, t_s_3seb, t_vine_3seb, t_cc_3seb, t_ac_3seb, ln_sub_3seb, ln_vine_3seb,
 ln_cc_3seb, ln_s_3seb, le_vine_3seb, h_vine_3seb, le_cc_3seb, h_cc_3seb,
 le_s_3seb, h_s_3seb, g_3seb, r_s_3seb, r_sub_3seb, r_x_3seb, r_a_3seb, u_friction_3seb,
 l_mo_3seb, n_iterations_3seb, alpha_final, alpha_final_sub] = py3seb.ThreeSEB_PT(
    Tr_K=Trad,
    vza=0,
    T_A_K=Tair,
    u=u,
    ea=ea,
    p=P_atm,
    Sn_C=Sn_T,
    Sn_S=Sn_S,
    Sn_C_sub=Sn_CC,
    L_dn=L_inc,
    LAI=LAI_T,
    LAI_sub=LAI_CC,
    h_C=h_T,
    h_C_sub=h_CC,
    emis_C=emis_leaf,
    emis_sub=emis_leaf,
    emis_S=emis_soil,
    z_0M=z_0m_T,
    z_0M_sub=z_0m_CC,
    d_0=d_0_T,
    d_0_sub=d_0_CC,
    z_u=2,
    z_T=2,
    leaf_width=0.01,
    leaf_width_sub=0.01,
    z0_soil=0.01,
    alpha_PT=1.26,
    x_LAD=1,
    x_LAD_sub=1,
    f_c=fv_T,
    f_c_sub=fv_CC,
    f_g=1.0,
    f_g_sub=1.0,
    w_C=w_T/h_T,
    w_C_sub=1.0,
    resistance_form=[0, {}],
    calcG_params=[[1],0.35],
    massman_profile=[0.0, []],
    const_L=None)

from matplotlib import pyplot as plt

plt.plot(Trad_var, t_s_3seb,  marker="o", label="le_s_3seb")
plt.plot(Trad_var, t_cc_3seb, marker="o", label="le_cc_3seb")
plt.plot(Trad_var, t_vine_3seb, marker="o", label="le_vine_3seb")
plt.legend()
plt.ylabel("py3SEB")
plt.xlabel("Trad_var")
plt.show()
#
# print("H_C_sub mean:", np.nanmean(h_cc_3seb))
# print("H_S mean:", np.nanmean(h_s_3seb))
# print("Rn_C_sub mean:", np.nanmean(ln_cc_3seb + Sn_CC))
# print("Rn_S mean:", np.nanmean(ln_s_3seb + Sn_S))
# print("G mean:", np.nanmean(g_3seb))
# print("T_C_sub mean:", np.nanmean(t_cc_3seb))
# print("T_S mean:", np.nanmean(t_s_3seb))
# print("T_AC mean:", np.nanmean(t_ac_3seb))
# print("R_x mean:", np.nanmean(r_x_3seb))
# print("R_S mean:", np.nanmean(r_s_3seb))

# print("f_g_sub mean:", np.mean(f_g_sub))