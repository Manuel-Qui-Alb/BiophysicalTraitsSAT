# Cover crop Shortwave Tranmittance Models
import numpy as np
import TSEB.functions as myTSEB
import pvlib
import pandas as pd
import pyTSEB.TSEB as TSEB
import GeneralFunctions as GF
from pyTSEB import net_radiation as rad
from matplotlib import pyplot as plt
import pyTSEB.meteo_utils as met
from pyTSEB import MO_similarity as MO
from pyTSEB import resistances as res
import py3seb
from pyTSEB import clumping_index as CI

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

const_L = None

Trad_var = np.arange(0, 5, 0.1)

# LAI_CC = np.arange(0, 10, 1).astype(float)
LAI_CC = np.full_like(Trad_var, 5)
LAI_T = np.full_like(LAI_CC, 3)
# LAI_T = np.random.uniform(2.9, 3, 1000)
Tair = np.full_like(LAI_CC, 301)
# Tair = np.random.uniform(301, 302, 1000)
# Trad_var = np.full_like(LAI_CC, 4)
Trad = Tair + Trad_var
# Trad_T_var = np.random.uniform(0, 0.1, 1000)
# Trad_G_var = np.arange(0, 4, 0.4)[::-1]
# Trad_G_var = np.random.uniform(1, 1.01, 1000)
# Trad_S_var = np.random.uniform(, 2, 1000)

u = np.full_like(LAI_CC, 2)
ea = np.full_like(LAI_CC, 20)
# ea = np.random.uniform(20, 21, 1000)

doy = np.full_like(LAI_CC, 179)
# doy = np.random.uniform(179, 180, 1000)
hour = np.full_like(LAI_CC, 12)
# hour = np.random.uniform(12, 13, 1000)

z_u = np.full_like(LAI_CC, 2)
z_T = np.full_like(LAI_CC, 2)

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

K_be = GF.estimate_Kbe(x_LAD=1, sza=0)

fv_CC = (1 - np.exp(-K_be * LAI_CC))
fv_T = (1 - np.exp(-K_be * LAI_T))

h_T = np.full_like(LAI_T, 2)
h_CC = np.full_like(LAI_T, 0.2)

w_T = np.full_like(LAI_T, 1)
w_CC = np.full_like(LAI_T, 1)

########################################################################################################
##################################### Add clumping index #####################################
########################################################################################################
omega0_T = CI.calc_omega0_Kustas(LAI_T, fv_T, x_LAD=x_LAD, isLAIeff=True)
Omega0 = np.full_like(LAI_T, 1)

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

T_T = np.min([Tair, Trad], axis=0)

T_G = myTSEB.estimate_Trad_S(
    Trad=Trad,
    Trad_V=T_T,
    f_theta=fv_T
)

T_CC = (T_G + Tair) / 2
T_S = myTSEB.estimate_Trad_S(
    Trad=T_G,
    Trad_V=T_CC,
    f_theta=fv_CC
)

L_inc = rad.calc_longwave_irradiance(
    ea,
    Tair,
    p=1013,
    z_T=2,
    h_C=2
)


# Rn_T = Sn_T + Ln_T
# Rn_G = Sn_G + Ln_G

P_atm = np.full_like(LAI_T, 1013)
alpha_PT = np.full_like(LAI_T, 1.26)
c_p = met.calc_c_p(P_atm, ea)

########################################################################################################################
# R_x = boundary layer resistance of the complete canopy of leaves
# R_S = soil-surface resistance
# R_A = aerodynamic resistance above the canopy
########################################################################################################################
h_T = np.full_like(LAI_T, 2)
w_T = np.full_like(LAI_T, 1)
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
    landcover=np.full_like(LAI_CC, TSEB.res.GRASS),
    f_c=fv_CC
)
d_0_CC[d_0_CC < 0] = 0
z_0m_CC[z_0m_CC < np.min(z0_soil)] = np.mean(z0_soil)

KB_1_DEFAULTC = 0
z_0H_T = res.calc_z_0H(
    z_0m_T,
    kB=KB_1_DEFAULTC)

z_0H_CC = res.calc_z_0H(
    z_0m_CC,
    kB=KB_1_DEFAULTC)

rho = met.calc_rho(
    P_atm,
    ea,
    Tair
)
T_AC = Tair.copy()

u_friction = MO.calc_u_star(
    u,
    z_u,
    L,
    d_0_T,
    z_0m_T
)
U_FRICTION_MIN = np.full_like(LAI_T, 0.01)

u_friction = np.asarray(np.nanmax([U_FRICTION_MIN, u_friction], axis=0), dtype=np.float32)

resistance_form_ov = [0, {}]
res_params_ov = resistance_form_ov[1]
# KUSTAS_NORMAN_1999 (0), CHOUDHURY_MONTEITH_1988 (1), MCNAUGHTON_VANDERHURK (2), CHOUDHURY_MONTEITH_ALPHA_1988(3)
resistance_form_ov = 0

resistance_form_un = 1

F_T = np.asarray(LAI_T / fv_T, dtype=np.float32)
F_CC = np.asarray(LAI_CC / fv_CC, dtype=np.float32)

leaf_width_T = np.full_like(LAI_T, 0.05)
leaf_width_CC = np.full_like(LAI_T, 0.05)
massman_profile = [0, []]


[LE_T, LE_G, LE_CC, LE_S, LE, H_T, H_G, H_S, H, H_CC,
 Ln_T, Ln_G, Ln_CC, Ln_S, Rn_T, Rn_G, Rn_CC, Rn_S,
 T_AC_CC, G, taudt_Ldn,
 R_A, R_x, R_S_G, R_x_G, R_S, AELE_S] = [np.full_like(LAI_T, -9999) for i in range(27)]

j1 = np.full_like(LAI_T, True, dtype=bool)
j2 = np.full_like(LAI_T, True, dtype=bool)
max_iterations = 14
iterations = 1
# alpha_condition = np.any(alpha_PT > 0)
# flag = np.full_like(LAI_T, FLAG_OK)
# Rn_V0_copy, Rn_S0_copy = Rn_V0.copy(), Rn_S0.copy()
while (np.any(j1) and iterations <= max_iterations):

    R_A[j1], _, R_S_G[j1] = TSEB.calc_resistances(
        resistance_form_ov,
        {
            "R_A": {
                "z_T": z_T[j1],
                "u_friction": u_friction[j1],
                "L": L[j1],
                "d_0": d_0_T[j1],
                "z_0H": z_0H_T[j1],
            },
            "R_S": {
                "u_friction": u_friction[j1],
                "h_C": h_T[j1],
                "d_0": d_0_T[j1],
                "z_0M": z_0m_T[j1],
                "L": L[j1],
                "F": F_T[j1],
                "omega0": omega0_T[j1],
                "LAI": LAI_T[j1],
                "leaf_width": leaf_width_T[j1],
                "z0_soil": z0_soil[j1],
                "z_u": z_u[j1],
                "deltaT": T_G[j1] - T_T[j1],
                "u": u[j1],
                "rho": rho[j1],
                "c_p": c_p[j1],
                "f_cover": fv_T[j1],
                "w_C": w_T[j1] / h_T[j1],
                "massman_profile": massman_profile,
                "res_params": {k: res_params_ov[k] for k in res_params_ov.keys()},
            },
        },
    )

    Ln_T[j1], Ln_G[j1] = rad.calc_L_n_Campbell(
        T_C=T_T[j1],
        T_S=T_G[j1],
        L_dn=L_inc[j1],
        lai=LAI_T[j1],
        emisVeg=emis_leaf[j1],
        emisGrd=emis_soil[j1],
        x_LAD=x_LAD[j1]
    )

    Rn_T[j1] = Sn_T[j1] + Ln_T[j1]
    Rn_G[j1] = Sn_G[j1] + Ln_G[j1]

    # apply Cambpell longwave transmittance on substrate
    # (calculate sub-canopy LW and soil LW radiation)
    # _, _, _, taudt_Ldn[loop_con_canopy] = rad.calc_spectra_Cambpell(LAI_CC[loop_con_canopy],
    #                                             0,
    #                                             1 - emis_leaf[loop_con_canopy],
    #                                             0,
    #                                             1 - emis_soil[loop_con_canopy],
    #                                             x_lad=x_LAD[loop_con_canopy])
    #
    # Ln_S[loop_con_canopy] = Ln_G[loop_con_canopy] * taudt_Ldn[loop_con_canopy]
    # Ln_CC[loop_con_canopy] = Ln_G[loop_con_canopy] - Ln_S[loop_con_canopy]
    #
    # Rn_CC[loop_con_canopy] = Sn_CC[loop_con_canopy] + Ln_CC[loop_con_canopy]
    # Rn_S[loop_con_canopy] = Sn_S[loop_con_canopy] + Ln_S[loop_con_canopy]

    LE_T[j1] = myTSEB.Priestly_Taylor_LE_V(
        fv_g=1,  # np.full_like(f_theta[loop_con], 1),
        Rn_V=Rn_T[j1],  # This change after every loop
        alpha_PT=alpha_PT[j1],
        Tair=Tair[j1],
        P_atm=P_atm[j1],
        c_p=c_p[j1]
    )

    H_T[j1] = Rn_T[j1] - LE_T[j1]
    # print(H_T[loop_con_canopy])
    # get primary canopy temperature with parallel approach, inversion of equation 14 in Norman 1995
    T_T[j1] = ((H_T[j1] * R_A[j1]) /
               (rho[j1] * c_p[j1])) + Tair[j1]

    T_G[j1] = myTSEB.estimate_Trad_S(
        Trad=Trad[j1],
        Trad_V=T_T[j1],
        f_theta=fv_T[j1]
    )

    #You have to change the transmittance and albedo estimates within this code
    Ln_T[j1], Ln_G[j1] = rad.calc_L_n_Campbell(
        T_C=T_T[j1],
        T_S=T_G[j1],
        L_dn=L_inc[j1],
        lai=LAI_T[j1],
        emisVeg=emis_leaf[j1],
        emisGrd=emis_soil[j1],
        x_LAD=x_LAD[j1]
    )

    Rn_T[j1] = Sn_T[j1] + Ln_T[j1]
    Rn_G[j1] = Sn_G[j1] + Ln_G[j1]

    # print(rf'Alpha: {alpha_PT}, Trad_S: {Trad_S}, Trad_V: {Trad_V}')
    # print(Trad_S)
    ########################################################################################################################
    # Reestimate Soil Sensible Heat Flux (H_S) because it depends on Trad_S
    ########################################################################################################################
    _, _, R_S_G[j1] = TSEB.calc_resistances(
        resistance_form_ov,
        {
            "R_S": {
                "u_friction": u_friction[j1],
                "h_C": h_T[j1],
                "d_0": d_0_T[j1],
                "z_0M": z_0m_T[j1],
                "L": L[j1],
                "F": F_T[j1],
                "omega0": omega0_T[j1],
                "LAI": LAI_T[j1],
                "leaf_width": leaf_width_T[j1],
                "z0_soil": z0_soil[j1],
                "z_u": z_u[j1],
                "deltaT": T_G[j1] - T_T[j1],
                "u": u[j1],
                "rho": rho[j1],
                "c_p": c_p[j1],
                "f_cover": fv_T[j1],
                "w_C": w_T[j1] / h_T[j1],
                "massman_profile": massman_profile,
                "res_params": {k: res_params_ov[k] for k in res_params_ov.keys()},
            }
        }
    )

    # 3) Soil sensible heat flux H_S, array form
    H_G[j1] = (rho[j1] * c_p[j1] *
               (T_G[j1] - Tair[j1])) / (R_S_G[j1] + R_A[j1])

    G[j1] = Rn_G[j1] * 0.05
    LE_G[j1] = Rn_G[j1] - H_G[j1] - G[j1]
    LE_T[j1] = Rn_T[j1] - H_T[j1]

    H[j1] = H_T[j1] + H_G[j1]
    LE[j1] = LE_T[j1] + LE_G[j1]

    if const_L is None:
        L[j1] = MO.calc_L(
            u_friction[j1],
            Tair[j1],
            rho[j1],
            c_p[j1],
            H[j1],
            LE[j1])
        # Calculate again the friction velocity with the new stability
        # correctios
        u_friction[j1] = MO.calc_u_star(
            u[j1], z_u[j1], L[j1], d_0_T[j1], z_0m_T[j1])
        u_friction[j1] = np.asarray(np.maximum(U_FRICTION_MIN[j1], u_friction[j1]), dtype=np.float32)


    con_LE_G = (LE_G < 0)
    j1 = (con_LE_G) & (alpha_PT > 0)
    alpha_PT[j1] = np.maximum(alpha_PT[j1] - 0.1, 0.0)
    iterations += 1

iterations = 1

T_AC_CC = Tair.copy()

alpha_CC = np.full_like(LAI_CC, 1.26)
while np.any(j2):
    R_A[j2], R_x[j2], R_S[j2] = TSEB.calc_resistances(
        resistance_form_un,
        {
            "R_A": {
                "z_T": z_T[j2],
                "u_friction": u_friction[j2],
                "L": L[j2],
                "d_0": d_0_CC[j2],
                "z_0H": z_0H_CC[j2],
            },
                "R_x": {
                    "u_friction": u_friction[j2],
                    "h_C": h_CC[j2],
                    "d_0": d_0_CC[j2],
                    "z_0M": z_0m_CC[j2],
                    "L": L[j2],
                    "F": F_CC[j2],
                    "LAI": LAI_CC[j2],
                    "leaf_width": leaf_width_CC[j2],
                    "z0_soil": z0_soil[j2],
                    "massman_profile": massman_profile,
                    "res_params": {k: res_params_ov[k] for k in res_params_ov.keys()},
                },
                "R_S": {
                    "u_friction": u_friction[j2],
                    "h_C": h_CC[j2],
                    "d_0": d_0_CC[j2],
                    "z_0M": z_0m_CC[j2],
                    "L": L[j2],
                    "F": F_CC[j2],
                    "omega0": Omega0[j2],
                    "LAI": LAI_CC[j2],
                    "leaf_width": leaf_width_CC[j2],
                    "z0_soil": z0_soil[j2],
                    "z_u": z_u[j2],
                    "deltaT": T_S[j2] - T_CC[j2],
                    "u": u[j2],
                    "rho": rho[j2],
                    "c_p": c_p[j2],
                    "f_cover": fv_CC[j2],
                    "w_C": np.full_like(LAI_CC, 1),
                    "massman_profile": massman_profile,
                    "res_params": {k: res_params_ov[k] for k in res_params_ov.keys()},
                },
            },
        )

    Ln_CC[j2], Ln_S[j2] = rad.calc_L_n_Campbell(
        T_C=T_CC[j2],
        T_S=T_S[j2],
        L_dn=L_inc[j2],
        lai=LAI_CC[j2],
        emisVeg=emis_leaf[j2],
        emisGrd=emis_soil[j2],
        x_LAD=x_LAD[j2]
    )

    Rn_CC[j2] = Sn_CC[j2] + Ln_CC[j2]
    Rn_S[j2] = Sn_S[j2] + Ln_S[j2]

    LE_CC[j2] = myTSEB.Priestly_Taylor_LE_V(
            fv_g=1, #np.full_like(f_theta[loop_con], 1),
            Rn_V=Rn_CC[j2], # This change after every loop
            alpha_PT=alpha_CC[j2],
            Tair=Tair[j2],
            P_atm=P_atm[j2],
            c_p=c_p[j2]
    )

    H_CC[j2] = Rn_CC[j2] - LE_CC[j2]
    T_CC[j2] = TSEB.calc_T_C_series(
            Tr_K=T_G[j2],
            T_A_K=Tair[j2],
            R_A=R_A[j2],
            R_x=R_x[j2],
            R_S=R_S[j2],
            f_theta=fv_CC[j2],
            H_C=H_CC[j2],
            rho=rho[j2],
            c_p=c_p[j2]
        )

    T_S[j2] = myTSEB.estimate_Trad_S(
            Trad=T_G[j2],
            Trad_V=T_CC[j2],
            f_theta=fv_CC[j2]
        )

    Ln_CC[j2], Ln_S[j2] = rad.calc_L_n_Campbell(
            T_C=T_CC[j2],
            T_S=T_S[j2],
            L_dn=L_inc[j2],
            lai=LAI_CC[j2],
            emisVeg=emis_leaf[j2],
            emisGrd=emis_soil[j2],
            x_LAD=x_LAD[j2]
        )

    Rn_CC[j2] = Sn_CC[j2] + Ln_CC[j2]
    Rn_S[j2] = Sn_S[j2] + Ln_S[j2]

    _, _, R_S[j2] = TSEB.calc_resistances(
        resistance_form_un,
            {
                "R_S": {
                    "u_friction": u_friction[j2],
                    "h_C": h_CC[j2],
                    "d_0": d_0_CC[j2],
                    "z_0M": z_0m_CC[j2],
                    "L": L[j2],
                    "F": F_CC[j2],
                    "omega0": Omega0[j2],
                    "LAI": LAI_CC[j2],
                    "leaf_width": leaf_width_CC[j2],
                    "z0_soil": z0_soil[j2],
                    "z_u": z_u[j2],
                    "deltaT": T_S[j2] - T_CC[j2],
                    "u": u[j2],
                    "rho": rho[j2],
                    "c_p": c_p[j2],
                    "f_cover": fv_CC[j2],
                    "w_C": np.full_like(LAI_CC, 1),
                    "massman_profile": massman_profile,
                    "res_params": {k: res_params_ov[k] for k in res_params_ov.keys()},
                },
            },
        )

    # 2) Air temperature at canopy interface (T_AC), array form
    T_AC_CC[j2] = (
            (Tair[j2] / R_A[j2] + T_S[j2] / R_S[j2] + T_CC[j2] / R_x[j2]) /
            (1.0 / R_A[j2] + 1.0 / R_S[j2] + 1.0 / R_x[j2])
    )

    # 3) Soil sensible heat flux H_S, array form
    H_S[j2] = rho[j2] * c_p[j2] * (T_S[j2] - T_AC_CC[j2]) / R_S[j2]
    H_CC[j2] = rho[j2] * c_p[j2] * (T_CC[j2] - T_AC_CC[j2]) / R_x[j2]
    H_G[j2] = H_CC[j2] + H_S[j2]

    G[j2] = 0.3 * Rn_S[j2]
    LE_S[j2] = Rn_S[j2] - G[j2] - H_S[j2]
    LE_CC[j2] = Rn_CC[j2] - H_CC[j2]
    LE_G[j2] = LE_CC[j2] + LE_S[j2]


    # alpha_condition = alpha_PT > 0
    AELE_S[j2] = (1.0 - 0.3) * Rn_S[j2]

    con_LE_S = (LE_S < 0)
    con_AE_pos = (AELE_S > 0)
    con_AE_neg = (AELE_S < 0)

    # Flag physically negative available energy
    # flag[con_AE_neg] = FLAG_AELES_NEGATIVE

    j2 = con_LE_S & con_AE_pos & (alpha_CC > 0)
    alpha_CC[j2] = np.maximum(alpha_CC[j2] - 0.1, 0.0)

    if const_L is None:
        L[j2] = MO.calc_L(
            u_friction[j2],
            Tair[j2],
            rho[j2],
            c_p[j2],
            H_G[j2],
            LE_G[j2])
        # Calculate again the friction velocity with the new stability
        # correctios
        u_friction[j2] = MO.calc_u_star(
            u[j2], z_u[j2], L[j2], d_0_CC[j2], z_0m_CC[j2])
        u_friction[j2] = np.asarray(np.maximum(U_FRICTION_MIN[j2], u_friction[j2]), dtype=np.float32)

    # only evaluate where AE_S > 0 to avoid nonsense division

    iterations += 1
    if iterations == 15:
        print(LE_S)

# plt.plot(LAI_CC, Rn_G, c="brown", linestyle="dashed")
# plt.plot(LAI_CC, Sn_T, c="green", linestyle="dashed")
# plt.plot(LAI_CC, Ln_T, c="green", linestyle="dashed")
plt.plot(Trad_var, LE_T, c="green", linestyle="dashed", marker="o")
plt.plot(Trad_var, LE_CC, c="blue", linestyle="dashed", marker="o")
plt.plot(Trad_var, LE_S, c="brown", linestyle="dashed", marker="o")
# plt.plot(Trad_var, LE_S/(Rn_T + Rn_G), c="brown", linestyle="dashed", marker="o")
# plt.plot(LAI_CC, LE_G, c="brown")
# plt.plot(LAI_CC, LE_T, c="green")
plt.show()
    # AE = Sn_G + Ln_G
    #
    # Rn_T = Sn_T + Ln_T
