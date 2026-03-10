from my3SEB import my3SEB
import numpy as np
import pvlib
import GeneralFunctions as GF


# LAI_un =
Trad_var = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
LAI_un = np.full_like(Trad_var, 4)
# LAI_un = np.full_like(Trad_var, 0)
# LAI_un[int(len(Trad_var)/2):] = 0.5
LAI_ov = np.full_like(LAI_un, 3)
Tair = np.full_like(LAI_un, 301)
Trad = Tair + Trad_var
k_fc_ov = np.full_like(LAI_un, 1)

u = np.full_like(LAI_un, 2)
ea = np.full_like(LAI_un, 20)
P_atm = np.full_like(LAI_un, 1013)

z_u = np.full_like(LAI_un, 2)
z_T = np.full_like(LAI_un, 2)

rho_vis_leaf = np.full_like(LAI_un, 0.10)
rho_nir_leaf = np.full_like(LAI_un, 0.45)
tau_vis_leaf = np.full_like(LAI_un, 0.05)
tau_nir_leaf =np.full_like(LAI_un, 0.45)

rho_vis_soil =  np.full_like(LAI_un, 0.15)
rho_nir_soil =  np.full_like(LAI_un, 0.40)

emis_leaf = np.full_like(LAI_un, 0.97)
emis_soil = np.full_like(LAI_un, 0.98)

# ------------------------------------------------------------------
# 0) Estimating S_dn from day of year
# ------------------------------------------------------------------
site = pvlib.location.Location(
    latitude=38.5449,
    longitude=-121.7405,
    tz='US/Pacific',
    altitude=18,      # meters; small effect, 15–25 m is fine
    name='Davis_CA'
)

doy = np.full_like(LAI_un, 179)
hour = np.full_like(LAI_un, 12)
times = GF.solar_doy_hour_to_times(doy, hour, year=2024, lon_deg=site.longitude, tz=site.tz)

solpos = site.get_solarposition(times)
sza_degrees = solpos.zenith.values
saa_degrees = solpos.azimuth.values

irradiance_cs = site.get_clearsky(times)
S_dn = irradiance_cs.ghi.values

K_be = GF.estimate_Kbe(x_LAD=1, sza=0)

fv_ov = (1 - np.exp(-K_be * k_fc_ov * LAI_ov))
fv_un = (1 - np.exp(-K_be * 0.8 * LAI_un))

h_ov = np.full_like(LAI_ov, 2)
h_un = np.full_like(LAI_ov, 0.2)

w_ov = np.full_like(LAI_ov, 1)
w_un = np.full_like(LAI_ov, 1)

out = my3SEB(Trad, Tair, u, ea, P_atm, S_dn, sza_degrees, LAI_ov, fv_ov, h_ov, w_ov,
           emis_leaf, emis_soil, rho_vis_leaf, rho_nir_leaf, tau_vis_leaf, tau_nir_leaf, rho_vis_soil, rho_nir_soil,
           z_u, z_T, x_LAD=1.0, leaf_width_ov=0.05, alpha_PT_ov=1.26,
           # optional understory inputs
           LAI_un=LAI_un, fv_un=fv_un, h_un=h_un, w_un=w_un, leaf_width_un=0.05, alpha_PT_un=1.0, omega0_un=0.8,
           # options
           const_L=None, max_outer=30, max_it1=20, max_it2=20, step_alpha=0.1, L_thres=0.05, scheme_ov=0
)

out_series = my3SEB(Trad, Tair, u, ea, P_atm, S_dn, sza_degrees, LAI_ov, fv_ov, h_ov, w_ov,
           emis_leaf, emis_soil, rho_vis_leaf, rho_nir_leaf, tau_vis_leaf, tau_nir_leaf, rho_vis_soil, rho_nir_soil,
           z_u, z_T, x_LAD=1.0, leaf_width_ov=0.05, alpha_PT_ov=1.26,
           # optional understory inputs
           LAI_un=LAI_un, fv_un=fv_un, h_un=h_un, w_un=w_un, leaf_width_un=0.05, alpha_PT_un=1.0, omega0_un=None,
           # options
           const_L=None, max_outer=30, max_it1=20, max_it2=20, step_alpha=0.1, L_thres=0.05, scheme_ov=1
)



# err_tot = (
#         (out["Rn_ov"] + out["Rn_un"] + out["Rn_soil"] - out["G"])
#         - (out["H_ov"] + out["H_un"] + out["H_soil"]
#            + out["LE_ov"] + out["LE_un"] + out["LE_soil"])
# )
#
# err_sub = (
#         (out["Rn_sub"] - out["G"])
#         - (out["H_sub"] + out["LE_sub"])
# )
#
# err_un = out["Rn_un"] - (out["H_un"] + out["LE_un"])
#
# err_soil = (
#         (out["Rn_soil"] - out["G"])
#         - (out["H_soil"] + out["LE_soil"]) )

# print("f_g_sub mean:", np.mean(f_g_sub))
# print("f_c_sub mean:", np.mean(f_c_sub))
# print("f_theta_sub mean:", np.mean(f_theta_sub))

from matplotlib import pyplot as plt

# plt.plot(Trad_var, out['Ln_soil'], marker="o", label="Ln_soil")
# plt.plot(Trad_var, out['Ln_un'], marker="o", label="Ln_un")
# plt.plot(Trad_var, out['Rn_ov'], marker="o", label="Rn_ov")
# plt.ylabel("Parallel")
# plt.xlabel("Trad_var")
# plt.legend()
# plt.show()

# plt.figure()
# # plt.plot(Trad_var, out_series['T_AC_un'], marker="o", label="Rn_soil")
# # plt.show()
# plt.plot(Trad_var, out['LE_soil'],  marker="o", label="le_s_3seb")
# plt.plot(Trad_var, out['LE_un'], marker="o", label="le_cc_3seb")
# plt.plot(Trad_var, out['LE_ov'], marker="o", label="le_vine_3seb")
# plt.legend()
# # plt.ylabel("Parallel")
# # plt.xlabel("Trad_var")
# plt.show()

plt.figure()
plt.plot(Trad_var, out_series['T_ov'] - Tair, marker="o", label="T_ov")
plt.plot(Trad_var, out_series['T_un'] - Tair, marker="o", label="T_un")
plt.plot(Trad_var, out_series['T_soil'] - Tair, marker="o", label="T_soil")
# plt.ylabel("Series")
plt.xlabel("Trad_var")
plt.legend()
plt.show()

plt.figure()
plt.plot(Trad_var, out_series['LE_ov'], marker="o", label="LE_ov")
plt.plot(Trad_var, out_series['LE_un'], marker="o", label="LE_un")
plt.plot(Trad_var, out_series['LE_soil'], marker="o", label="LE_soil")
# plt.ylabel("Series")
plt.xlabel("Trad_var")
plt.legend()
plt.show()

# plt.figure()
# plt.plot(Trad_var, out_series['T_AC_un'], marker="o", label="Rn_soil")
# plt.show()
# plt.plot(Trad_var, out_series['LE_soil'],  marker="o", label="le_s_3seb")
# plt.plot(Trad_var, out_series['LE_un'], marker="o", label="le_cc_3seb")
# plt.plot(Trad_var, out_series['LE_ov'], marker="o", label="le_vine_3seb")
# plt.legend()
# plt.ylabel("Series")
# plt.xlabel("Trad_var")
# plt.show()

# plt.figure()
# plt.plot(Trad_var, out_series['LE_soil'],  marker="o", label="le_s_3seb")
# plt.plot(Trad_var, out_series['LE_un'], marker="o", label="le_cc_3seb")
# plt.plot(Trad_var, out_series['LE_ov'], marker="o", label="le_vine_3seb")
# plt.legend()
# plt.ylabel("Series")
# plt.xlabel("Trad_var")
# plt.show()



# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.set_style('whitegrid')
# plt.figure()
#
# plt.plot(LAI_un, out['LE_soil'], marker="o", label="H_soil")
# plt.plot(LAI_un, out['LE_un'], marker="o", label="H_soil")
# plt.plot(LAI_un, out['LE_ov'], marker="o", label="H_soil")
# plt.plot(LAI_un, out['H_soil'], marker="o", label="H_soil")
# plt.plot(LAI_un, out['Rn_soil'], marker="o", label="Rn_soil")
# plt.plot(LAI_un[1:], out['T_soil'][1:] - out['T_AC_un'][1:], marker="o", label='T_soil - T_AC_un')

# plt.plot(LAI_un[1:], out['R_S_un'][1:], marker="o", label='R_S_un')
# plt.plot(LAI_un, out['H_soil']/ (out['Rn_soil'] + out['G']), marker="o", label="H_soil/AELE_soil")
# plt.plot(Trad_var, LE_sub, marker="o", label="LE_sub")
# plt.plot(LAI_un, out['LE_un'], marker="o", label="LE_un")
# plt.plot(Trad_var, LE, marker="o", label="LE")

# plt.xlabel('LAI_un')
# plt.ylabel('LE')
# plt.legend()
# plt.show()
# print(out)