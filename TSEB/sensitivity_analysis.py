import numpy as np
import pandas as pd

import functions as myTSEB
import pyTSEB.meteo_utils as met
from pyTSEB import resistances as res
from pyTSEB import MO_similarity as MO
import pyTSEB.TSEB as TSEB
import math
angle_degrees = 90

# Convert the angle to radians


Tair = np.random.uniform(283, 313, 10000)
diff = np.random.uniform(0,15,10000)
Trad = Tair + diff
Trad_V_0 = np.min([Trad, Tair], axis=0)

LAI = np.random.uniform(0, 8, 10000)
fv = np.random.uniform(0, 1, 10000)
F = np.asarray(LAI / fv, dtype=np.float32)
h_V = np.random.uniform(0, 2, 10000)
w_V = np.random.uniform(0, 1, 10000)

ea = np.random.uniform(0,80,10000)
P_atm = np.random.uniform(990, 1010, 10000)
u = np.random.uniform(0, 20, 10000)
c_p = met.calc_c_p(P_atm, ea)

sza_degrees = np.random.uniform(0, 90,10000) #(?) between 0 to 60% does not matter. solar zenith angle
sza_rad = np.radians(sza_degrees)

vza_degrees = np.random.uniform(0, 90,10000)  # View zenith angle
vza_rad = np.radians(vza_degrees)
x_LAD = np.round(np.random.default_rng().uniform(0, 2,10000), 1)
Sdn = np.random.uniform(0, 1000, 10000)

z_T = np.full(LAI.shape, 2)
z_u = np.full(LAI.shape, 2)

KB_1_DEFAULT = np.full(LAI.shape, 0.0)
alpha_PT = np.full_like(LAI, 1.7)

# None Parameters
ITERATIONS = 15
L = np.zeros(Trad.shape) + np.inf
max_iterations = ITERATIONS
massman_profile = [0, []]

G_constant = 0.35
calcG_params = [[1], G_constant]

(emis_leaf, emis_soil, rho_vis_leaf, tau_vis_leaf, rho_nir_leaf, tau_nir_leaf, rho_vis_soil, rho_nir_soil, z0_soil,
 leaf_width) = myTSEB.get_emiss_rho_tau(LAI)

########################################################################################################################
# f_theta = fraction of incident beam radiation intercepted by the plant
# f_theta correspond to the radiation available for scattering, transpiration, and photosynthesis.
########################################################################################################################
omega = myTSEB.estimate_omega0(LAI, fv, x_LAD, sza_rad)
f_theta = myTSEB.estimate_f_theta(LAI=LAI, x_LAD=x_LAD, omega=omega, sza=sza_rad)

########################################################################################################################
# R_x = boundary layer resistance of the complete canopy of leaves
# R_S = soil-surface resistance
# R_A = aerodynamic resistance above the canopy
########################################################################################################################
z_0m, d_0 = TSEB.res.calc_roughness(LAI,
                                    h_V,
                                    w_V,
                                    np.full_like(LAI, TSEB.res.CROP),
                                    f_c=fv)
d_0[d_0 < 0] = 0
z_0m[z_0m < np.min(z0_soil)] = np.mean(z0_soil)

KB_1_DEFAULTC = 0
z_0H = res.calc_z_0H(z_0m, kB=KB_1_DEFAULTC)

res_params = [0,{}]
rho = met.calc_rho(P_atm, ea, Tair)
T_AC = Tair.copy()

u_friction = MO.calc_u_star(u, z_u, L, d_0, z_0m)
U_FRICTION_MIN = np.full_like(LAI, 0.01)
u_friction = np.asarray(np.nanmax([U_FRICTION_MIN, u_friction], axis=0), dtype=np.float32)

resistance_form=[0,{}]
res_params = resistance_form[1]
resistance_form = 0 #KUSTAS_NORMAN_1999 (0), CHOUDHURY_MONTEITH_1988 (1), MCNAUGHTON_VANDERHURK (2), CHOUDHURY_MONTEITH_ALPHA_1988(3)

R_A, R_x, _ = TSEB.calc_resistances(
    resistance_form,
    {
        "R_A": {"z_T": z_T, "u_friction": u_friction, "L": L,
                "d_0": d_0, "z_0H": z_0H},
        "R_x": {"u_friction": u_friction, "h_C": h_V,
                "d_0": d_0, "z_0M": z_0m, "L": L, "F": F, "LAI": LAI, "leaf_width": leaf_width, "z0_soil": z0_soil,
                "massman_profile": massman_profile,
                "res_params": {k: res_params[k] for k in res_params.keys()}},
     }
)
########################################################################################################################
# Fist Net Radiation estimation assuming Trad_V = Tair
# This first estimation of Trad_S will be used to estimate Ln_S and R_S
########################################################################################################################
Trad_S = myTSEB.estimate_Trad_S(Trad=Trad, Trad_V=Trad_V_0, f_theta=f_theta)
con = Trad_S<350
Rn_V, Rn_S = myTSEB.estimate_Rn(S_dn=Sdn[con], sza=sza_degrees[con], P_atm=P_atm[con], LAI=LAI[con], x_LAD=x_LAD[con], omega=omega[con],
                                Tair=Tair[con], ea=ea[con], Trad_S=Trad_S[con], Trad_V=Trad_V_0[con],
                                rho_vis_leaf=rho_vis_leaf[con], rho_nir_leaf=rho_nir_leaf[con], tau_vis_leaf=tau_vis_leaf[con],
                                tau_nir_leaf=tau_nir_leaf[con], rho_vis_soil=rho_vis_soil[con], rho_nir_soil=rho_nir_soil[con],
                                emis_leaf=emis_leaf[con], emis_soil=emis_soil[con])


# sns.kdeplot(Sdn_dir + Sdn_dif)

# from matplotlib import pyplot as plt
# import seaborn as sns
#
# # diff = Trad_S - Tair
# # con = diff>30
# # diff = diff[con]
#
# f_theta_test = f_theta[con]
# Trad_V_0_test = Trad_V_0[con]
# Trad_test = Trad[con]
#
# pd.DataFrame({'Trad_V_0':Trad_V_0_test, 'Trad': Trad_test, 'f_theta_test': f_theta_test })
#
# sns.scatterplot(x=Trad_V_0, y=Trad, hue=diff)
# plt.hist(f_theta_test, bins=10)
# plt.plot([283, 312], [283, 312])
# plt.ylim([283, 312])
# plt.xlim([283, 312])
# plt.show()

# idx = np.where(Rn_V < 0)

########################################################################################################################
# Maximum Vegetation Latent Heat Flux (LE_V) and its respective Latent Heat Flux (H_V) using Priestly-Taylor
########################################################################################################################
# LE_S = np.full_like(LAI, -1)
# idx = np.where(LE_S < 0)[0]

[LE_V, H_V, H_S, Trad_V, G] = [np.zeros(LAI.shape, np.float32) + np.nan for i in range(5)]
