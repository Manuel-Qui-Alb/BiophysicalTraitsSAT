import os

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pyTSEB.TSEB import calc_F_theta_campbell
from pyTSEB import clumping_index as CI
from pyTSEB.TSEB import calc_H_C_PT
import pyTSEB.TSEB as TSEB
from pyTSEB import net_radiation as rad
from pyTSEB import resistances as res
from pyTSEB import MO_similarity as MO
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
import pyTSEB.meteo_utils as met


fig_folder = os.path.normpath(rf'C:\Users\mqalborn\Desktop\ET_3SEB\figures\Sensibility Analisys')
Tair = np.random.uniform(283, 313, 10000)
diff = np.random.uniform(0,15,10000)
Trad = Tair + diff
Trad_V_0 = np.min([Trad, Tair], axis=0)
ea = np.random.uniform(0,80,10000)
P_atm = np.random.uniform(990, 1010, 10000)
u = np.random.uniform(0, 20, 10000)
# ts = np.max(tr)
                                    
LAI = np.random.uniform(0,8,10000)
fv = np.random.uniform(0, 1, 10000)
F = np.asarray(LAI / fv, dtype=np.float32)
h_V = np.random.uniform(0, 2, 10000)
w_V = np.random.uniform(0, 1, 10000)
theta_s = np.random.uniform(0, 1,10000) #(?) between 0 to 60% does not matter. solar view angle
theta = np.random.uniform(0, 0.1,10000)  # View zenith angle
x_LAD = np.round(np.random.default_rng().uniform(0, 2,10000), 1)
Sdn = np.random.uniform(0, 1000, 10000)
# theta = 0

alpha_PT = np.full_like(LAI, 1.7) #np.random.uniform(1,1.7,10000)
emis_leaf = np.full(LAI.shape, 0.97)  # Canopy emissivity
emis_soil = np.full(LAI.shape, 0.95)  # Soil emissivity
KB_1_DEFAULT = np.full(LAI.shape, 0.0)

rho_vis_leaf = np.full(LAI.shape, 0.07)
tau_vis_leaf = np.full(LAI.shape, 0.08)
rho_nir_leaf = np.full(LAI.shape, 0.32)
tau_nir_leaf = np.full(LAI.shape, 0.33)
rho_vis_soil = np.full(LAI.shape, 0.15)
rho_nir_soil = np.full(LAI.shape, 0.25)

z0_soil = np.full(LAI.shape, 0.01)
leaf_width = np.full(LAI.shape, 0.1)
z_T = np.full(LAI.shape, 2)
z_u = np.full(LAI.shape, 2)

# None Parameters
ITERATIONS = 15
L = np.zeros(Trad.shape) + np.inf
max_iterations = ITERATIONS
massman_profile = [0, []]

G_constant = 0.35
calcG_params = [[1], G_constant]
# Estimate the beam extinction coefficient based on a ellipsoidal LAD function
# Eq. 15.4 of Campbell and Norman (1998)


########################################################################################################################
# f_theta = fraction of incident beam radiation intercepted by the plant
# f_theta correspond to the radiation available for scattering, transpiration, and photosynthesis.
########################################################################################################################
omega = estimate_omega0(LAI, fv, x_LAD, theta_s)
f_theta = estimate_f_theta(LAI, x_LAD, omega, theta=0)
c_p = met.calc_c_p(P_atm, ea)

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
Trad_S = estimate_Trad_S(Trad, Trad_V_0, f_theta)
Rn_V, Rn_S = estimate_Rn(S_dn=Sdn, theta_s=theta_s, atmPress=P_atm, LAI=LAI, x_LAD=x_LAD, omega=omega,
                         Tair=Tair, ea=ea, Trad_S=Trad_S, Trad_V=Trad_V_0,
                         rho_vis_leaf=rho_vis_leaf, rho_nir_leaf=rho_nir_leaf, tau_vis_leaf=tau_vis_leaf,
                         tau_nir_leaf=tau_nir_leaf, rho_vis_soil=rho_vis_soil, rho_nir_soil=rho_nir_soil,
                         emis_leaf=emis_leaf, emis_soil=emis_soil)

idx = np.where(Rn_V < 0)
########################################################################################################################
# Estimate Soil Sensible Heat Flux (H_S)
########################################################################################################################
_, _, R_S = TSEB.calc_resistances(
    resistance_form,
    {"R_S": {"u_friction": u_friction, "h_C": h_V, "d_0": d_0,
             "z_0M": z_0m, "L": L, "F": F, "omega0": omega,
             "LAI": LAI, "leaf_width": leaf_width,
             "z0_soil": z0_soil, "z_u": z_u,
             "deltaT": Trad_S - T_AC, "u": u, "rho": rho,
             "c_p": c_p, "f_cover": fv, "w_C": w_V,
             "massman_profile": massman_profile,
             "res_params": {k: res_params[k] for k in res_params.keys()}}
     }
)

########################################################################################################################
# Maximum Vegetation Latent Heat Flux (LE_V) and its respective Latent Heat Flux (H_V) using Priestly-Taylor
########################################################################################################################
LE_S = np.full_like(LAI, -1)
idx = np.where(LE_S < 0)[0]

[LE_V, H_V, H_S, Trad_V, G] = [np.zeros(LAI.shape, np.float32) + np.nan for i in range(5)]

while len(idx)>0:

    print(len(idx))
    alpha_PT[idx]-= 0.1
    alpha_PT[idx] = np.where(alpha_PT[idx] < 0, 0, alpha_PT[idx])

    if np.any(alpha_PT[idx] == 0):
        print("Zero found in the array! Stopping further processing.")

    # print(np.max(alpha_PT[idx]))
    LE_V[idx] = Priestly_Tayloy_max_LE_V(fv_g=f_theta[idx], Rn_V=Rn_V[idx], alpha_PT=alpha_PT[idx], Tair=Tair[idx],
                                         P_atm=P_atm[idx], c_p=c_p[idx])

    H_V[idx] = Rn_V[idx] - LE_V[idx]
    print(np.max(H_V[idx]))

    ########################################################################################################################
    # Reestimate of Trad_V and Trad_S using Sensible Heat Flux
    ########################################################################################################################
    Trad_V[idx] = TSEB.calc_T_C_series(Tr_K=Trad[idx], T_A_K=Tair[idx], R_A=R_A[idx], R_x=R_x[idx], R_S=R_S[idx],
                                  f_theta=f_theta[idx], H_C=H_V[idx], rho=rho[idx], c_p=c_p[idx])
    Trad_S[idx] = estimate_Trad_S(Trad[idx], Trad_V[idx], f_theta[idx])

    ########################################################################################################################
    # Estimate Soil Sensible Heat Flux (H_S)
    ########################################################################################################################
    _, _, R_S[idx] = TSEB.calc_resistances(
                resistance_form,
                {"R_S": {"u_friction": u_friction[idx], "h_C": h_V[idx], "d_0": d_0[idx],
                         "z_0M": z_0m[idx], "L": L[idx], "F": F[idx], "omega0": omega[idx],
                         "LAI": LAI[idx], "leaf_width": leaf_width[idx],
                         "z0_soil": z0_soil[idx], "z_u": z_u[idx],
                         "deltaT": Trad_S[idx] - T_AC[idx], "u": u[idx], "rho": rho[idx],
                         "c_p": c_p[idx], "f_cover": fv[idx], "w_C": w_V[idx],
                         "massman_profile": massman_profile,
                         "res_params": {k: res_params[k] for k in res_params.keys()}}
                 }
            )

    # Get air temperature at canopy interface
    T_AC[idx] = ((Tair[idx] / R_A[idx] + Trad_S[idx] / R_S[idx] + Trad_V[idx] / R_x[idx])
                       / (1.0 / R_A[idx] + 1.0 / R_S[idx] + 1.0 / R_x[idx]))

    # Calculate soil fluxes
    H_S[idx] = rho[idx] * c_p[idx] * (Trad_S[idx] - T_AC[idx]) / R_S[idx]

    ########################################################################################################################
    # Compute Soil Heat Flux Ratio as a Ratio of Rn_S
    ########################################################################################################################
    G_ratio = 0.35
    G[idx] = G_ratio * Rn_S[idx]

    ########################################################################################################################
    # Reestimate Net Radiation estimation with Trad_V and Trad_S from Sensible Heat Flux
    ########################################################################################################################
    Rn_V[idx], Rn_S[idx] = estimate_Rn(S_dn=Sdn[idx], theta_s=theta_s[idx], atmPress=P_atm[idx], LAI=LAI[idx],
                                       x_LAD=x_LAD[idx], omega=omega[idx], Tair=Tair[idx], ea=ea[idx], Trad_S=Trad_S[idx],
                                       Trad_V=Trad_V[idx], rho_vis_leaf=rho_vis_leaf[idx], rho_nir_leaf=rho_nir_leaf[idx],
                                       tau_vis_leaf=tau_vis_leaf[idx], tau_nir_leaf=tau_nir_leaf[idx],
                                       rho_vis_soil=rho_vis_soil[idx], rho_nir_soil=rho_nir_soil[idx],
                                       emis_leaf=emis_leaf[idx], emis_soil=emis_soil[idx])
    LE_S[idx] = Rn_S[idx] - G[idx] - H_S[idx]
    LE_V[idx] = Rn_V[idx] - H_V[idx]
    idx = np.where(LE_S < 0)[0]

"""
# Ln_C[i], Ln_S[i] = rad.calc_L_n_Campbell(
#                 T_C[i], T_S[i], L_dn[i], LAI[i], emis_C[i], emis_S[i], x_LAD=x_LAD[i])
# delta_Rn[i] = Sn_C[i] + Ln_C[i]
# calc_H_C_PT(delta_R_ni, f_g, T_A_K, P, c_p, alpha):

# tc , ts  = calculate_Ts(LAI, fc, x_LAD, theta_s, tr, ta)

# calculate_Ts(tr, f_theta, tc)
# plt.scatter(ta, tc)
# plt.colorbar(label='Color Value')
# plt.show()

# 1) Define the problem (edit bounds to your site/data)
problem = {
    "num_vars": 6,
    "names": ["LAI", "FracCover", "X_LAD", "SolarAngle", "Ta", "diffTr"],
    "bounds": [
        [0.0, 8.0],      # LAI
        [0.0, 1],     # Fractional cover (0–1; adjust if you use a different definition)
        [0.5, 1.5],      # X_LAD (e.g., 0.5 erectophile ... 2 planophile)
        [0, 0.6],    # Solar zenith angle (degrees)
        [10, 40],
        [-1, 15]
    ],
}

N = 5000
second_order = False
X = saltelli.sample(problem, N, calc_second_order=second_order)   # shape (nsamples, 4)

# Option A (vectorized via columns) – safest and fast
LAI, fv, x_LAD, theta_s, Tair, diffTr = X.T
# Y = calculate_f_theta(LAI, fc, x_LAD, theta_s)
Y = calculate_Trad_S(LAI, fv, x_LAD, theta_s, Tair, diffTr)
# shape (nsamples,)
# Y = np.asarray(Y).reshape(-1)                      # ensure 1-D

# Option B (row-wise loop) – slower but foolproof
# Y = np.empty(X.shape[0])
# for i, (lai, xlad, sza, fc) in enumerate(X):
#     Y[i] = calculate_f_theta(lai, fc, xlad, sza)

# Sanity checks to prevent the broadcast error
# assert Y.ndim == 1, f"Y must be 1-D; got {Y.shape}"
# assert Y.shape[0] == X.shape[0], f"len(Y)={Y.shape[0]} vs nsamples={X.shape[0]}"

Si = sobol.analyze(problem, Y, calc_second_order=second_order, print_to_console=False)

for n, s1, st, c1, ct in zip(problem["names"], Si["S1"], Si["ST"], Si["S1_conf"], Si["ST_conf"]):
    print(f"{n:>12s} | S1={s1:.3f} (±{c1:.3f})  ST={st:.3f} (±{ct:.3f})")


# fig, ax = plt.subplots(2,2, figsize = (15,10))
# sns.kdeplot(x=omega0, y=f_theta, fill=True, cmap='viridis', ax=ax[0, 0])
# sns.kdeplot(x=K_be, y=f_theta, fill=True, cmap='viridis', ax=ax[0, 1])
# sns.kdeplot(x=F, y=f_theta, fill=True, cmap='viridis', ax=ax[1, 0])
# sns.kdeplot(x=fc, y=f_theta, fill=True, cmap='viridis', ax=ax[1, 1])
# ax[0, 0].set(ylabel=r'$f_{(\theta), c}$', xlabel=r'$\Omega_{c}$')
# ax[0, 1].set(ylabel=r'$f_{(\theta), c}$', xlabel="K$_{be}$($\Theta_{S}$)")
# ax[1, 0].set(ylabel=r'$f_{(\theta), c}$', xlabel=r'LAI')
#
# plt.show()
"""

"""
# omega0 = CI.calc_omega0_Kustas(LAI, fc, x_LAD=1, isLAIeff=False)


# F = np.asarray(LAI / f_c, dtype=np.float32)
# f_theta = calc_F_theta_campbell(vza, F, w_C=w_C, Omega0=omega0, x_LAD=x_LAD)

# fig, ax = plt.subplots(3,1)
# sns.kdeplot(theta, color='blue', linewidth=2, ax=ax[0])
# sns.kdeplot(K_be, color='red', linewidth=2, ax=ax[1])
# sns.kdeplot(omega0, color='red', linewidth=2, ax=ax[2])
# sns.kdeplot(LAI, color='green', linewidth=2, ax=ax[1])
# plt.show()

sns.scatterplot(x=LAI, y=f_theta, hue=fc)
plt.plot([0, 8], [0, 1], c='grey')
plt.show()
"""

#Graphs


"""
### Graphs of extinction coefficient
sns.set_palette("Greys")
theta_s_graph = theta_s*100
sns.lineplot(x=theta_s_graph, y=K_be, hue=x_LAD, palette="Greys", legend=False
             # edgecolor="black", linewidth=0.1
             )
sns.lineplot(x=theta_s_graph[x_LAD == 1], y=K_be[x_LAD == 1], color='blue', label='Spherical', linestyle='-'
             # edgecolor="black", linewidth=0.1
             )

sns.lineplot(x=theta_s_graph[x_LAD == 0], y=K_be[x_LAD == 0], color='blue', label='Vertical', linestyle=':'
             # edgecolor="black", linewidth=0.1
             )

sns.lineplot(x=theta_s_graph[x_LAD == 2], y=K_be[x_LAD == 2], color='blue', label='Horizontal', linestyle='--'
             # edgecolor="black", linewidth=0.1
             )

plt.ylabel("K$_{be}$($\Theta_{S}$)")
plt.xlabel("$\Theta_{S}$ (°)")
plt.legend(title='Leaf Angle Distribution', loc=4)
plt.show()

# plt.savefig(os.path.join(fig_folder, "Kbe_thetaS_XE.png"), dpi=300)
"""





