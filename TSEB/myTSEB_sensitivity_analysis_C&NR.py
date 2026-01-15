import numpy as np
import pandas as pd
import csv
import os

import functions as myTSEB
import pyTSEB.meteo_utils as met
from pyTSEB import resistances as res
from pyTSEB import MO_similarity as MO
import pyTSEB.TSEB as TSEB
from SALib.sample import saltelli
from SALib.analyze import sobol

# params = (pd.read_csv(rf'files/inputs_sensitivity_analysis.csv')
#           .drop('Unnamed: 0', axis=1))
#
#
# params = {col: params[col].to_numpy() for col in params.columns}

# None Parameters
ITERATIONS = 50
# L = np.zeros(np.array(params['LAI']).shape) + np.inf
max_iterations = ITERATIONS
massman_profile = [0, []]

G_constant = 0.1
calcG_params = [[1], G_constant]
G_ratio = 0.1

alpha_PT = 1.7
save_inputs = True
T_MODEL = 'CN_R'

########################################################################################################################
# f_theta = fraction of incident beam radiation intercepted by the plant
# f_theta correspond to the radiation available for scattering, transpiration, and photosynthesis.
########################################################################################################################
# myTSEB(LAI, fv, x_LAD, sza_rad)
docker_inputs = []

def sensitivity_analysis_myTSEB(LAI, difffv, h_V, row_sep, x_LAD,
                                leaf_width, diffTrad, Tair, Sdn, u,
                                P_atm, ea, sza_degrees, saa_degrees, G_ratio, row_azimuth,
                                emis_leaf, emis_soil, rho_vis_leaf, tau_vis_leaf,
                                rho_nir_leaf, tau_nir_leaf, rho_vis_soil, rho_nir_soil, T_MODEL='CN_H'):


    Trad = Tair + diffTrad
    z_u = h_V + 2
    z_T = h_V + 2

    sza_rad =  np.radians(sza_degrees)
    # vza_rad = np.radians(vza_degrees)
    saa_rad = np.radians(saa_degrees)
    row_azimuth_rad = np.radians(row_azimuth)

    # Fv based on LAI. fv and LAI are inherently related.
    # It was considered that the FV can vary between 30 and 1.1 with respect to models based on LAI.
    K_be = myTSEB.estimate_Kbe(x_LAD, 0)
    fv02 = np.clip((1 - np.exp(-K_be * LAI)) * difffv, 1e-6, 1)
    w_V = fv02 * row_sep
    F = np.asarray(LAI / fv02, dtype=np.float32)

    c_p = met.calc_c_p(P_atm, ea)

    alpha_PT = np.full_like(LAI,  1.6)
    L = np.zeros(Trad.shape) + np.inf

    # (emis_leaf, emis_soil, rho_vis_leaf_2, tau_vis_leaf, rho_nir_leaf_2, tau_nir_leaf, rho_vis_soil, rho_nir_soil,
    #  z0_soil) = myTSEB.get_emiss_rho_tau(LAI)
    z0_soil = np.full(LAI.shape, 0.01)

    if T_MODEL == 'CN_H':
        omega = myTSEB.off_nadir_clumpling_index_Kustas_Norman(
            LAI,
            fv02,
            h_V,
            w_V,
            x_LAD,
            sza_rad
        )
    elif T_MODEL == 'CN_R':
        omega = myTSEB.rectangular_row_clumping_index_parry(
            LAI=LAI,
            fv0=fv02,
            w_V=w_V,
            h_V=h_V,
            sza=sza_rad,
            saa=saa_rad,
            row_azimuth=row_azimuth_rad,
            hb_V=0,
            L=None,
            x_LAD=1
        )

    f_theta = myTSEB.estimate_f_theta(
        LAI=LAI,
        x_LAD=x_LAD,
        omega=omega,
        sza=sza_rad
    )

    ########################################################################################################################
    # R_x = boundary layer resistance of the complete canopy of leaves
    # R_S = soil-surface resistance
    # R_A = aerodynamic resistance above the canopy
    ########################################################################################################################
    z_0m, d_0 = TSEB.res.calc_roughness(
        LAI,
        h_V,
        w_V,
        np.full_like(LAI, TSEB.res.CROP),
        f_c=fv02
    )
    d_0[d_0 < 0] = 0
    z_0m[z_0m < np.min(z0_soil)] = np.mean(z0_soil)
    # params.update({'d_0': d_0, 'z_0m': z_0m})

    KB_1_DEFAULTC = 0
    z_0H = res.calc_z_0H(
        z_0m,
        kB=KB_1_DEFAULTC)
    # params.update({'z_0H': z_0H})

    res_params = [0,{}]
    rho = met.calc_rho(
        P_atm,
        ea,
        Tair
    )
    T_AC = Tair.copy()
    # params.update({'rho': rho, 'T_AC': T_AC})

    u_friction = MO.calc_u_star(
        u,
        z_u,
        L,
        d_0,
        z_0m
    )
    U_FRICTION_MIN = np.full_like(LAI, 0.01)
    # params.update({'u_friction': u_friction, 'U_FRICTION_MIN': U_FRICTION_MIN})

    u_friction = np.asarray(np.nanmax([U_FRICTION_MIN, u_friction], axis=0), dtype=np.float32)
    # params.update({'u_friction': u_friction})

    resistance_form=[0,{}]
    res_params = resistance_form[1]
    resistance_form = 0 #KUSTAS_NORMAN_1999 (0), CHOUDHURY_MONTEITH_1988 (1), MCNAUGHTON_VANDERHURK (2), CHOUDHURY_MONTEITH_ALPHA_1988(3)

    ########################################################################################################################
    # Fist Net Radiation estimation assuming Trad_V = Tair
    # This first estimation of Trad_S will be used to estimate R_S, Ln_S and R_S
    ########################################################################################################################
    Trad_V0 = np.min([Tair, Trad], axis=0) # We set the first Trad_V equal to Tair, under potential ET conditions.
    Trad_S_0 = myTSEB.estimate_Trad_S(
        Trad=Trad,
        Trad_V=Trad_V0,
        f_theta=f_theta
    )
    # params.update({'Trad_S_0': Trad_S_0})

    R_A, R_x, R_S = TSEB.calc_resistances(
        resistance_form,
        {
            "R_A": {
                "z_T": z_T,
                "u_friction": u_friction,
                "L": L,
                "d_0": d_0,
                "z_0H": z_0H,
            },
            "R_x": {
                "u_friction": u_friction,
                "h_C": h_V,
                "d_0": d_0,
                "z_0M": z_0m,
                "L": L,
                "F": F,
                "LAI": LAI,
                "leaf_width": leaf_width,
                "z0_soil": z0_soil,
                "massman_profile": massman_profile,
                "res_params": {k: res_params[k] for k in res_params.keys()},
            },
            "R_S": {
                "u_friction": u_friction,
                "h_C": h_V,
                "d_0": d_0,
                "z_0M": z_0m,
                "L": L,
                "F": F,
                "omega0": omega,
                "LAI": LAI,
                "leaf_width": leaf_width,
                "z0_soil": z0_soil,
                "z_u": z_u,
                "deltaT": Trad_S_0 - T_AC,
                "u": u,
                "rho": rho,
                "c_p": c_p,
                "f_cover": fv02,
                "w_C": w_V,
                "massman_profile": massman_profile,
                "res_params": {k: res_params[k] for k in res_params.keys()},
            },
        },
    )

    if np.any(np.array([R_A, R_x, R_S])<=0):
        print([R_A, R_x, R_S])

    Rn_V0, Rn_S0 = myTSEB.estimate_Rn(
        S_dn=Sdn,
        sza=sza_degrees,
        P_atm=P_atm,
        LAI=LAI,
        x_LAD=x_LAD,
        omega=omega,
        Tair=Tair,
        ea=ea,
        Trad_S=Trad_S_0,
        Trad_V=Trad_V0,
        rho_vis_leaf=rho_vis_leaf,
        rho_nir_leaf=rho_nir_leaf,
        tau_vis_leaf=tau_vis_leaf,
        tau_nir_leaf=tau_nir_leaf,
        rho_vis_soil=rho_vis_soil,
        rho_nir_soil=rho_nir_soil,
        emis_leaf=emis_leaf,
        emis_soil=emis_soil
    )

    # Rn_V0 = np.array(np.clip(Rn_V0, -150, 1000))
    # Rn_S0 = np.array(np.clip(Rn_S0, -150, 1000))

    Rn_V0_copy = Rn_V0.copy()
    Rn_S0_copy = Rn_S0.copy()

    [LE_V, LE_S, H_V, H_S, G, Trad_V, Trad_S] = [np.full_like(LAI, -9999) for i in range(7)]

    loop_con = True
    max_iterations = 50
    iterations = 1
    alpha_condition = np.any(alpha_PT > 0)

    while (loop_con and alpha_condition and iterations < max_iterations) :
        # print(iterations)
        # while len(loop_con)>0:
        # print(loop_con))
        # print('Iteration', iterations)

        # alpha_PT = np.where(alpha_PT < 0, 0, alpha_PT)

        if iterations > 1:
            Rn_V0, Rn_S0 = Rn_V, Rn_S

        LE_V = myTSEB.Priestly_Taylor_LE_V(
            fv_g=f_theta,
            Rn_V=Rn_V0,
            alpha_PT=alpha_PT,
            Tair=Tair,
            P_atm=P_atm,
            c_p=c_p
        )


        H_V = Rn_V0 - LE_V
        # print(rf'Alpha: {alpha_PT}, Rn_V0: {Rn_V0}, H_V: {H_V}')

        # print(alpha_PT)
        # print(LE_V)

        ########################################################################################################################
        # Reestimate of Trad_V and Trad_S using Sensible Heat Flux
        ########################################################################################################################
        Trad_V = TSEB.calc_T_C_series(
            Tr_K=Trad,
            T_A_K=Tair,
            R_A=R_A,
            R_x=R_x,
            R_S=R_S,
            f_theta=f_theta,
            H_C=H_V,
            rho=rho,
            c_p=c_p
        )
        # print(Trad_V)

        Trad_S = myTSEB.estimate_Trad_S(
            Trad=Trad,
            Trad_V=Trad_V,
            f_theta=f_theta
        )
        # print(rf'Alpha: {alpha_PT}, Trad_S: {Trad_S}, Trad_V: {Trad_V}')
        # print(Trad_S)
        ########################################################################################################################
        # Reestimate Soil Sensible Heat Flux (H_S) because it depends on Trad_S
        ########################################################################################################################
        _, _, R_S = TSEB.calc_resistances(
            resistance_form,
            {
                "R_S": {
                    "u_friction": u_friction,
                    "h_C": h_V,
                    "d_0": d_0,
                    "z_0M": z_0m,
                    "L": L,
                    "F": F,
                    "omega0": omega,
                    "LAI": LAI,
                    "leaf_width": leaf_width,
                    "z0_soil": z0_soil,
                    "z_u": z_u,
                    "deltaT": Trad_S - T_AC,  # Trad_S - T_AC
                    "u": u,
                    "rho": rho,
                    "c_p": c_p,
                    "f_cover": fv02,
                    "w_C": w_V,
                    "massman_profile": massman_profile,
                    "res_params": {k: res_params[k] for k in res_params.keys()},
                }
            }
        )

        # 2) Air temperature at canopy interface (T_AC), array form
        T_AC = (
                (Tair / R_A + Trad_S / R_S + Trad_V / R_x) /
                (1.0 / R_A + 1.0 / R_S + 1.0 / R_x)
        )

        # 3) Soil sensible heat flux H_S, array form
        H_S = rho * c_p * (Trad_S - T_AC) / R_S

        ########################################################################################################################
        # Reestimate Net Radiation estimation with Trad_V and Trad_S from Sensible Heat Flux
        ########################################################################################################################
        Rn_V, Rn_S = myTSEB.estimate_Rn(
            S_dn=Sdn,
            sza=sza_degrees,
            P_atm=P_atm,
            LAI=LAI,
            x_LAD=x_LAD,
            omega=omega,
            Tair=Tair,
            ea=ea,
            Trad_S=Trad_S,
            Trad_V=Trad_V,
            rho_vis_leaf=rho_vis_leaf,
            rho_nir_leaf=rho_nir_leaf,
            tau_vis_leaf=tau_vis_leaf,
            tau_nir_leaf=tau_nir_leaf,
            rho_vis_soil=rho_vis_soil,
            rho_nir_soil=rho_nir_soil,
            emis_leaf=emis_leaf,
            emis_soil=emis_soil
        )

        ########################################################################################################################
        # Compute Soil Heat Flux Ratio as a Ratio of Rn_S
        ########################################################################################################################
        G = G_ratio * Rn_S

        LE_S = Rn_S - G - H_S
        LE_V = Rn_V - H_V
        LE_total = LE_V + LE_S
        # print(rf'LE_S: {LE_S}')

        alpha_PT = np.maximum(alpha_PT - 0.1, 0.0)
        iterations += 1
        alpha_condition = np.any(alpha_PT > 0)
        loop_con = np.any(LE_S < 0)

    return LE_V, LE_S, LE_total

# path_si = rf'files/outputs_sensitivity_analysis_C&NR.csv'
# if os.path.exists(path_si):
#     os.remove(path_si)
#     print(f"File '{path_si}' has been deleted.")
# else:
#     print(f"File '{path_si}' does not exist.")

# Independent base inputs only
names = [
        "LAI", "difffv", "h_V", "row_sep", "x_LAD",
        "leaf_width", "diffTrad", "Tair", "Sdn", "u",
        "P_atm", "ea", "sza_degrees", "saa_degrees", 'G_ratio',
        'row_azimuth', 'emis_leaf', 'emis_soil', 'rho_vis_leaf', 'tau_vis_leaf',
        'rho_nir_leaf', 'tau_nir_leaf', 'rho_vis_soil', 'rho_nir_soil']

problem = {
    "num_vars": 24,
    "names": names,
    "bounds": [
        [0.5, 8],  # 1. LAI
        [0.7, 1.1], # 2. difffv respect to LAI: f(LAI) * difffv [0.01, 1]
        [0.1, 5],  # 3. h_V (m)
        [1, 7],  # 4. row_sep (m)
        [0.5, 1.5],  # 5. x_LAD
        [0.005, 0.2],  # 6. leaf_width (NOT DEFINED in screenshot - fill if needed) (m)
        [-2, 10],  # 7. Trad = Tair + diffTrad  (Tair 283–313, diff 0–15 → max = 328)
        [283, 313],  # 8. Tair (kelvin)
        [700, 1000],  # 9. Sdn (w m-2)
        [0.2, 5],  # 10. u (m s-1)
        [990, 1010],  # 11. P_atm (mb)
        [5, 35],  # 12. ea (mb)
        [10, 70],  # 13. sza_degrees
        [0, 360], # 14. saa_degrees
        [0.01, 0.4], # 15. G_ratio
        [0, 179], # 16. row_azimuth
        [0.96, 0.99],  # 17. emis_leaf
        [0.90, 0.98],  # 18. emis_soil
        [0.03, 0.18],  # 19. rho_vis_leaf
        [0.02, 0.10],  # 20. tau_vis_leaf
        [0.32, 0.55],  # 21. rho_nir_leaf (tightened)
        [0.25, 0.45],  # 22. tau_nir_leaf (tightened)
        [0.05, 0.35],  # 23. rho_vis_soil
        [0.15, 0.50],  # 24. rho_nir_soil
]
}

# 2) Saltelli sampling
N = 1000  # or 2000, 5000… (trade-off cost vs accuracy)
second_order = False

X = saltelli.sample(problem, N, calc_second_order=second_order)

# 3) Evaluate the model for each row of X
def model_from_iso(row):
    (LAI, difffv, h_V, row_sep, x_LAD,
     leaf_width, diffTrad, Tair, Sdn, u,
     P_atm, ea, sza_degrees, saa_degrees, G_ratio,
     row_azimuth, emis_leaf, emis_soil, rho_vis_leaf, tau_vis_leaf,
     rho_nir_leaf, tau_nir_leaf, rho_vis_soil, rho_nir_soil) = row

    if (rho_vis_leaf + tau_vis_leaf >= 1) or (rho_nir_leaf + tau_nir_leaf >= 1):
        return np.nan, np.nan, np.nan

    return sensitivity_analysis_myTSEB(LAI, difffv, h_V, row_sep, x_LAD,
                                       leaf_width, diffTrad, Tair, Sdn, u,
                                       P_atm, ea, sza_degrees, saa_degrees, G_ratio,
                                       row_azimuth, emis_leaf, emis_soil, rho_vis_leaf, tau_vis_leaf,
                                       rho_nir_leaf, tau_nir_leaf, rho_vis_soil, rho_nir_soil, T_MODEL=T_MODEL)


Y_LEV = np.zeros(X.shape[0])
Y_LES = np.zeros(X.shape[0])
Y_LE_total = np.zeros(X.shape[0])

for i, row in enumerate(X):
    LEV, LES, LE_total = model_from_iso(row)

    mLEV = np.nanmean(LEV)
    mLES = np.nanmean(LES)
    mLE_total= np.nanmean(LE_total)

    if np.any(np.isnan([mLE_total, mLEV, mLES])):
        Y_LE_total[i] = -9999
        Y_LEV[i] = -9999
        Y_LES[i] = -9999
    else:
        Y_LE_total[i] = mLE_total
        Y_LEV[i] = mLEV
        Y_LES[i] = mLES

Si_LE_total = sobol.analyze(problem, Y_LE_total,  calc_second_order=False)
Si_LEV = sobol.analyze(problem, Y_LEV, calc_second_order=False)
Si_LES = sobol.analyze(problem, Y_LES, calc_second_order=False)

df_inputs = pd.DataFrame(X)
df_inputs.columns = names
df_inputs.loc[:, 'outcome'] = Y_LE_total

path = r'files/inputs_sensitivity_analysis_{}.csv'.format(T_MODEL)
df_inputs.to_csv(path)
#
# assert Y.ndim == 1, f"Y must be 1-D; got {Y.shape}"
# assert Y.shape[0] == X.shape[0], f"len(Y)={Y.shape[0]} vs nsamples={X.shape[0]}"

# 5) Sobol analysis
# Si = sobol.analyze(problem, Y, calc_second_order=second_order, print_to_console=False)

# si_df = pd.DataFrame(Si)
# si_df.loc[:, 'params'] = problem["names"]
# si_df.loc[:, 'N'] = N
#
for n, s1, st, c1, ct in zip(problem["names"], Si_LE_total["S1"], Si_LE_total["ST"], Si_LE_total["S1_conf"], Si_LE_total["ST_conf"]):
    print(f"{n:>12s} | S1={s1:.3f} (±{c1:.3f})  ST={st:.3f} (±{ct:.3f})")


    # data = pd.DataFrame.from_dict({
    #     'LAI': LAI,
    #     'fv': fv02,
    #     'h_V': h_V,
    #     'row_sep': row_sep,
    #     'x_LAD': x_LAD,
    #     'leaf_width': leaf_width,
    #     'Trad': Trad - Tair,
    #     'Trad_V': Trad_V,
    #     'Trad_S': Trad_S,
    #     'Tair': Tair,
    #     'Sdn': Sdn,
    #     'u': u,
    #     'P_atm': P_atm,
    #     'ea': ea,
    #     'sza_degrees': sza_degrees,
    #     'saa_degrees': saa_degrees,
    #     'G_ratio': G_ratio,
    #     # 'vza_degrees':vza_degrees,
    #     'row_azimuth': row_azimuth,
    #     'LE_V': LE_V,
    #     'LE_S': LE_S,
    #     'f_theta': f_theta,
    #     'Sdn': Sdn,
    #     'Rn_V': Rn_V,
    #     'Rn_S': Rn_S,
    #     'Rn_V0': Rn_V0_copy,
    #     'Rn_S0': Rn_S0_copy,
    #     'iteration': iterations,
    # }, orient='index').T

    # docker_inputs = docker_inputs.append(data)
    # if save_inputs:
    #     if os.path.exists(path):
    #         with open(path, mode="a", newline="") as f:
    #             writer = csv.writer(f)
    #             writer.writerow(data.iloc[0].to_list())
    #     else:
    #         data.to_csv(path, index=False)