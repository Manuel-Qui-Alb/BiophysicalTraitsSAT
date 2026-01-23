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

save_inputs = True
T_MODEL = 'CN_H'

FLAG_OK                = 0
FLAG_NO_CONVERGENCE     = 1 << 0   # 1
FLAG_RAD_INCONSISTENCY  = 1 << 1   # 2
FLAG_RES_INVALID        = 1 << 2   # 4
FLAG_NUMERICAL          = 1 << 3   # 8
FLAG_OPTICS_INVALID     = 1 << 4   # 16
FLAG_LEV_NEGATIVE      = 1 << 5   # 32
FLAG_LES_NEGATIVE      = 1 << 6   # 64
FLAG_LETOTAL_NEGATIVE  = 1 << 7   # 128

########################################################################################################################
# f_theta = fraction of incident beam radiation intercepted by the plant
# f_theta correspond to the radiation available for scattering, transpiration, and photosynthesis.
########################################################################################################################
# myTSEB(LAI, fv, x_LAD, sza_rad)
docker_inputs = []

def sensitivity_analysis_myTSEB(LAI, fv_var, h_V, row_sep, x_LAD,
                                leaf_width, Trad_var, Tair, Sdn, u,
                                P_atm, ea, sza_degrees, saa_degrees, G_ratio, row_azimuth,
                                emis_leaf, emis_soil, rho_vis_leaf, tau_vis_leaf,
                                rho_nir_leaf, tau_nir_leaf, rho_vis_soil, rho_nir_soil, T_MODEL='CN_H', alpha_PT=1.26):


    Trad = Tair + Trad_var
    z_u = h_V + 2
    z_T = h_V + 2

    sza_rad =  np.radians(sza_degrees)
    # vza_rad = np.radians(vza_degrees)
    saa_rad = np.radians(saa_degrees)
    row_azimuth_rad = np.radians(row_azimuth)

    # Fv based on LAI. fv and LAI are inherently related.
    # It was considered that the FV can vary between 30 and 1.1 with respect to models based on LAI.
    K_be = myTSEB.estimate_Kbe(x_LAD, 0)
    fv02 = np.clip((1 - np.exp(-K_be * LAI)) * fv_var, 1e-6, 1)
    w_V = fv02 * row_sep
    F = np.asarray(LAI / fv02, dtype=np.float32)

    c_p = met.calc_c_p(P_atm, ea)

    alpha_PT = np.full_like(LAI,  alpha_PT)
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
    max_iterations = 13
    iterations = 1
    alpha_condition = np.any(alpha_PT > 0)
    flag = np.full_like(LAI, FLAG_OK)

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
        flag_RAD_INCONSISTENCY_test = np.isnan(Trad_S)
        flag[flag_RAD_INCONSISTENCY_test] = FLAG_RAD_INCONSISTENCY

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

    flag_NO_CONVERGENCE_test = LE_S < 0
    flag[flag_NO_CONVERGENCE_test] = FLAG_NO_CONVERGENCE

    flag_FLAG_NUMERICAL_test = np.isnan(LE_S) & (flag == 0)
    flag[flag_FLAG_NUMERICAL_test] = FLAG_NUMERICAL

    flag_LEV_NEGATIVE_test = LE_V < 0
    flag[flag_LEV_NEGATIVE_test] = FLAG_LEV_NEGATIVE

    flag_LETOTAL_NEGATIVE_test = LE_total < 0
    flag[flag_LETOTAL_NEGATIVE_test] = FLAG_LETOTAL_NEGATIVE

    return LE_V, LE_S, LE_total, Rn_V, Rn_S, Trad_V, Trad_S, f_theta, flag
#


# Independent base inputs only
names = [
        "LAI", "fv_var", "h_V", "row_sep", "x_LAD",
        "leaf_width", "Trad_var", "Tair", "Sdn", "u",
        "P_atm", "ea", "sza_degrees", "saa_degrees", 'G_ratio',
        'row_azimuth', 'emis_leaf', 'emis_soil', 'rho_vis_leaf', 'tau_vis_leaf',
        'rho_nir_leaf', 'tau_nir_leaf', 'rho_vis_soil', 'rho_nir_soil']

problem = {
    "num_vars": 24,
    "names": names,
    "bounds": [
        [0.5, 8],  # 1. LAI
        [0.7, 1.1], # 2. fv_var respect to LAI: f(LAI) * fv_var [0.01, 1]
        [0.1, 5],  # 3. h_V (m)
        [1, 7],  # 4. row_sep (m)
        [0.5, 1.5],  # 5. x_LAD
        [0.005, 0.2],  # 6. leaf_width (NOT DEFINED in screenshot - fill if needed) (m)
        [-2, 10],  # 7. Trad = Tair + Trad_var  (Tair 283–313, diff 0–15 → max = 328)
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

# 3) Evaluate the model for each row of X
def model_from_iso(row):
    (LAI, fv_var, h_V, row_sep, x_LAD,
     leaf_width, Trad_var, Tair, Sdn, u,
     P_atm, ea, sza_degrees, saa_degrees, G_ratio,
     row_azimuth, emis_leaf, emis_soil, rho_vis_leaf, tau_vis_leaf,
     rho_nir_leaf, tau_nir_leaf, rho_vis_soil, rho_nir_soil) = row

    if (rho_vis_leaf + tau_vis_leaf >= 1) or (rho_nir_leaf + tau_nir_leaf >= 1):
        return np.nan, np.nan, np.nan, 16

    return sensitivity_analysis_myTSEB(LAI, fv_var, h_V, row_sep, x_LAD,
                                       leaf_width, Trad_var, Tair, Sdn, u,
                                       P_atm, ea, sza_degrees, saa_degrees, G_ratio,
                                       row_azimuth, emis_leaf, emis_soil, rho_vis_leaf, tau_vis_leaf,
                                       rho_nir_leaf, tau_nir_leaf, rho_vis_soil, rho_nir_soil, T_MODEL=T_MODEL)

N = 5000
second_order = False
X = saltelli.sample(problem, N, calc_second_order=second_order)

D = problem["num_vars"]
block_size = (2 * D + 2) if second_order else (D + 2)
assert X.shape[0] % block_size == 0
N_blocks = X.shape[0] // block_size

# lists of kept scalar outputs, block-by-block
Y_blocks = {k: [] for k in ["LE_total","LEV","LES","Rn_V","Rn_S","Trad_V","Trad_S","f_theta","fail"]}

def is_ok(flag, vals):
    # flag is scalar or array
    flag_ok = np.all(np.asarray(flag) == 0)
    vals_ok = np.all(np.isfinite(vals))
    return flag_ok and vals_ok

kept = 0
dropped = 0

for b in range(N_blocks):
    rows = X[b*block_size:(b+1)*block_size]

    block = {k: np.empty(block_size, dtype=float) for k in Y_blocks.keys()}
    block_ok = True

    for j, row in enumerate(rows):
        out = model_from_iso(row)

        # expected: LE_V, LE_S, LE_total, Rn_V, Rn_S, Trad_V, Trad_S, f_theta, flag
        LE_V, LE_S, LE_total, Rn_V, Rn_S, Trad_V, Trad_S, f_theta, flag = out

        mLEV      = np.nanmean(LE_V)
        mLES      = np.nanmean(LE_S)
        mLE_total = np.nanmean(LE_total)
        mRn_V     = np.nanmean(Rn_V)
        mRn_S     = np.nanmean(Rn_S)
        mTrad_V   = np.nanmean(Trad_V)
        mTrad_S   = np.nanmean(Trad_S)
        mf_theta  = np.nanmean(f_theta)

        vals = [mLE_total, mLEV, mLES, mRn_V, mRn_S, mTrad_V, mTrad_S, mf_theta]

        ok = is_ok(flag, vals)

        if not ok:
            block_ok = False
            break

        block["LE_total"][j] = mLE_total
        block["LEV"][j]      = mLEV
        block["LES"][j]      = mLES
        block["Rn_V"][j]     = mRn_V
        block["Rn_S"][j]     = mRn_S
        block["Trad_V"][j]   = mTrad_V
        block["Trad_S"][j]   = mTrad_S
        block["f_theta"][j]  = mf_theta
        block["fail"][j]     = 0.0  # all ok in kept blocks

    if block_ok:
        for k in Y_blocks:
            Y_blocks[k].append(block[k])
        kept += 1
    else:
        dropped += 1

if kept == 0:
    raise RuntimeError("No valid Saltelli blocks kept. Cannot run Sobol.")

# Concatenate kept blocks (length = kept * block_size)
Y = {k: np.concatenate(Y_blocks[k]) for k in Y_blocks}

print(f"Kept blocks: {kept}/{N_blocks} ({100*kept/N_blocks:.1f}%)")
print(f"Y length: {Y['LE_total'].size} (must be divisible by {block_size})")

# Now Sobol works because Y has correct Saltelli length/structure
Si_LE_total = sobol.analyze(problem, Y["LE_total"], calc_second_order=False)
Si_LEV      = sobol.analyze(problem, Y["LEV"],      calc_second_order=False)
Si_LES      = sobol.analyze(problem, Y["LES"],      calc_second_order=False)
Si_Rn_V     = sobol.analyze(problem, Y["Rn_V"],     calc_second_order=False)
Si_Rn_S     = sobol.analyze(problem, Y["Rn_S"],     calc_second_order=False)
Si_Trad_V   = sobol.analyze(problem, Y["Trad_V"],   calc_second_order=False)
Si_Trad_S   = sobol.analyze(problem, Y["Trad_S"],   calc_second_order=False)
Si_f_theta  = sobol.analyze(problem, Y["f_theta"],  calc_second_order=False)

path = r'files/sensitivity_analysis_{}_5000.xlsx'.format(T_MODEL)

def si_to_table(data, problem):
    df = pd.DataFrame(data)
    df.insert(0, 'params', problem['names'])
    return df
si_tables = [si_to_table(df, problem) for df in [Si_LE_total, Si_LEV, Si_LES, Si_Rn_V, Si_Rn_S,
                                                    Si_Trad_V, Si_Trad_S, Si_f_theta]]
names_outcomes = ['LE_total', 'Si_LEV', 'Si_LES', 'Si_Rn_V', 'Si_Rn_S', 'Si_Trad_V', 'Si_Trad_S', 'Si_f_theta']
# [si_tables[x].to_excel(path, sheet_name=names[x]) for x in range(0, len(si_tables))]
with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
    si_tables[0].to_excel(writer, sheet_name=names_outcomes[0], index=False)
    si_tables[1].to_excel(writer, sheet_name=names_outcomes[1], index=False)
    si_tables[2].to_excel(writer, sheet_name=names_outcomes[2], index=False)
    si_tables[3].to_excel(writer, sheet_name=names_outcomes[3], index=False)
    si_tables[4].to_excel(writer, sheet_name=names_outcomes[4], index=False)
    si_tables[5].to_excel(writer, sheet_name=names_outcomes[5], index=False)
    si_tables[6].to_excel(writer, sheet_name=names_outcomes[6], index=False)
    si_tables[7].to_excel(writer, sheet_name=names_outcomes[7], index=False)
    # si_tables[8].to_excel(writer, sheet_name=names[8], index=False)

"""
df_inputs = pd.DataFrame(X)
df_inputs.columns = names
df_inputs.loc[:, 'LEV'] = Y_LEV
df_inputs.loc[:, 'LES'] = Y_LES
df_inputs.loc[:, 'LE_total'] = Y_LE_total
df_inputs.loc[:, 'flag'] = Y_flag

path = r'files/inputs_sensitivity_analysis_{}.csv'.format(T_MODEL)
df_inputs.to_csv(path)
"""
#
for n, s1, st, c1, ct in zip(problem["names"], Si_LE_total["S1"], Si_LE_total["ST"], Si_LE_total["S1_conf"],
                             Si_LE_total["ST_conf"]):
    print(f"{n:>12s} | S1={s1:.3f} (±{c1:.3f})  ST={st:.3f} (±{ct:.3f})")
