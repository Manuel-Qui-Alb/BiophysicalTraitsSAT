import numpy as np
import pandas as pd

import functions as myTSEB
import pyTSEB.meteo_utils as met
from pyTSEB import resistances as res
from pyTSEB import MO_similarity as MO
import pyTSEB.TSEB as TSEB
from SALib.sample import saltelli
from SALib.analyze import sobol
from binomial_model.BinomialModelPrism_DirectandDiffuseRadiation import compute_binomial_prism_manuel
from pyTSEB import net_radiation as rad
import time

massman_profile = [0, []]

G_constant = 0.1
calcG_params = [[1], G_constant]
G_ratio = 0.1
const_L = None

save_inputs = True
T_MODEL = 'CN_R'

FLAG_OK                = 0
FLAG_NO_CONVERGENCE     = 1 << 0   # 1
FLAG_RAD_INCONSISTENCY  = 1 << 1   # 2
FLAG_RES_INVALID        = 1 << 2   # 4
FLAG_NUMERICAL          = 1 << 3   # 8
FLAG_OPTICS_INVALID     = 1 << 4   # 16
FLAG_LEV_NEGATIVE      = 1 << 5   # 32
FLAG_LES_NEGATIVE      = 1 << 6   # 64
FLAG_LETOTAL_NEGATIVE  = 1 << 7   # 128
FLAG_AELES_NEGATIVE  = 1 << 8

########################################################################################################################
# f_theta = fraction of incident beam radiation intercepted by the plant
# f_theta correspond to the radiation available for scattering, transpiration, and photosynthesis.
########################################################################################################################
# myTSEB(LAI, fv, x_LAD, sza_rad)
docker_inputs = []

def myTSEB_CN(LAI, fv_var, h_V, row_sep, x_LAD, phi_degrees,
              leaf_width, Trad_var, Tair, u, plant_sep_var,
              P_atm, ea, sza_degrees, G_ratio,
              emis_leaf, emis_soil, rho_vis_leaf, tau_vis_leaf,
              rho_nir_leaf, tau_nir_leaf, rho_vis_soil, rho_nir_soil,
              T_MODEL='CN_R', alpha_PT=1.26):

    Trad = Tair + Trad_var
    z_u = h_V + 2
    z_T = h_V + 2

    abs_vis_leaf = 1 - rho_vis_leaf - tau_vis_leaf
    abs_nir_leaf = 1 - rho_nir_leaf - tau_nir_leaf

    sza_rad =  np.radians(sza_degrees)
    phi_rad = np.radians(phi_degrees)

    # vza_rad = np.radians(vza_degrees)
    # Campbell and Norman 1998. Page 172.
    m = 101.3 / (101.3 * np.cos(sza_rad))
    tau_atms = 0.6  # atmospheric transmittance
    Sp0 = 1368  # Extraterrestrial flux density
    Sp = 1368 * (tau_atms ** m)
    Sb = Sp * np.cos(sza_rad)

    Sd = 0.3 * (1 - tau_atms ** m) * Sp0 * np.cos(sza_rad)
    # f_diff = np.clip(0.15 + 0.6 * (1 - mu), 0.15, 0.7)
    Sdn = Sb + Sd

    # Fv based on LAI. fv and LAI are inherently related.
    # It was considered that the FV can vary between 30 and 1.1 with respect to models based on LAI.
    K_be = myTSEB.estimate_Kbe(x_LAD, 0)
    fv02 = np.clip((1 - np.exp(-K_be * LAI)) * fv_var, 1e-6, 1)
    w_V = fv02 * row_sep
    # plant_sep = row_sep * plant_sep_var  # sp [0.5, 1]

    F = np.asarray(LAI / fv02, dtype=np.float32)

    c_p = met.calc_c_p(P_atm, ea)

    alpha_PT = np.full_like(LAI,  alpha_PT)
    L = np.zeros(Trad.shape) + np.inf

    # (emis_leaf, emis_soil, rho_vis_leaf_2, tau_vis_leaf, rho_nir_leaf_2, tau_nir_leaf, rho_vis_soil, rho_nir_soil,
    #  z0_soil) = myTSEB.get_emiss_rho_tau(LAI)
    z0_soil = np.full(LAI.shape, 0.01)

    Omega0 = TSEB.CI.calc_omega0_Kustas(
        LAI,
        fv02,
        x_LAD=x_LAD,
        isLAIeff=False
    )

    if T_MODEL == 'CN_H':
        # omega = myTSEB.off_nadir_clumpling_index_Kustas_Norman(
        #     LAI,
        #     fv02,
        #     h_V,
        #     w_V,
        #     x_LAD,
        #     sza_rad
        # )

        omega = TSEB.CI.calc_omega_Kustas(
            Omega0,
            sza_degrees,
            w_C= w_V / h_V
        )
    elif T_MODEL == 'CN_R':
        omega = TSEB.CI.calc_omega_rows(
            LAI,
            fv02,
            theta=sza_degrees,
            psi=phi_degrees,
            w_c=w_V / h_V,
            x_lad=x_LAD,
            is_lai_eff=False
        )
        # omega = np.full_like(LAI, 1)
    else:
        omega = np.full_like(LAI, 1)

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

    # res_params = [0,{}]
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
    Trad_V = np.min([Tair, Trad], axis=0) # We set the first Trad_V equal to Tair, under potential ET conditions.
    Trad_S = myTSEB.estimate_Trad_S(
        Trad=Trad,
        Trad_V=Trad_V,
        f_theta=f_theta
    )
    # params.update({'Trad_S_0': Trad_S_0})
    difvis, difnir, fvis, fnir = TSEB.rad.calc_difuse_ratio(
        S_dn=Sdn,
        sza=sza_degrees,
        press=np.full_like(Sdn, 1013)
    )

    skyl = fvis * difvis + fnir * difnir
    Sdn_dir = (1. - skyl) * Sdn
    Sdn_dif = skyl * Sdn

    Sn_V, Sn_S, Rn_V0, Rn_S0 = myTSEB.estimate_Rn(
        Sdn_dir=Sdn_dir,
        Sdn_dif=Sdn_dif,
        fvis=fvis,
        fnir=fnir,
        sza=sza_degrees,
        LAI=LAI,
        Trad_S=Trad_S,
        Trad_V=Trad_V,
        Tair=Tair,
        ea=ea,
        P_atm=P_atm,
        omega=omega,
        x_LAD=x_LAD,
        rho_vis_leaf=rho_vis_leaf,
        rho_nir_leaf=rho_nir_leaf,
        tau_vis_leaf=tau_vis_leaf,
        tau_nir_leaf=tau_nir_leaf,
        rho_vis_soil=rho_vis_soil,
        rho_nir_soil=rho_nir_soil,
        emis_leaf=emis_leaf,
        emis_soil=emis_soil)

    # Rn_V0 = np.array(np.clip(Rn_V0, -150, 1000))
    # Rn_S0 = np.array(np.clip(Rn_S0, -150, 1000))

    # Rn_V0_copy = Rn_V0.copy()
    # Rn_S0_copy = Rn_S0.copy()
    L_dn = rad.calc_longwave_irradiance(
        ea,
        Tair,
        p=P_atm,
        z_T=z_T,
        h_C=h_V
    )

    Ln_V, Ln_S = rad.calc_L_n_Campbell(
        T_C=Trad_V,
        T_S=Trad_S,
        L_dn=L_dn,
        lai=LAI,
        emisVeg=emis_leaf,
        emisGrd=emis_soil,
        x_LAD=x_LAD
    )

    Rn_V0 = Sn_V + Ln_V
    Rn_S0 = Sn_S + Ln_S

    [LE_V, LE_S, LE, H_V, H_S, H, G, Ln_V, Ln_S, Rn_V, Rn_S, R_A, R_x, R_S, AELE_S] = [np.full_like(LAI, -9999)
                                                                                       for i in range(15)]

    loop_con = np.full_like(LAI, True, dtype=bool)
    max_iterations = 14
    iterations = 1
    # alpha_condition = np.any(alpha_PT > 0)
    flag = np.full_like(LAI, FLAG_OK)

    # Rn_V0_copy, Rn_S0_copy = Rn_V0.copy(), Rn_S0.copy()
    while (np.any(loop_con) and iterations <= max_iterations) :
        # print(iterations)
        # while len(loop_con)>0:
        # print(loop_con))
        # print('Iteration', iterations)

        R_A[loop_con], R_x[loop_con], R_S[loop_con] = TSEB.calc_resistances(
            resistance_form,
            {
                "R_A": {
                    "z_T": z_T[loop_con],
                    "u_friction": u_friction[loop_con],
                    "L": L[loop_con],
                    "d_0": d_0[loop_con],
                    "z_0H": z_0H[loop_con],
                },
                "R_x": {
                    "u_friction": u_friction[loop_con],
                    "h_C": h_V[loop_con],
                    "d_0": d_0[loop_con],
                    "z_0M": z_0m[loop_con],
                    "L": L[loop_con],
                    "F": F[loop_con],
                    "LAI": LAI[loop_con],
                    "leaf_width": leaf_width[loop_con],
                    "z0_soil": z0_soil[loop_con],
                    "massman_profile": massman_profile,
                    "res_params": {k: res_params[k] for k in res_params.keys()},
                },
                "R_S": {
                    "u_friction": u_friction[loop_con],
                    "h_C": h_V[loop_con],
                    "d_0": d_0[loop_con],
                    "z_0M": z_0m[loop_con],
                    "L": L[loop_con],
                    "F": F[loop_con],
                    "omega0": Omega0[loop_con],
                    "LAI": LAI[loop_con],
                    "leaf_width": leaf_width[loop_con],
                    "z0_soil": z0_soil[loop_con],
                    "z_u": z_u[loop_con],
                    "deltaT": Trad_S[loop_con] - T_AC[loop_con],
                    "u": u[loop_con],
                    "rho": rho[loop_con],
                    "c_p": c_p[loop_con],
                    "f_cover": fv02[loop_con],
                    "w_C": w_V[loop_con] / h_V[loop_con],
                    "massman_profile": massman_profile,
                    "res_params": {k: res_params[k] for k in res_params.keys()},
                },
            },
        )

        ########################################################################################################################
        # Estimate Net Radiation estimation with Trad_V and Trad_S from Sensible Heat Flux
        ########################################################################################################################
        if iterations == 1:
            Rn_V[loop_con] = Rn_V0[loop_con]
            Rn_S[loop_con] = Rn_S0[loop_con]
        else:
            Rn_V[loop_con] = Sn_V[loop_con] + Ln_V[loop_con]
            Rn_S[loop_con] = Sn_S[loop_con] + Ln_S[loop_con]

        # alpha_PT = np.where(alpha_PT < 0, 0, alpha_PT)
        LE_V[loop_con] = myTSEB.Priestly_Taylor_LE_V(
            fv_g=1, #np.full_like(f_theta[loop_con], 1),
            Rn_V=Rn_V[loop_con], # This change after every loop
            alpha_PT=alpha_PT[loop_con],
            Tair=Tair[loop_con],
            P_atm=P_atm[loop_con],
            c_p=c_p[loop_con]
        )

        H_V[loop_con] = Rn_V[loop_con] - LE_V[loop_con]

        ########################################################################################################################
        # Reestimate of Trad_V and Trad_S using Sensible Heat Flux
        ########################################################################################################################
        Trad_V[loop_con] = TSEB.calc_T_C_series(
            Tr_K=Trad[loop_con],
            T_A_K=Tair[loop_con],
            R_A=R_A[loop_con],
            R_x=R_x[loop_con],
            R_S=R_S[loop_con],
            f_theta=f_theta[loop_con],
            H_C=H_V[loop_con],
            rho=rho[loop_con],
            c_p=c_p[loop_con]
        )
        # print(Trad_V)

        Trad_S[loop_con] = myTSEB.estimate_Trad_S(
            Trad=Trad[loop_con],
            Trad_V=Trad_V[loop_con],
            f_theta=f_theta[loop_con]
        )
        flag_RAD_INCONSISTENCY_test = np.isnan(Trad_S)
        flag[flag_RAD_INCONSISTENCY_test] = FLAG_RAD_INCONSISTENCY

        Ln_V[loop_con], Ln_S[loop_con] = rad.calc_L_n_Campbell(
            T_C=Trad_V[loop_con],
            T_S=Trad_S[loop_con],
            L_dn=L_dn[loop_con],
            lai=LAI[loop_con],
            emisVeg=emis_leaf[loop_con],
            emisGrd=emis_soil[loop_con],
            x_LAD=x_LAD[loop_con]
        )

        Rn_V[loop_con] = Sn_V[loop_con] + Ln_V[loop_con]
        Rn_S[loop_con] = Sn_S[loop_con] + Ln_S[loop_con]

        # print(rf'Alpha: {alpha_PT}, Trad_S: {Trad_S}, Trad_V: {Trad_V}')
        # print(Trad_S)
        ########################################################################################################################
        # Reestimate Soil Sensible Heat Flux (H_S) because it depends on Trad_S
        ########################################################################################################################
        _, _, R_S[loop_con] = TSEB.calc_resistances(
            resistance_form,
            {
                "R_S": {
                    "u_friction": u_friction[loop_con],
                    "h_C": h_V[loop_con],
                    "d_0": d_0[loop_con],
                    "z_0M": z_0m[loop_con],
                    "L": L[loop_con],
                    "F": F[loop_con],
                    "omega0": Omega0[loop_con],
                    "LAI": LAI[loop_con],
                    "leaf_width": leaf_width[loop_con],
                    "z0_soil": z0_soil[loop_con],
                    "z_u": z_u[loop_con],
                    "deltaT": Trad_S[loop_con] - T_AC[loop_con],
                    "u": u[loop_con],
                    "rho": rho[loop_con],
                    "c_p": c_p[loop_con],
                    "f_cover": fv02[loop_con],
                    "w_C": w_V[loop_con] / h_V[loop_con],
                    "massman_profile": massman_profile,
                    "res_params": {k: res_params[k] for k in res_params.keys()},
                }
            }
        )

        # 2) Air temperature at canopy interface (T_AC), array form
        T_AC[loop_con] = (
                (Tair[loop_con] / R_A[loop_con] + Trad_S[loop_con] / R_S[loop_con] + Trad_V[loop_con] / R_x[loop_con]) /
                (1.0 / R_A[loop_con] + 1.0 / R_S[loop_con] + 1.0 / R_x[loop_con])
        )

        # 3) Soil sensible heat flux H_S, array form
        H_S[loop_con] = rho[loop_con] * c_p[loop_con] * (Trad_S[loop_con] - T_AC[loop_con]) / R_S[loop_con]

        ########################################################################################################################
        # Compute Soil Heat Flux Ratio as a Ratio of Rn_S
        # In first iteration, initial Rn_V and Rn_S is used to keep consistency in the use th initial Rn_V and Rn_S
        # to estimate H_V and H_S.
        ########################################################################################################################
        G[loop_con] = G_ratio[loop_con] * Rn_S[loop_con]

        H[loop_con] = H_V[loop_con] + H_S[loop_con]
        LE_S[loop_con] = Rn_S[loop_con] - G[loop_con] - H_S[loop_con]
        LE_V[loop_con] = Rn_V[loop_con] - H_V[loop_con]
        LE[loop_con] = LE_V[loop_con] + LE_S[loop_con]
        # print(rf'LE_S: {LE_S}')

        alpha_PT[loop_con] = np.maximum(alpha_PT[loop_con] - 0.1, 0.0)

        # alpha_condition = alpha_PT > 0
        AELE_S[loop_con] = (1.0 - G_ratio[loop_con]) * Rn_S[loop_con]

        con_LE_S = (LE_S < 0)
        con_AE_pos = (AELE_S > 0)
        con_AE_neg = (AELE_S < 0)

        # Flag physically negative available energy
        flag[con_AE_neg] = FLAG_AELES_NEGATIVE

        loop_con = con_LE_S & con_AE_pos

        if const_L is None:
            L[loop_con] = MO.calc_L(
                u_friction[loop_con],
                Tair[loop_con],
                rho[loop_con],
                c_p[loop_con],
                H[loop_con],
                LE[loop_con])
            # Calculate again the friction velocity with the new stability
            # correctios
            u_friction[loop_con] = MO.calc_u_star(
                u[loop_con], z_u[loop_con], L[loop_con], d_0[loop_con], z_0m[loop_con])
            u_friction[loop_con] = np.asarray(np.maximum(U_FRICTION_MIN[loop_con], u_friction[loop_con]), dtype=np.float32)

        # only evaluate where AE_S > 0 to avoid nonsense division

        iterations += 1
        # print(np.where(loop_con)[0].shape)
        # print("L_dn:", np.nanmin(L_dn), np.nanmean(L_dn), np.nanmax(L_dn))


    flag_NO_CONVERGENCE_test = LE_S < 0
    flag[flag_NO_CONVERGENCE_test] = FLAG_NO_CONVERGENCE

    flag_FLAG_NUMERICAL_test = np.isnan(LE_S) & (flag == 0)
    flag[flag_FLAG_NUMERICAL_test] = FLAG_NUMERICAL

    flag_LEV_NEGATIVE_test = LE_V < 0
    flag[flag_LEV_NEGATIVE_test] = FLAG_LEV_NEGATIVE

    flag_LETOTAL_NEGATIVE_test = LE < 0
    flag[flag_LETOTAL_NEGATIVE_test] = FLAG_LETOTAL_NEGATIVE

    return omega, f_theta, Sn_V, Sn_S, Trad_V, Trad_S, Ln_V, Ln_S, Rn_V, Rn_S, LE_V, LE_S, flag
#

names = [
        "LAI", "fv_var", "h_V", "row_sep", "x_LAD", 'phi_degrees',
        "leaf_width", "Trad_var", "Tair", "u", "plant_sep_var",
        "P_atm", "ea", "sza_degrees", 'G_ratio',
        'emis_leaf', 'emis_soil', 'rho_vis_leaf', 'tau_vis_leaf',
        'rho_nir_leaf', 'tau_nir_leaf', 'rho_vis_soil', 'rho_nir_soil']

problem = {
    "num_vars": 23,
    "names": names,
    "bounds": [
        [0.2, 5],  # 1. LAI
        [0.7, 1.1], # 2. fv_var respect to LAI: f(LAI) * fv_var [0.01, 1]
        [1, 5],  # 3. h_V (m)
        [2, 8],  # 4. row_sep (m)
        [0.5, 1.5],  # 5. x_LAD
        [0, 180], # 6. phi_degrees
        [0.005, 0.2],  # 7. leaf_width (NOT DEFINED in screenshot - fill if needed) (m)
        [-2, 5],  # 8. Trad = Tair + Trad_var  (Tair 283–313, diff 0–15 → max = 328)
        [283, 313],  # 9. Tair (kelvin)
        [0.2, 5],  # 10. u (m s-1)
        [0.5, 1],  # 5. 11. plant_sep_var: plant space in relation to sr
        [990, 1010],  # 12. P_atm (mb)
        [5, 35],  # 13. ea (mb)
        [15, 65],  # 14. sza_degrees
        [0.34, 0.36], # 15. G_ratio
        [0.96, 0.99],  # 16. emis_leaf
        [0.90, 0.98],  # 17. emis_soil
        [0.03, 0.18],  # 18. rho_vis_leaf
        [0.02, 0.10],  # 19. tau_vis_leaf
        [0.32, 0.55],  # 20. rho_nir_leaf (tightened)
        [0.25, 0.45],  # 21. tau_nir_leaf (tightened)
        [0.05, 0.30],  # 22. rho_vis_soil
        [0.20, 0.45],  # 23. rho_nir_soil
]
}


N = 1000
second_order = False
X = saltelli.sample(problem, N, calc_second_order=second_order)

inputs_dict = {f"{names[i]}": X[:, i:i+1] for i in range(X.shape[1])}

start_time = time.perf_counter()
omega, f_theta, Sn_V, Sn_S, Trad_V, Trad_S, Ln_V, Ln_S, Rn_V, Rn_S, LE_V, LE_S, flag = (
    myTSEB_CN(
        LAI=inputs_dict['LAI'],
        fv_var=inputs_dict['fv_var'],
        h_V=inputs_dict['h_V'],
        row_sep=inputs_dict['row_sep'],
        x_LAD=inputs_dict['x_LAD'],
        phi_degrees=inputs_dict['phi_degrees'],
        leaf_width=inputs_dict['leaf_width'],
        Trad_var=inputs_dict['Trad_var'],
        Tair=inputs_dict['Tair'],
        u=inputs_dict['u'],
        plant_sep_var=inputs_dict['plant_sep_var'],
        P_atm=inputs_dict['P_atm'],
        ea=inputs_dict['ea'],
        sza_degrees=inputs_dict['sza_degrees'],
        G_ratio=inputs_dict['G_ratio'],
        emis_leaf=inputs_dict['emis_leaf'],
        emis_soil=inputs_dict['emis_soil'],
        rho_vis_leaf=inputs_dict['rho_vis_leaf'],
        tau_vis_leaf=inputs_dict['tau_vis_leaf'],
        rho_nir_leaf=inputs_dict['rho_nir_leaf'],
        tau_nir_leaf=inputs_dict['tau_nir_leaf'],
        rho_vis_soil=inputs_dict['rho_vis_soil'],
        rho_nir_soil=inputs_dict['rho_nir_soil'],
        T_MODEL='CN_R',
        alpha_PT=1.26
    )
)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Execution time TSEB B&N: {elapsed_time} seconds")

df_inputs = pd.DataFrame(X)
df_inputs.columns = names

df_inputs.loc[:, 'omega_CN'] = omega
df_inputs.loc[:, 'f_theta_CN'] = f_theta
df_inputs.loc[:, 'Sn_V_CN'] = Sn_V
df_inputs.loc[:, 'Sn_S_CN'] = Sn_S
df_inputs.loc[:, 'Trad_V_CN'] = Trad_V
df_inputs.loc[:, 'Trad_S_CN'] = Trad_S
df_inputs.loc[:, 'Ln_V_CN'] = Ln_V
df_inputs.loc[:, 'Ln_S_CN'] = Ln_S
df_inputs.loc[:, 'Rn_V_CN'] = Rn_V
df_inputs.loc[:, 'Rn_S_CN'] = Rn_S
df_inputs.loc[:, 'LE_V_CN'] = LE_V
df_inputs.loc[:, 'LE_S_CN'] = LE_S

df_inputs.loc[:, 'flag_CN'] = flag

df_inputs.to_csv('files/TSEB_CN.csv')

LE = LE_V + LE_S
Si_LE_V = sobol.analyze(problem, LE_V[:, 0], calc_second_order=second_order, print_to_console=False)
Si_LE_S = sobol.analyze(problem, LE_S[:, 0], calc_second_order=second_order, print_to_console=False)
Si_LE_total = sobol.analyze(problem, LE[:, 0], calc_second_order=second_order, print_to_console=False)

Si_Sn_V = sobol.analyze(problem, Sn_V[:, 0], calc_second_order=second_order, print_to_console=False)
Si_Sn_S = sobol.analyze(problem, Sn_S[:, 0], calc_second_order=second_order, print_to_console=False)

Si_Rn_V = sobol.analyze(problem, Rn_V[:, 0], calc_second_order=second_order, print_to_console=False)
Si_Rn_S = sobol.analyze(problem, Rn_S[:, 0], calc_second_order=second_order, print_to_console=False)

Si_Trad_V = sobol.analyze(problem, Trad_V[:, 0], calc_second_order=second_order, print_to_console=False)
Si_Trad_S = sobol.analyze(problem, Trad_S[:, 0], calc_second_order=second_order, print_to_console=False)

Si_f_theta = sobol.analyze(problem, f_theta[:, 0], calc_second_order=second_order, print_to_console=False)

def si_to_df(si):
    df_si = pd.DataFrame(si)
    df_si.insert(0, 'params', problem['names'])
    return df_si

(df_Si_LE_V, df_Si_LE_S, df_Si_LE_total, df_Si_Sn_V, df_Si_Sn_S, df_Si_Rn_V, df_Si_Rn_S, df_Si_Trad_V, df_Si_Trad_S,
 df_Si_f_theta)= [si_to_df(x) for x in [Si_LE_V, Si_LE_S, Si_LE_total, Si_Sn_V, Si_Sn_S, Si_Rn_V, Si_Rn_S,
                                        Si_Trad_V, Si_Trad_S, Si_f_theta]
     ]

# Rn_V, Rn_S
path = 'files/sensitivity_analysis_TSEB_CN.xlsx'
with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
    # df_inputs.to_excel(writer, sheet_name='inputs', index=False)
    df_Si_LE_V.to_excel(writer, sheet_name='LE_V', index=False)
    df_Si_LE_S.to_excel(writer, sheet_name='LE_S', index=False)
    df_Si_LE_total.to_excel(writer, sheet_name='LE_total', index=False)

    df_Si_Sn_V.to_excel(writer, sheet_name='Sn_V', index=False)
    df_Si_Sn_S.to_excel(writer, sheet_name='Sn_S', index=False)

    df_Si_Rn_V.to_excel(writer, sheet_name='Rn_V', index=False)
    df_Si_Rn_S.to_excel(writer, sheet_name='Rn_S', index=False)

    df_Si_Trad_V.to_excel(writer, sheet_name='Trad_V', index=False)
    df_Si_Trad_S.to_excel(writer, sheet_name='Trad_S', index=False)
    df_Si_f_theta.to_excel(writer, sheet_name='f_theta', index=False)