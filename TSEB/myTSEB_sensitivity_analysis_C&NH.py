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


# None Parameters
ITERATIONS = 50
# L = np.zeros(np.array(params['LAI']).shape) + np.inf
max_iterations = ITERATIONS
massman_profile = [0, []]

G_constant = 0.1
calcG_params = [[1], G_constant]
G_ratio = 0.35

alpha_PT = 1.7

########################################################################################################################
# f_theta = fraction of incident beam radiation intercepted by the plant
# f_theta correspond to the radiation available for scattering, transpiration, and photosynthesis.
########################################################################################################################
# myTSEB(LAI, fv, x_LAD, sza_rad)
def sensitivity_analysis_myTSEB(LAI, fv, h_V, row_sep, x_LAD,
                                leaf_width, Trad, Tair, Sdn, u,
                                P_atm, ea, sza_degrees, saa_degrees, G_ratio, row_azimuth,
                                emis_leaf, emis_soil, rho_vis_leaf, tau_vis_leaf,
                                rho_nir_leaf, tau_nir_leaf, rho_vis_soil, rho_nir_soil):

    Trad = Tair + Trad
    z_u = h_V + 2
    z_T = h_V + 2

    sza_rad =  np.radians(sza_degrees)
    # vza_rad = np.radians(vza_degrees)
    saa_rad = np.radians(saa_degrees)
    row_azimuth_rad = np.radians(row_azimuth)

    # Fv based on LAI. fv and LAI are inherently related.
    # It was considered that the FV can vary between 30 and 1.1 with respect to models based on LAI.
    K_be = myTSEB.estimate_Kbe(x_LAD, 0)
    fv02 = np.clip((1 - np.exp(-K_be * LAI)) * fv, 0, 1)
    w_V = fv02 * row_sep
    F = np.asarray(LAI / fv02, dtype=np.float32)

    c_p = met.calc_c_p(P_atm, ea)

    alpha_PT = np.full_like(LAI,  1.7)
    L = np.zeros(Trad.shape) + np.inf

    # (emis_leaf, emis_soil, rho_vis_leaf_2, tau_vis_leaf, rho_nir_leaf_2, tau_nir_leaf, rho_vis_soil, rho_nir_soil,
    #  z0_soil) = myTSEB.get_emiss_rho_tau(LAI)
    z0_soil = np.full(LAI.shape, 0.01)

    omega = myTSEB.off_nadir_clumpling_index_Kustas_Norman(LAI, fv, h_V, w_V, x_LAD, sza_rad)

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

    KB_1_DEFAULTC = 0
    z_0H = res.calc_z_0H(
        z_0m,
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
        d_0,
        z_0m
    )

    U_FRICTION_MIN = np.full_like(LAI, 0.01)

    u_friction = np.asarray(np.nanmax([U_FRICTION_MIN, u_friction], axis=0), dtype=np.float32)

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

    Rn_V0 = np.array(np.clip(Rn_V0, -150, 1000))
    Rn_S0 = np.array(np.clip(Rn_S0, -150, 1000))

    Rn_V0_copy = Rn_V0.copy()
    Rn_S0_copy = Rn_S0.copy()

    [LE_V, LE_S, H_V, H_S, G, Trad_V, Trad_S] = [np.full_like(LAI, -9999) for i in range(7)]

    loop_con = True
    max_iterations = 50
    iterations = 1
    alpha_condition = True

    while (loop_con and alpha_condition and iterations < max_iterations) :
        # print(iterations)

        iterations += 1
        alpha_PT -= 0.1
        alpha_condition = alpha_PT > 0

        alpha_PT = np.where(alpha_PT < 0, 0, alpha_PT)

        try:
            Rn_V0 = Rn_V
            Rn_S0 = Rn_S
        except:
            None

        LE_V = myTSEB.Priestly_Taylor_LE_V(
            fv_g=f_theta,
            Rn_V=Rn_V0,
            alpha_PT=alpha_PT,
            Tair=Tair,
            P_atm=P_atm,
            c_p=c_p
        )


        H_V = Rn_V0 - LE_V

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
        # Compute Soil Heat Flux Ratio as a Ratio of Rn_S
        ########################################################################################################################
        G = G_ratio * Rn_S0

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

        LE_S = Rn_S - G - H_S
        LE_V = Rn_V - H_V
        # print(rf'LE_S: {LE_S}')
        loop_con = LE_S < 0

    return alpha_PT


problem = {
    "num_vars": 24,
    "names":  [
        "LAI", "fv", "h_V", "row_sep", "x_LAD",
        "leaf_width", "Trad", "Tair", "Sdn", "u",
        "P_atm", "ea", "sza_degrees", "saa_degrees", 'G_ratio',
        'row_azimuth', 'emis_leaf', 'emis_soil', 'rho_vis_leaf', 'tau_vis_leaf',
        'rho_nir_leaf', 'tau_nir_leaf', 'rho_vis_soil', 'rho_nir_soil'],
    "bounds": [
        [0.5, 8],  # 1. LAI
        [0.7, 1.1], # 2. fv respect to LAI: f(LAI) * fv [0.01, 1],  # 2. fv
        [0.1, 5],  # 3. h_V (m)
        [1, 7],  # 4. row_sep (m)
        [0.5, 1.5],  # 5. x_LAD
        [0.005, 0.2],  # 6. leaf_width (NOT DEFINED in screenshot - fill if needed) (m)
        [-2, 10],  # 7. Trad = Tair + diff  (Tair 283–313, diff 0–15 → max = 328)
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

df_docker = []
for x in np.arange(1000, 16000, 1000):
    # 2) Saltelli sampling
    print('runing {} samples'.format(x))
    N = x # or 2000, 5000… (trade-off cost vs accuracy)
    second_order = False

    X = saltelli.sample(problem, N, calc_second_order=second_order)

    def model_from_iso(row):
        (LAI, fv, h_V, row_sep, x_LAD,
         leaf_width, Trad, Tair, Sdn, u,
         P_atm, ea, sza_degrees, saa_degrees, G_ratio,
         row_azimuth, emis_leaf, emis_soil, rho_vis_leaf, tau_vis_leaf,
         rho_nir_leaf, tau_nir_leaf, rho_vis_soil, rho_nir_soil) = row
        return sensitivity_analysis_myTSEB(LAI, fv, h_V, row_sep, x_LAD,
                                           leaf_width, Trad, Tair, Sdn, u,
                                           P_atm, ea, sza_degrees, saa_degrees, G_ratio,
                                           row_azimuth, emis_leaf, emis_soil, rho_vis_leaf, tau_vis_leaf,
                                           rho_nir_leaf, tau_nir_leaf, rho_vis_soil, rho_nir_soil)

    Y = np.array([model_from_iso(row) for row in X], dtype=float)

    assert Y.ndim == 1, f"Y must be 1-D; got {Y.shape}"
    assert Y.shape[0] == X.shape[0], f"len(Y)={Y.shape[0]} vs nsamples={X.shape[0]}"

    # 5) Sobol analysis
    Si = sobol.analyze(problem, Y, calc_second_order=second_order, print_to_console=False)

    si_df = pd.DataFrame(Si)
    si_df.loc[:, 'params'] = problem["names"]
    si_df.loc[:, 'N'] = N

    # print(si_df.shape)
    df_docker.append(si_df)
    for n, s1, st, c1, ct in zip(problem["names"], Si["S1"], Si["ST"], Si["S1_conf"], Si["ST_conf"]):
        print(f"{n:>12s} | S1={s1:.3f} (±{c1:.3f})  ST={st:.3f} (±{ct:.3f})")


si_total_df = pd.concat(df_docker)
path_si = rf'files/outputs_sensitivity_analysis_C&NH.csv'
si_total_df.to_csv(path_si)