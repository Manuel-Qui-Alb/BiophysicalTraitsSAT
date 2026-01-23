import numpy as np
import pandas as pd
import csv
import os

import functions as myTSEB
import pyTSEB.meteo_utils as met
from pyTSEB import resistances as res
from pyTSEB import MO_similarity as MO
import pyTSEB.TSEB as TSEB
from pyTSEB import net_radiation as rad

from SALib.sample import saltelli
from SALib.analyze import sobol


FLAG_OK                = 0
FLAG_NO_CONVERGENCE     = 1 << 0   # 1
FLAG_RAD_INCONSISTENCY  = 1 << 1   # 2
FLAG_RES_INVALID        = 1 << 2   # 4
FLAG_NUMERICAL          = 1 << 3   # 8
FLAG_OPTICS_INVALID     = 1 << 4   # 16
FLAG_LEV_NEGATIVE      = 1 << 5   # 32
FLAG_LES_NEGATIVE      = 1 << 6   # 64
FLAG_LETOTAL_NEGATIVE  = 1 << 7   # 128

def estimate_lai_2_fv(LAI, fv_var, x_LAD):
    K_be = myTSEB.estimate_Kbe(x_LAD, 0)
    fv02 = np.clip((1 - np.exp(-K_be * LAI)) * fv_var, 1e-6, 1)
    return fv02

def estimate_Rn_3SEB(LAI_ov, LAI_un, omega_ov, omega_un, x_LAD_ov, x_LAD_un, fv_ov, fv_un,
                     Trad_ov, Trad_ground,
                     Sdn, sza_degrees, Tair, ea, P_atm,
                     rho_vis_soil, rho_nir_soil,
                     rho_vis_leaf_ov, rho_nir_leaf_ov,
                     tau_vis_leaf_ov, tau_nir_leaf_ov,
                     rho_vis_leaf_un, rho_nir_leaf_un,
                     tau_vis_leaf_un, tau_nir_leaf_un,
                     emis_leaf_ov, emis_soil):
        # Check how the Radiation interact between sources
    LAI_ov_eff = LAI_ov * omega_ov
    LAI_un_eff = LAI_un * omega_un

    difvis, difnir, fvis, fnir = TSEB.rad.calc_difuse_ratio(
            S_dn=Sdn,
            sza=sza_degrees,
            press=P_atm
    )

    skyl = fvis * difvis + fnir * difnir
    Sdn_dir = (1. - skyl) * Sdn
    Sdn_dif = skyl * Sdn

    rho_soil = np.array((rho_vis_soil, rho_nir_soil))

    albb_un, albd_un, taubt_un, taudt_un = rad.calc_spectra_Cambpell(lai=LAI_un,
                                                                     sza=sza_degrees,
                                                                     rho_leaf=np.array(
                                                                         (rho_vis_leaf_un, rho_nir_leaf_un)),
                                                                     tau_leaf=np.array(
                                                                             (tau_vis_leaf_un, tau_nir_leaf_un)),
                                                                     rho_soil=rho_soil,
                                                                     x_lad=x_LAD_un,
                                                                     lai_eff=LAI_un_eff)

    rho_ground_b = fv_un * albb_un + (1 - fv_un) * rho_soil
    rho_ground_d = fv_un * albd_un + (1 - fv_un) * rho_soil
    #
    # rho_vis = rho_b_un[0] * fvis + rho_d_un[0] * fnir
    # rho_nir = rho_b_un[1] * fvis + rho_d_un[1] * fnir

    # Beam-conditioned background
    albb_ov_b, albd_ov_b, taubt_ov_b, taudt_ov_b = rad.calc_spectra_Cambpell(
        lai=LAI_ov,
        sza=sza_degrees,
        rho_leaf=np.array((rho_vis_leaf_ov, rho_nir_leaf_ov)),
        tau_leaf=np.array((tau_vis_leaf_ov, tau_nir_leaf_ov)),
        rho_soil=rho_ground_b,
        x_lad=x_LAD_ov,
        lai_eff=LAI_ov_eff
    )

    # Diffuse-conditioned background
    albb_ov_d, albd_ov_d, taubt_ov_d, taudt_ov_d = rad.calc_spectra_Cambpell(
        lai=LAI_ov,
        sza=sza_degrees,
        rho_leaf=np.array((rho_vis_leaf_ov, rho_nir_leaf_ov)),
        tau_leaf=np.array((tau_vis_leaf_ov, tau_nir_leaf_ov)),
        rho_soil=rho_ground_d,
        x_lad=x_LAD_ov,
        lai_eff=LAI_ov_eff
    )

    # Keep the “right” outputs from each run:
    albb_ov = albb_ov_b
    taubt_ov = taubt_ov_b
    albd_ov = albd_ov_d
    taudt_ov = taudt_ov_d

    Sn_ov = ((1.0 - taubt_ov[0]) * (1.0 - albb_ov[0]) * Sdn_dir * fvis
                 + (1.0 - taubt_ov[1]) * (1.0 - albb_ov[1]) * Sdn_dir * fnir
                 + (1.0 - taudt_ov[0]) * (1.0 - albd_ov[0]) * Sdn_dif * fvis
                 + (1.0 - taudt_ov[1]) * (1.0 - albd_ov[1]) * Sdn_dif * fnir)

    S_dir_vis_un = taubt_ov[0] * Sdn_dir * fvis
    S_dif_vis_un = taudt_ov[0] * Sdn_dif * fvis
    S_dir_nir_un = taubt_ov[1] * Sdn_dir * fnir
    S_dif_nir_un = taudt_ov[1] * Sdn_dif * fnir

    Sn_un = ((1.0 - taubt_un[0]) * (1.0 - albb_un[0]) * S_dir_vis_un
                 + (1.0 - taubt_un[1]) * (1.0 - albb_un[1]) * S_dir_nir_un
                 + (1.0 - taudt_un[0]) * (1.0 - albd_un[0]) * S_dif_vis_un
                 + (1.0 - taudt_un[1]) * (1.0 - albd_un[1]) * S_dif_nir_un)

    Sn_S = (taubt_un[0] * (1.0 - rho_vis_soil) * S_dir_vis_un
                + taubt_un[1] * (1.0 - rho_nir_soil) * S_dir_nir_un
                + taudt_un[0] * (1.0 - rho_vis_soil) * S_dif_vis_un
                + taudt_un[1] * (1.0 - rho_nir_soil) * S_dif_nir_un)

    Ln = rad.calc_longwave_irradiance(ea, Tair, p=P_atm, z_T=2.0, h_C=2.0)
    Ln_ov, Ln_ground = rad.calc_L_n_Campbell(Trad_ov, Trad_ground, Ln, LAI_ov_eff,
                                                 emis_leaf_ov, emis_soil, x_LAD=x_LAD_ov)

    Rn_ov = Sn_ov + Ln_ov
    Rn_ground = (Sn_un + Sn_S) + Ln_ground

    return Rn_ov, Rn_ground


def sensitivity_analysis_my3SEB(LAI_ov, fv_var_ov, h_V_ov, x_LAD_ov, leaf_width_ov,
                                LAI_un, fv_var_un, h_V_un, x_LAD_un, leaf_width_un,
                                Trad_var, row_sep,
                                Tair, Sdn, u,
                                P_atm, ea, sza_degrees, saa_degrees, G_ratio, row_azimuth,
                                emis_leaf_ov, emis_leaf_un, emis_soil,
                                rho_vis_leaf_ov, tau_vis_leaf_ov,
                                rho_nir_leaf_ov, tau_nir_leaf_ov,
                                rho_vis_leaf_un, tau_vis_leaf_un,
                                rho_nir_leaf_un, tau_nir_leaf_un,
                                rho_vis_soil, rho_nir_soil,
                                T_MODEL='CN_H', alpha_PT=1.26):

    sza_rad =  np.radians(sza_degrees)
    # vza_rad = np.radians(vza_degrees)
    saa_rad = np.radians(saa_degrees)
    row_azimuth_rad = np.radians(row_azimuth)

    Trad = Tair + Trad_var
    z_u = h_V_ov + 2
    z_T = h_V_ov + 2
    rho = met.calc_rho(P_atm, ea, Tair)
    T_AC = Tair.copy()

    c_p = met.calc_c_p(P_atm, ea)
    alpha_PT = np.full_like(LAI_ov,  alpha_PT)
    L = np.zeros(Trad.shape) + np.inf
    z0_soil = np.full(LAI_ov.shape, 0.01)
    massman_profile = [0, []]

    # Fv based on LAI. fv and LAI are inherently related.
    # It was considered that the FV can vary between 30 and 1.1 with respect to models based on LAI.
    fv_ov = estimate_lai_2_fv(LAI_ov, fv_var_ov, x_LAD_ov)
    fv_un = estimate_lai_2_fv(LAI_un, fv_var_un, x_LAD_un)

    w_V_ov = fv_ov * row_sep
    F_ov = np.asarray(LAI_ov / fv_ov, dtype=np.float32)

    # Clumpling index for Overstory vegetation
    if T_MODEL == 'CN_H':
        omega_ov = myTSEB.off_nadir_clumpling_index_Kustas_Norman(
            LAI_ov,
            fv_ov,
            h_V_ov,
            w_V_ov,
            x_LAD_ov,
            sza_rad
        )
    elif T_MODEL == 'CN_R':
        omega_ov = myTSEB.rectangular_row_clumping_index_parry(
            LAI=LAI_ov,
            fv0=fv_ov,
            w_V=w_V_ov,
            h_V=h_V_ov,
            sza=sza_rad,
            saa=saa_rad,
            row_azimuth=row_azimuth_rad,
            hb_V=0,
            L=None,
            x_LAD=1
        )

    # Clumpling index for Understory vegetation
    # Does the clumping index have to be estimated for cover crops?
    omega_un = np.full_like(LAI_un, 1)

    f_ov_theta = myTSEB.estimate_f_theta(
        LAI=LAI_ov,
        x_LAD=x_LAD_ov,
        omega=omega_ov,
        sza=sza_rad
    )

    f_un_theta = myTSEB.estimate_f_theta(
        LAI=LAI_un,
        x_LAD=x_LAD_un,
        omega=omega_un,
        sza=sza_rad
    )

    ########################################################################################################################
    # R_x = boundary layer resistance of the complete canopy of leaves
    # R_S = soil-surface resistance
    # R_A = aerodynamic resistance above the canopy
    ########################################################################################################################
    z_0m_ov, d_0_ov = TSEB.res.calc_roughness(
        LAI_ov,
        h_V_ov,
        w_V_ov,
        np.full_like(LAI_ov, TSEB.res.CROP),
        f_c=fv_ov
    )

    d_0_ov[d_0_ov < 0] = 0
    z_0m_ov[z_0m_ov < np.min(z0_soil)] = np.mean(z0_soil)

    KB_1_DEFAULTC = 0
    z_0H_ov = res.calc_z_0H(
        z_0m_ov,
        kB=KB_1_DEFAULTC)

    res_params = [0, {}]
    u_friction = MO.calc_u_star(
        u,
        z_u,
        L,
        d_0_ov,
        z_0m_ov
    )
    U_FRICTION_MIN = np.full_like(LAI_ov, 0.01)
    # params.update({'u_friction': u_friction, 'U_FRICTION_MIN': U_FRICTION_MIN})

    u_friction = np.asarray(np.nanmax([U_FRICTION_MIN, u_friction], axis=0), dtype=np.float32)
    # params.update({'u_friction': u_friction})

    resistance_form = [0, {}]
    res_params = resistance_form[1]
    resistance_form = 0  # KUSTAS_NORMAN_1999 (0), CHOUDHURY_MONTEITH_1988 (1), MCNAUGHTON_VANDERHURK (2), CHOUDHURY_MONTEITH_ALPHA_1988(3)

    ########################################################################################################################
    # Fist Net Radiation estimation assuming Trad_V = Tair
    # This first estimation of Trad_S will be used to estimate R_S, Ln_S and R_S
    ########################################################################################################################
    Trad_ov_0 = np.min([Tair, Trad], axis=0) # We set the first Trad_V equal to Tair, under potential ET conditions.
    Trad_ground_0 = myTSEB.estimate_Trad_S(
        Trad=Trad,
        Trad_V=Trad_ov_0,
        f_theta=f_ov_theta
    )
    # params.update({'Trad_S_0': Trad_S_0})

    R_A, R_x, R_ground = TSEB.calc_resistances(
        resistance_form,
        {
            "R_A": {
                "z_T": z_T,
                "u_friction": u_friction,
                "L": L,
                "d_0": d_0_ov,
                "z_0H": z_0H_ov,
            },
            "R_x": {
                "u_friction": u_friction,
                "h_C": h_V_ov,
                "d_0": d_0_ov,
                "z_0M": z_0m_ov,
                "L": L,
                "F": F_ov,
                "LAI": LAI_ov,
                "leaf_width": leaf_width_ov,
                "z0_soil": z0_soil,
                "massman_profile": massman_profile,
                "res_params": {k: res_params[k] for k in res_params.keys()},
            },
            "R_S": {
                "u_friction": u_friction,
                "h_C": h_V_ov,
                "d_0": d_0_ov,
                "z_0M": z_0m_ov,
                "L": L,
                "F": F_ov,
                # "omega0": omega,
                "LAI": LAI_ov,
                "leaf_width": leaf_width_ov,
                "z0_soil": z0_soil,
                "z_u": z_u,
                "deltaT": Trad_ground_0 - T_AC,
                "u": u,
                "rho": rho,
                "c_p": c_p,
                "f_cover": fv_ov,
                "w_C": w_V_ov,
                "massman_profile": massman_profile,
                "res_params": {k: res_params[k] for k in res_params.keys()},
            },
        },
    )

    if np.any(np.array([R_A, R_x, R_ground])<=0):
        print([R_A, R_x, R_ground])

    Rn_ov, Rn_ground = estimate_Rn_3SEB(LAI_ov, LAI_un, omega_ov, omega_un, x_LAD_ov, x_LAD_un, fv_ov, fv_un,
                                        Trad_ov_0, Trad_ground_0,
                                        Sdn, sza_degrees, Tair, ea, P_atm,
                                        rho_vis_soil, rho_nir_soil,
                                        rho_vis_leaf_ov, rho_nir_leaf_ov,
                                        tau_vis_leaf_ov, tau_nir_leaf_ov,
                                        rho_vis_leaf_un, rho_nir_leaf_un,
                                        tau_vis_leaf_un, tau_nir_leaf_un,
                                        emis_leaf_ov, emis_soil)
    [LE_V_ov, LE_S_ov, H_V_ov, Trad_V_ov,
     LE_V_un, LE_S_un, H_V_un, Trad_V_un,
     H_S, G, Trad_ground] = [np.full_like(LAI_ov, -9999) for i in range(7)]

    loop_con = True
    max_iterations = 13
    iterations = 1
    alpha_condition = np.any(alpha_PT > 0)
    flag = np.full_like(LAI_ov, FLAG_OK)

    while (loop_con and alpha_condition and iterations < max_iterations):
        if iterations > 1:
            Rn_ov0, Rn_ground0 = Rn_ov, Rn_ground

        LE_ov = myTSEB.Priestly_Taylor_LE_V(
            fv_g=f_ov_theta,
            Rn_V=Rn_ov0,
            alpha_PT=alpha_PT,
            Tair=Tair,
            P_atm=P_atm,
            c_p=c_p
        )

        H_ov = Rn_ov - LE_V_ov

        ########################################################################################################################
        # Reestimate of Trad_V and Trad_S using Sensible Heat Flux
        ########################################################################################################################
        Trad_ov = TSEB.calc_T_C_series(
            Tr_K=Trad,
            T_A_K=Tair,
            R_A=R_A,
            R_x=R_x,
            R_S=R_ground,
            f_theta=f_ov_theta,
            H_C=H_V_ov,
            rho=rho,
            c_p=c_p
        )

        Trad_ground = myTSEB.estimate_Trad_S(
            Trad=Trad,
            Trad_V=Trad_ov,
            f_theta=f_ov_theta
        )

        flag_RAD_INCONSISTENCY_test = np.isnan(Trad_ground)
        flag[flag_RAD_INCONSISTENCY_test] = FLAG_RAD_INCONSISTENCY

        ########################################################################################################################
        # Reestimate Soil Sensible Heat Flux (H_S) because it depends on Trad_S
        ########################################################################################################################
        _, _, R_ground = TSEB.calc_resistances(
            resistance_form,
            {
                "R_S": {
                    "u_friction": u_friction,
                    "h_C": h_V_ov,
                    "d_0": d_0_ov,
                    "z_0M": z_0m_ov,
                    "L": L,
                    "F": F_ov,
                    "omega0": omega_ov,
                    "LAI": LAI_ov,
                    "leaf_width": leaf_width_ov,
                    "z0_soil": z0_soil, #Check if this variable should change with cover crop
                    "z_u": z_u,
                    "deltaT": Trad_ground - T_AC,  # Trad_S - T_AC
                    "u": u,
                    "rho": rho,
                    "c_p": c_p,
                    "f_cover": fv_ov,
                    "w_C": w_V_ov,
                    "massman_profile": massman_profile,
                    "res_params": {k: res_params[k] for k in res_params.keys()},
                }
            }
        )

        # 2) Air temperature at canopy interface (T_AC), array form
        T_AC = (
                (Tair / R_A + Trad_ground / R_ground + Trad_ov / R_x) /
                (1.0 / R_A + 1.0 / R_ground + 1.0 / R_x)
        )

        H_ground = rho * c_p * (Trad_ground - T_AC) / R_ground

        ########################################################################################################################
        # Reestimate Net Radiation estimation with Trad_V and Trad_S from Sensible Heat Flux
        ########################################################################################################################
        Rn_ov, Rn_ground = estimate_Rn_3SEB(LAI_ov, LAI_un, omega_ov, omega_un, x_LAD_ov, x_LAD_un, fv_ov, fv_un,
                                            Trad_ov, Trad_ground,
                                            Sdn, sza_degrees, Tair, ea, P_atm,
                                            rho_vis_soil, rho_nir_soil,
                                            rho_vis_leaf_ov, rho_nir_leaf_ov,
                                            tau_vis_leaf_ov, tau_nir_leaf_ov,
                                            rho_vis_leaf_un, rho_nir_leaf_un,
                                            tau_vis_leaf_un, tau_nir_leaf_un,
                                            emis_leaf_ov, emis_soil)

        ########################################################################################################################
        # Compute Soil Heat Flux Ratio as a Ratio of Rn_S
        ########################################################################################################################
        G = G_ratio * Rn_ground

        LE_ground = Rn_ground - G - H_ground
        LE_ov = Rn_ov - H_ov
        LE_total = LE_ov + LE_ground
        # print(rf'LE_S: {LE_S}')

        alpha_PT = np.maximum(alpha_PT - 0.1, 0.0)
        iterations += 1
        alpha_condition = np.any(alpha_PT > 0)
        loop_con = np.any(LE_ground < 0)

        ################################################################################################################
        # Which resistance I am using and what is the effect on T estimation
        # Which resistance I should used for Canopy-Ground and Ground-Soil interaction
        #################################################################################################################


    return None
