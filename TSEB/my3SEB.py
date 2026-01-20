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


def estimate_lai_2_fv(LAI, fv_var, x_LAD):
    K_be = myTSEB.estimate_Kbe(x_LAD, 0)
    fv02 = np.clip((1 - np.exp(-K_be * LAI)) * fv_var, 1e-6, 1)
    return fv02


def sensitivity_analysis_my3SEB(LAI_ov, fv_var_ov, h_V_ov, x_LAD_ov, leaf_width_ov,
                                LAI_un, fv_var_un, h_V_un, x_LAD_un, leaf_width_un,
                                Trad_var, row_sep,
                                Tair, Sdn, u,
                                P_atm, ea, sza_degrees, saa_degrees, G_ratio, row_azimuth,
                                emis_leaf, emis_soil, rho_vis_leaf, tau_vis_leaf,
                                rho_nir_leaf, tau_nir_leaf, rho_vis_soil, rho_nir_soil,
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
    Trad_sub_0 = myTSEB.estimate_Trad_S(
        Trad=Trad,
        Trad_V=Trad_ov_0,
        f_theta=f_ov_theta
    )
    # params.update({'Trad_S_0': Trad_S_0})

    R_A, R_x, R_sub = TSEB.calc_resistances(
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
                "deltaT": Trad_sub_0 - T_AC,
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

    if np.any(np.array([R_A, R_x, R_sub])<=0):
        print([R_A, R_x, R_sub])

    # Check how the Radiation interact between sources
    LAI_ov_eff = LAI_ov * omega_ov
    LAI_un_eff = LAI_ov * omega_un

    difvis, difnir, fvis, fnir = TSEB.rad.calc_difuse_ratio(
        S_dn=Sdn,
        sza=sza_degrees,
        press=P_atm
    )

    skyl = fvis * difvis + fnir * difnir
    Sdn_dir = (1. - skyl) * Sdn
    Sdn_dif = skyl * Sdn


    albb_ov, albd_ov, taubt_ov, taudt_ov = rad.calc_spectra_Cambpell(lai=LAI_ov,
                                                         sza=sza_degrees,
                                                         rho_leaf=np.array((rho_vis_leaf, rho_nir_leaf)),
                                                         tau_leaf=np.array((tau_vis_leaf, tau_nir_leaf)),
                                                         rho_soil=np.array((rho_vis_soil, rho_nir_soil)),
                                                         x_lad=x_LAD_ov,
                                                         lai_eff=LAI_ov_eff)

    albb_un, albd_un, taubt_un, taudt_un = rad.calc_spectra_Cambpell(lai=LAI_un,
                                                         sza=sza_degrees,
                                                         rho_leaf=np.array((rho_vis_leaf, rho_nir_leaf)),
                                                         tau_leaf=np.array((tau_vis_leaf, tau_nir_leaf)),
                                                         rho_soil=np.array((rho_vis_soil, rho_nir_soil)),
                                                         x_lad=LAI_un,
                                                         lai_eff=LAI_un_eff)

    S_dir_vis_un = taubt_ov[0] * Sdn_dir * fvis
    S_dif_vis_un = taudt_ov[0] * Sdn_dif * fvis
    S_dir_nir_un = taubt_ov[1] * Sdn_dir * fnir
    S_dif_nir_un = taudt_ov[1] * Sdn_dif * fnir

    Sn_ov = ((1.0 - taubt_ov[0]) * (1.0 - albb_ov[0]) * Sdn_dir * fvis
            + (1.0 - taubt_ov[1]) * (1.0 - albb_ov[1]) * Sdn_dir * fnir
            + (1.0 - taudt_ov[0]) * (1.0 - albd_ov[0]) * Sdn_dif * fvis
            + (1.0 - taudt_ov[1]) * (1.0 - albd_ov[1]) * Sdn_dif * fnir)

    Sn_un = ((1.0 - taubt_un[0]) * (1.0 - albb_un[0]) * S_dir_vis_un
            + (1.0 - taubt_un[1]) * (1.0 - albb_un[1]) * S_dir_nir_un
            + (1.0 - taudt_un[0]) * (1.0 - albd_un[0]) * S_dif_vis_un
            + (1.0 - taudt_un[1]) * (1.0 - albd_un[1]) * S_dif_nir_un)


    # Rn_V0, Rn_S0 = myTSEB.estimate_Rn(
    #     S_dn=Sdn,
    #     sza=sza_degrees,
    #     P_atm=P_atm,
    #     LAI=LAI_ov,
    #     x_LAD=x_LAD_ov,
    #     omega=omega_ov,
    #     Tair=Tair,
    #     ea=ea,
    #     Trad_S=Trad_sub_0,
    #     Trad_V=Trad_ov_0,
    #     rho_vis_leaf=rho_vis_leaf,
    #     rho_nir_leaf=rho_nir_leaf,
    #     tau_vis_leaf=tau_vis_leaf,
    #     tau_nir_leaf=tau_nir_leaf,
    #     rho_vis_soil=rho_vis_soil,
    #     rho_nir_soil=rho_nir_soil,
    #     emis_leaf=emis_leaf,
    #     emis_soil=emis_soil
    # )

    return None
