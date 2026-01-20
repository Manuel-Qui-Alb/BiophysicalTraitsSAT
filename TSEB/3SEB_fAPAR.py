import numpy as np
import pyTSEB.TSEB as TSEB
from pyTSEB import net_radiation as rad
import functions as myTSEB


def estimate_lai_2_fv(LAI, fv_var, x_LAD):
    K_be = myTSEB.estimate_Kbe(x_LAD, 0)
    fv02 = np.clip((1 - np.exp(-K_be * LAI)) * fv_var, 1e-6, 1)
    return fv02

def fAPAR_3SEB(LAI_ov, fv_ov, h_V_ov, x_LAD_ov,
               LAI_un, fv_un, h_V_un, x_LAD_un,
               row_sep, row_azimuth,
               Sdn, sza_degrees, saa_degrees, P_atm,
               rho_vis_leaf, rho_nir_leaf, tau_vis_leaf, tau_nir_leaf, rho_vis_soil, rho_nir_soil,
               T_MODEL='CN_H'):

    sza_rad =  np.radians(sza_degrees)
    saa_rad = np.radians(saa_degrees)
    row_azimuth_rad = np.radians(row_azimuth)
    # vza_rad = np.radians(vza_degrees)

    # Fv based on LAI. fv and LAI are inherently related.
    # It was considered that the FV can vary between 30 and 1.1 with respect to models based on LAI.
    # fv_ov = estimate_lai_2_fv(LAI_ov, fv_var_ov, x_LAD_ov)
    # fv_un = estimate_lai_2_fv(LAI_un, fv_var_un, x_LAD_un)

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

    omega_un = np.full_like(LAI_un, 1)

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
                                                                     rho_leaf=np.array((rho_vis_leaf, rho_nir_leaf)),
                                                                     tau_leaf=np.array((tau_vis_leaf, tau_nir_leaf)),
                                                                     rho_soil=rho_soil,
                                                                     x_lad=x_LAD_un,
                                                                     lai_eff=LAI_un_eff)

    rho_b_un = fv_ov * albb_un + (1 - fv_ov) * rho_soil
    rho_d_un = fv_ov * albd_un + (1 - fv_ov) * rho_soil

    rho_vis = rho_b_un[0] * fvis + rho_d_un[0] * fnir
    rho_nir = rho_b_un[1] * fvis + rho_d_un[1] * fnir

    albb_ov, albd_ov, taubt_ov, taudt_ov = rad.calc_spectra_Cambpell(lai=LAI_ov,
                                                                     sza=sza_degrees,
                                                                     rho_leaf=np.array((rho_vis_leaf, rho_nir_leaf)),
                                                                     tau_leaf=np.array((tau_vis_leaf, tau_nir_leaf)),
                                                                     rho_soil=np.array([rho_nir, rho_vis]),
                                                                     x_lad=x_LAD_ov,
                                                                     lai_eff=LAI_ov_eff)

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

    Sn_S = (taubt_un[0] * (1.0 - rho_vis) * S_dir_vis_un
            + taubt_un[1] * (1.0 - rho_nir) * S_dir_nir_un
            + taudt_un[0] * (1.0 - rho_vis) * S_dif_vis_un
            + taudt_un[1] * (1.0 - rho_nir) * S_dif_nir_un)

    return Sn_ov, Sn_un, Sn_S



Sn_ov, Sn_un, Sn_S = fAPAR_3SEB(LAI_ov=3, fv_ov=0.4, h_V_ov=3, x_LAD_ov=1,
           LAI_un=2, fv_un=0.4, h_V_un=0.1, x_LAD_un=0.5,
           row_sep=7, row_azimuth=0,
           Sdn=1000, sza_degrees=10, saa_degrees=10, P_atm=1013,
           rho_vis_leaf=0.07, rho_nir_leaf=0.32, tau_vis_leaf=0.08, tau_nir_leaf=0.33, rho_vis_soil=0.15, rho_nir_soil=0.25,
           T_MODEL='CN_H')


print(Sn_ov)
