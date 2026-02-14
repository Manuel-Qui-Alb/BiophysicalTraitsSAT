import pyTSEB.TSEB as TSEB
from pyTSEB import net_radiation as rad
import numpy as np
import pyTSEB.meteo_utils as met
import warnings


def get_emiss_rho_tau(list_ref):
    var_shape = list_ref.shape
    emis_leaf = np.full(var_shape, 0.97)  # Canopy emissivity
    emis_soil = np.full(var_shape, 0.95)  # Soil emissivity

    rho_vis_leaf = np.full(var_shape, 0.07)
    tau_vis_leaf = np.full(var_shape, 0.08)
    rho_nir_leaf = np.full(var_shape, 0.32)
    tau_nir_leaf = np.full(var_shape, 0.33)
    rho_vis_soil = np.full(var_shape, 0.15)
    rho_nir_soil = np.full(var_shape, 0.25)

    z0_soil = np.full(var_shape, 0.01)

    return (emis_leaf, emis_soil, rho_vis_leaf, tau_vis_leaf, rho_nir_leaf, tau_nir_leaf, rho_vis_soil,
            rho_nir_soil, z0_soil)


def estimate_Kbe(x_LAD, sza):
    """
    Estimate the beam extinction coefficient (Kbe) using the
    ellipsoidal leaf angle distribution model from Campbell (1986).

    Parameters
    ----------
    x_LAD : float
        Ellipsoidal leaf angle distribution parameter (dimensionless).
        x = 1 for spherical LAD, x < 1 for more vertical leaves,
        and x > 1 for more horizontal leaves.
    sza : float
        Solar zenith angle in radians.

    Returns
    -------
    float
        Extinction coefficient for direct beam radiation (dimensionless).
    """
    K_be = np.sqrt(x_LAD ** 2 + np.tan(sza) ** 2) / (x_LAD + 1.774 * (x_LAD + 1.182) ** -0.733)
    K_be = np.clip(K_be, 1e-6, None)
    return K_be


def nadir_clumpling_index_Kustas_Norman(LAI, fv, x_LAD, sza):
    """
    Estimate the clumpling index from Campbell and Norman (1998) and Parry (2019).

    Parameters
    ----------
    x_LAD : float
        Leaf Area Index
    fv : float
        Fractional Vegetation Cover.
    x_LAD : float
        Ellipsoidal leaf angle distribution parameter (dimensionless).
        x = 1 for spherical LAD, x < 1 for more vertical leaves,
        and x > 1 for more horizontal leaves.
    sza : float
        Solar zenith angle in radians.
    Returns
    -------
    float
        Omega0: Nadir clumping index (dimensionless).
    """
    LAI = np.clip(LAI, 1e-6, None)
    fv = np.clip(fv, 0.0, 1.0)

    ### Calculating Clumping Index
    K_be = estimate_Kbe(x_LAD, sza)

    # Calculate the gap fraction of our canopy
    f_gap = np.exp(-K_be * LAI)
    f_gap = np.where(np.abs(f_gap) < 1e-6, 1e-6, f_gap)

    log_arg = (1.0 - fv) + (fv * f_gap)
    log_arg = np.clip(log_arg, 1e-12, None)
    omega0 = np.log(log_arg) / -(K_be * LAI)
    omega0 = np.clip(omega0, 0.05, 2)
    return omega0


def off_nadir_clumpling_index_Kustas_Norman(LAI, fv, h_V, w_V, x_LAD, sza):
    """
    Estimate the off nadir clumpling index from Campbell and Norman (1998) and Parry (2019).

    Parameters
    ----------
    LAI : float
        Leaf Area Index.
    fv : float
        Fractional Vegetation Cover.
    h_V: float
        Vegetation Height.
    w_V: float
        Vegetation width.
    x_LAD : float
        Ellipsoidal leaf angle distribution parameter (dimensionless).
        x = 1 for spherical LAD, x < 1 for more vertical leaves,
        and x > 1 for more horizontal leaves.
    sza : float
        Solar zenith angle in radians.
    Returns
    -------
    float
        Omega: Off-Nadir clumping index (dimensionless).
    """
    omega0 = nadir_clumpling_index_Kustas_Norman(LAI, fv, x_LAD, sza)

    D = np.clip(h_V / w_V, 1.0, 3.34)
    p = 3.80 - 0.46 * D

    omega = omega0 / (omega0 + (1 - omega0) * np.exp(-2.2 * sza ** p))
    omega = np.clip(omega, 0.05, 2)
    return omega


def rectangular_row_clumping_index_parry(LAI, fv0, w_V, h_V, sza, phi, hb_V=0, L=None, x_LAD=1 ):
    """
    Estimate the off nadir clumpling index from Campbell and Norman (1998) and Parry (2019).

    Parameters
    ----------
    LAI : float
        Leaf Area Index.
    f_V0:
        Apparent nadir fractional cover
    w_V: float
        Vegetation width.
    h_V: float
        Vegetation Height.
    hb_V: float
        The height of the first living branch.
    L: float
        row separation (m)
    x_LAD : float
        Ellipsoidal leaf angle distribution parameter (dimensionless).
        x = 1 for spherical LAD, x < 1 for more vertical leaves,
        and x > 1 for more horizontal leaves.
    sza : float
        Solar zenith angle in radians.
    phi : float
        relative azimuth angle between the incident beam and the row direction (radians)
        np.arc(np.cos(row direction - solar azimuth))

    Returns
    -------
    float
        Omega: Off-Nadir clumping index (dimensionless).
    """

    # phi_radians = np.arccos(np.cos(saa_radians - row_azimuth_radians))

    K_be = estimate_Kbe(x_LAD, sza)

    # Solar canopy view factor f_sc(theta, phi) Eq. 15
    alpha = np.tan(sza) * np.abs(np.sin(phi))

    try:
        f_sc = (w_V + (h_V - hb_V) * alpha) / L
    except:
        f_sc = fv0 * (1 + ( (h_V - hb_V) * alpha ) / w_V )

    f_sc = np.clip(f_sc, 0.0, 1.0)

    # The gep fraction of the real-world canopy Eq 13.
    gap_phi = f_sc * np.exp(-K_be * LAI) + (1 - f_sc)
    gap_phi = np.clip(gap_phi, 0, 1.0)

    omega_row = -np.log(gap_phi) / (K_be * LAI)
    omega_row = np.clip(omega_row, 0.05, 2)
    return omega_row


def estimate_f_theta(LAI, x_LAD, omega, sza, vza=0):
    """
    Estimate the clumpling index from Campbell and Norman (1998) and Parry (2019).

    Parameters
    ----------
    LAI : float
        Leaf Area Index
    x_LAD : float
        Ellipsoidal leaf angle distribution parameter (dimensionless).
        x = 1 for spherical LAD, x < 1 for more vertical leaves,
        and x > 1 for more horizontal leaves.
    omega : float
        Clumping Index.
    vza : float
        view zenith angle in radians.
    Returns
    -------
    float
        fraction of incident beam radiation intercepted by the plant at view zenith angle (θ)
    """

    ### Calculating intercepted radiation by the canopy
    K_be = estimate_Kbe(x_LAD, sza)

    # omega0 = calculate_omega0(LAI, fv, x_LAD, theta_s)
    f_theta = 1 - np.exp( (-K_be * omega * LAI) / np.cos(vza))
    f_theta = np.clip(f_theta, 0.01, 1)
    return f_theta


def estimate_Trad_S(Trad, Trad_V, f_theta):
    """
    Estimate soil radiometric temperature following Norman and Kustas (1995).

    Parameters
    ----------
    Trad : float
        radiometric temperature in Kelvin.
    Trad_V : float
        Radiometric Vegetation Temperature in Kelvin.
    f_theta : float
        fraction of incident beam radiation intercepted by the plant at view zenith angle (θ)
    Returns
    -------
    float
        Extinction coefficient for direct beam radiation (dimensionless).
    """
    # --- Radiative consistency check ---
    Trad_V_f = (Trad_V ** 4) * f_theta
    Trad_4 = (Trad ** 4)


    Trad_diff = (Trad_4 - Trad_V_f)
    invalid_rad = Trad_diff < 0

    if np.any(invalid_rad):
        warnings.warn(
            "Radiative inconsistency detected: Trad_V > Trad "
            "leading to negative term in Trad_S estimation. "
            "Setting Trad_S = 0 for these cases.",
            RuntimeWarning
        )
        Trad_diff[invalid_rad] = 0

    Trad_S = (Trad_diff / (1 - f_theta)) ** (1 / 4)
    return Trad_S


def estimate_Rn(Sdn_dir, Sdn_dif, fvis, fnir, sza, LAI, Trad_S, Trad_V,
                Tair, ea, P_atm=1013, omega=1, x_LAD=1,
                rho_vis_leaf=0.07, rho_nir_leaf=0.32, tau_vis_leaf=0.08, tau_nir_leaf=0.33, rho_vis_soil=0.15,
                rho_nir_soil=0.25, emis_leaf=0.98, emis_soil=0.95):
    """
    Estimate net radiation (Rn)
    Parameters
    ----------
    S_dn : float
        Incoming shortwave radiation at the surface (W m-2).
    sza : float
        Solar zenith angle in degrees.
    LAI : float
        Leaf Area Index.
    Trad_S : float
        Soil radiometric temperature (K).
    Trad_V : float
        Vegetation radiometric temperature (K).
    Tair : float
        Air temperature (K).
    ea : float
        Vapor pressure (mba depending on implementation).
    P_atm : float, optional
        atmospheric pressure (mb), default at sea level (1013mb).
    omega : float, optional
        Clumping index (default = 1 = random canopy).
    x_LAD : float, optional
        Leaf angle distribution parameter for ellipsoidal LAD model.
    rho_vis_leaf, rho_nir_leaf : float
        Leaf reflectance in VIS and NIR.
    tau_vis_leaf, tau_nir_leaf : float
        Leaf transmittance in VIS and NIR.
    rho_vis_soil, rho_nir_soil : float
        Soil reflectance in VIS and NIR.
    emis_leaf, emis_soil : float
        Longwave emissivity of vegetation and soil.

    Returns
    -------
    float
        Estimated net radiation (W m-2).
    """
    LAI_eff = LAI * omega

    Sn_V, Sn_S = shortwave_transmittance_model_CN(
        Sdn_dir=Sdn_dir,
        Sdn_dif=Sdn_dif,
        fvis=fvis,
        fnir=fnir,
        sza=sza,
        LAI=LAI,
        omega=omega,
        x_LAD=x_LAD,
        rho_vis_leaf=rho_vis_leaf,
        rho_nir_leaf=rho_nir_leaf,
        tau_vis_leaf=tau_vis_leaf,
        tau_nir_leaf=tau_nir_leaf,
        rho_vis_soil=rho_vis_soil,
        rho_nir_soil=rho_nir_soil)

    # sn_veg[~np.isfinite(Sn_V)] = 0
    # sn_soil[~np.isfinite(sn_soil)] = 0

    Ln = rad.calc_longwave_irradiance(ea, Tair, p=P_atm, z_T=2.0, h_C=2.0)
    Ln_V, Ln_S = rad.calc_L_n_Campbell(Trad_V, Trad_S, Ln, LAI_eff, emis_leaf, emis_soil, x_LAD=x_LAD)

    Rn_V = Sn_V + Ln_V
    Rn_S = Sn_S + Ln_S

    # Rn_V = np.array(np.clip(Rn_V, -150, S_dn + Ln))
    # Rn_S = np.array(np.clip(Rn_S, -150, S_dn + Ln))
    return  Sn_V, Sn_S, Rn_V, Rn_S


def shortwave_transmittance_model_CN(Sdn_dir, Sdn_dif,fvis, fnir, sza,
                                     LAI, omega=1, x_LAD=1,
                                     rho_vis_leaf=0.07, rho_nir_leaf=0.32, tau_vis_leaf=0.08, tau_nir_leaf=0.33,
                                     rho_vis_soil=0.15, rho_nir_soil=0.25):
    """
       Estimate net radiation (Rn)
       Parameters
       ----------
       S_dn : float
           Incoming shortwave radiation at the surface (W m-2).
       sza : float
           Solar zenith angle in degrees.
       LAI : float
           Leaf Area Index.
       P_atm : float, optional
           atmospheric pressure (mb), default at sea level (1013mb).
       omega : float, optional
           Clumping index (default = 1 = random canopy).
       x_LAD : float, optional
           Leaf angle distribution parameter for ellipsoidal LAD model.
       rho_vis_leaf, rho_nir_leaf : float
           Leaf reflectance in VIS and NIR.
       tau_vis_leaf, tau_nir_leaf : float
           Leaf transmittance in VIS and NIR.
       rho_vis_soil, rho_nir_soil : float
           Soil reflectance in VIS and NIR.
       Returns
       -------
       float
           Estimated Vegetation and ground Shortwave Net Radiation (W m-2).
       """

    LAI_eff = LAI * omega
    # difvis, difnir, fvis, fnir = TSEB.rad.calc_difuse_ratio(
    #     S_dn=S_dn,
    #     sza=sza,
    #     press=P_atm
    # )
    #
    # skyl = fvis * difvis + fnir * difnir
    # Sdn_dir = (1. - skyl) * S_dn
    # Sdn_dif = skyl * S_dn

    albb, albd, taubt, taudt = rad.calc_spectra_Cambpell(lai=LAI,
                                                         sza=sza,
                                                         rho_leaf=np.array((rho_vis_leaf, rho_nir_leaf)),
                                                         tau_leaf=np.array((tau_vis_leaf, tau_nir_leaf)),
                                                         rho_soil=np.array((rho_vis_soil, rho_nir_soil)),
                                                         x_lad=x_LAD,
                                                         lai_eff=LAI_eff)

    Sn_V = ((1.0 - taubt[0]) * (1.0 - albb[0]) * Sdn_dir * fvis
            + (1.0 - taubt[1]) * (1.0 - albb[1]) * Sdn_dir * fnir
            + (1.0 - taudt[0]) * (1.0 - albd[0]) * Sdn_dif * fvis
            + (1.0 - taudt[1]) * (1.0 - albd[1]) * Sdn_dif * fnir)

    Sn_S = (taubt[0] * (1.0 - rho_vis_soil) * Sdn_dir * fvis
            + taubt[1] * (1.0 - rho_nir_soil) * Sdn_dir * fnir
            + taudt[0] * (1.0 - rho_vis_soil) * Sdn_dif * fvis
            + taudt[1] * (1.0 - rho_nir_soil) * Sdn_dif * fnir)

    Sn_V = np.asarray(Sn_V)
    Sn_S = np.asarray(Sn_S)

    return Sn_V, Sn_S

def Priestly_Taylor_LE_V(fv_g, Rn_V, alpha_PT, Tair, P_atm, c_p):
    """
    ----------
    fv_g: float
        fraction of vegetation that is green and hence transpiring
    VPD : float
        Vapour pressure deficit.
    Rn_V : float
        Vegetation (Canopy) Net Radiation  (w m-2).
    alpha_PT : float
        Priestley-Taylor parameter (mb).
    Tair : float
        air temperature at reference height (Kelvin).
    P_atm : float
        Atmospheric Pressure.
    c_p : float
        specific heat of air.
    Returns
    -------
    float
        LE_V = Vegetation (Canopy) Latent Heat Flux (W m-2).


    """
    # slope of the saturation pressure curve (kPa./deg C)
    s = met.calc_delta_vapor_pressure(Tair)
    s = s * 10  # to mb
    # latent heat of vaporisation (J./kg)
    Lambda = met.calc_lambda(Tair)
    # psychrometric constant (mb C-1)
    gama = met.calc_psicr(c_p, P_atm, Lambda)
    s_gama = s / (s + gama)
    LE_V = Rn_V * alpha_PT * fv_g * s_gama
    return LE_V


def g_S_monteith_1995(LAI, fv_g, Rn, G, Rn_V, Rn_S, Tair, R_A, R_x, R_S, P_atm, ea, VPD, g_M, El_M, f_t=1):
    """
    Leaf stomatal conductance by its relationship with VPD
    following Kustas et al., 2022: https://link.springer.com/article/10.1007/s00271-022-00778-y
    Parameters
    ----------
    LAI : float
        Leaf Area Index.
    fv_g: float
        fraction of vegetation that is green and hence transpiring
    VPD : float
        Vapour pressure deficit.
    P_atm : float
        total air pressure (dry air + water vapour) (mb).
    ea : float
        water vapor pressure at reference height above canopy (mb).
    T_A_K : float
        air temperature at reference height (Kelvin).

    g_M : float
        maximum stomata conductance.
    El_M : float
        Maximum rate of leaf transpiration.
    f_t: int
        Stomatao distribution in the leaves.
        1 for hypostomatous
        2 for amphistomatous leaves
    Returns
    -------
    float
        Estimated net radiation (W m-2).

    """

    g_S  = g_M / (1 + g_M * VPD / El_M) # EQUATION 3 Kustas et al. 2022: Leaf stomatal conductance
    R_C = 1 / (f_t * fv_g * LAI * g_S) # EQUATION 4 Kustas et al. 2022: Canopy Resisteance

    # slope of the saturation pressure curve (kPa./deg C)
    s = met.calc_delta_vapor_pressure(Tair)
    s = s * 10  # to mb
    rho = met.calc_rho(P_atm, ea, Tair)
    c_p = 1013 #specific heat of air (assumed constant at 1013 J kg−1 K−1)

    # psychrometric constant (mb C-1)
    # latent heat of vaporisation (J./kg)
    Lambda = met.calc_lambda(Tair)
    gamma = met.calc_psicr(c_p, P_atm, Lambda)

    PM_C_num = s * (Rn - G) + (rho * c_p * VPD - s * R_x * (Rn_S - G)) / (R_A + R_x) # EQUATION 9a Kustas et al. 2022
    PM_C_den = s + gamma * (1.0 + R_C / (R_A + R_x)) # EQUATION 9a Kustas et al. 2022
    PM_C = PM_C_num / PM_C_den # EQUATION 9a Kustas et al. 2022

    PM_S_num = (
        s * (Rn - G)
        + (rho * c_p * VPD - s * R_S * Rn_V) / (R_A + R_S)
    ) # EQUATION 9b Kustas et al. 2022
    PM_S_den = s + gamma * (1.0 + r_ss / (R_A + R_S)) # EQUATION 9b Kustas et al. 2022
    PM_S = PM_S_den / PM_S_num # EQUATION 9b Kustas et al. 2022

    LE = C_C * PM_C + C_S * PM_S # EQUATION 8 Kustas et al. 2022

    VPD0_pre = ( s * (Rn - G) - (s + gamma) * LE ) / rho * c_p # EQUATION 7 Kustas et al. 2022
    VPD0 = VPD + R_A * VPD0_pre # EQUATION 7 Kustas et al. 2022

    LE_C = (s * Rn_V + rho * c_p * VPD0) / (s + gamma (1 + R_C / R_x)) # EQUATION 6 Kustas et al. 2022
    return LE_C