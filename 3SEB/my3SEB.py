
import numpy as np
import TSEB.functions as myTSEB
import pyTSEB.TSEB as TSEB
import GeneralFunctions as GF
from pyTSEB import net_radiation as rad
import pyTSEB.meteo_utils as met
from pyTSEB import MO_similarity as MO
from pyTSEB import resistances as res



def my3SEB(Trad, Tair, u, ea, P_atm, S_dn, sza_degrees, LAI_ov, fv_ov, h_ov, w_ov,
           emis_leaf, emis_soil, rho_vis_leaf, rho_nir_leaf, tau_vis_leaf, tau_nir_leaf, rho_vis_soil, rho_nir_soil,
           z_u, z_T, x_LAD=1.0, leaf_width_ov=0.05, alpha_PT_ov=1.26,
           # optional understory inputs
           LAI_un=None, fv_un=None, h_un=None, w_un=None, leaf_width_un=0.05, alpha_PT_un=1.0, omega0_un=None,
           # options
           const_L=None, max_outer=30, max_it1=20, max_it2=20, step_alpha=0.1, L_thres=0.05, scheme_ov=0
):
    """
        Three-Source Energy Balance (3SEB) model with optional fallback to a
        Two-Source Energy Balance (2SEB) configuration when understory inputs
        are not provided.

        The model partitions the radiometric surface temperature and net radiation
        into three components:
            1) overstory canopy,
            2) understory canopy,
            3) soil.

        If no understory is defined (LAI_un is None or zero), the model switches
        automatically to a 2-source configuration in which the lower layer is
        treated as a single bulk soil/substrate source.

        Parameters
        ----------
        Trad : array_like
            Composite radiometric surface temperature [K] measured for the full scene.

        Tair : array_like
            Air temperature at reference height [K].

        u : array_like
            Wind speed at reference height [m s-1].

        ea : array_like
            Actual vapor pressure [mb or hPa, depending on pyTSEB convention].

        P_atm : array_like
            Atmospheric pressure [mb or hPa].

        S_dn : array_like
            Incoming shortwave radiation [W m-2].

        sza_degrees : array_like
            Solar zenith angle [degrees].

        LAI_ov : array_like
            Overstory leaf area index [m2 m-2].

        fv_ov : array_like
            Fractional cover or effective vegetation fraction of the overstory [-].

        h_ov : array_like
            Overstory canopy height [m].

        w_ov : array_like
            Overstory canopy width-to-height descriptor [-].

        emis_leaf : array_like
            Leaf emissivity [-].

        emis_soil : array_like
            Soil emissivity [-].

        rho_vis_leaf : array_like
            Leaf reflectance in the visible band [-].

        rho_nir_leaf : array_like
            Leaf reflectance in the near-infrared band [-].

        tau_vis_leaf : array_like
            Leaf transmittance in the visible band [-].

        tau_nir_leaf : array_like
            Leaf transmittance in the near-infrared band [-].

        rho_vis_soil : array_like
            Soil reflectance in the visible band [-].

        rho_nir_soil : array_like
            Soil reflectance in the near-infrared band [-].

        z_u : array_like
            Wind measurement height [m].

        z_T : array_like
            Air temperature / humidity measurement height [m].

        x_LAD : float or array_like, optional
            Campbell leaf angle distribution parameter [-].
            Default is 1.0, corresponding approximately to a spherical LAD.

        leaf_width_ov : float or array_like, optional
            Characteristic overstory leaf width [m].
            Default is 0.05 m.

        alpha_PT_ov : float or array_like, optional
            Priestley-Taylor coefficient for the overstory canopy [-].
            Default is 1.26.

        LAI_un : array_like, optional
            Understory leaf area index [m2 m-2].
            If None, the model runs in 2SEB mode and no understory canopy is solved.

        fv_un : array_like, optional
            Fractional cover or effective vegetation fraction of the understory [-].
            If None, it is estimated internally from LAI_un and clumping assumptions.

        h_un : array_like, optional
            Understory canopy height [m].
            Required when LAI_un is provided.

        w_un : array_like, optional
            Understory canopy width-to-height descriptor [-].
            If None, a value of 1 is assumed.

        leaf_width_un : float or array_like, optional
            Characteristic understory leaf width [m].
            Default is 0.05 m.

        alpha_PT_un : float or array_like, optional
            Priestley-Taylor coefficient for the understory canopy [-].
            Default is 1.0.

        omega0_un : array_like, optional
            Understory clumping factor or nadir clumping index [-].
            If None, a default value is assigned internally.

        const_L : float or array_like, optional
            Prescribed Monin-Obukhov length [m].
            If None, atmospheric stability is solved iteratively.

        max_outer : int, optional
            Maximum number of outer coupling iterations between loop-1, loop-2,
            and atmospheric stability. Default is 30.

        max_it1 : int, optional
            Maximum number of inner iterations for loop-1 (overstory vs bulk substrate).
            Default is 20.

        max_it2 : int, optional
            Maximum number of inner iterations for loop-2 (understory vs soil).
            Default is 20.

        step_alpha : float, optional
            Reduction step applied to Priestley-Taylor coefficients when negative
            latent heat is detected. Default is 0.1.

        L_thres : float, optional
            Relative convergence threshold for Monin-Obukhov length [-].
            Default is 0.05.

        scheme_ov : int, optional
            Resistance scheme used in loop-1 for the overstory:
                0 -> parallel overstory/substrate formulation
                1 -> series overstory/substrate formulation
            Default is 0.

        Returns
        -------
        dict
            Dictionary containing component temperatures, radiative fluxes,
            sensible heat fluxes, latent heat fluxes, soil heat flux, aerodynamic
            variables, final Priestley-Taylor coefficients, source masks, and
            convergence status.

        Notes
        -----
        - The model solves the overstory first against a bulk lower-layer source,
          then refines the lower layer into understory and soil when LAI_un > 0.
        - Pixels with LAI_un <= 0 are treated automatically as a 2-source case.
        - Negative soil evaporation can be corrected through an alpha-reduction
          strategy and, if necessary, a final non-negative soil evaporation fallback.

        References
        -----
        - Vicente Burchard-Levine, Héctor Nieto, William P. Kustas, Feng Gao, Joseph G. Alfieri, John H. Prueger,
        Lawrence E. Hipps, Nicolas Bambach-Ortiz, Andrew J. McElrone, Sebastian J. Castro, Maria Mar Alsina,
        Lynn G. McKee, Einara Zahn, Elie Bou-Zeid & Nick Dokoozlian. (2022) Application of a remote-sensing three-source
        energy balance model to improve evapotranspiration partitioning in vineyards.
        https://link.springer.com/article/10.1007/s00271-022-00787-x

        - J.M. Norman ay * , W.P. Kustas b, K.S. Humes b. Source approach for estimating soil and vegetation energy
        fluxes in observations of directional radiometric surface temperature.(1995).
        https://doi.org/10.1016/0168-1923(95)02265-Y

        - William P. Kustas, John M. Norman. Evaluation of soil and vegetation heat flux predictions using a simple
        two-source model with radiometric temperatures for partial canopy cover. (1999).
        https://doi.org/10.1016/S0168-1923(99)00005-2
a
        """
    resistance_form_ov = [0, {}]
    res_params_ov = resistance_form_ov[1]
    # KUSTAS_NORMAN_1999 (0), CHOUDHURY_MONTEITH_1988 (1), MCNAUGHTON_VANDERHURK (2), CHOUDHURY_MONTEITH_ALPHA_1988(3)
    resistance_form_ov = 0
    resistance_form_un = 0
    massman_profile = [0, []]

    # ------------------------------------------------------------------
    # 0) Convert inputs to arrays
    # ------------------------------------------------------------------
    Trad = np.asarray(Trad, dtype=float)
    Tair = np.asarray(Tair, dtype=float)
    u = np.asarray(u, dtype=float)
    ea = np.asarray(ea, dtype=float)
    P_atm = np.asarray(P_atm, dtype=float)
    S_dn = np.asarray(S_dn, dtype=float)
    LAI_ov = np.asarray(LAI_ov, dtype=float)
    h_ov = np.asarray(h_ov, dtype=float)
    w_ov = np.asarray(w_ov, dtype=float)
    z_u = np.asarray(z_u, dtype=float)
    z_T = np.asarray(z_T, dtype=float)


    shape = Trad.shape

    def _as_array(x, default=None):
        if x is None:
            x = default
        x = np.asarray(x, dtype=float)
        if x.shape == ():
            x = np.full(shape, float(x))
        return x

    x_LAD = _as_array(x_LAD, 1.0)
    leaf_width_ov = _as_array(leaf_width_ov, 0.05)

    # ------------------------------------------------------------------
    # 1) Understory defaults -> automatic 2SEB
    # ------------------------------------------------------------------
    if LAI_un is None:
        LAI_un = np.zeros_like(Trad)
        h_un = np.zeros_like(Trad)
        w_un = np.ones_like(Trad)
        omega0_un = np.zeros_like(Trad)
    else:
        LAI_un = _as_array(LAI_un)
        if h_un is None:
            raise ValueError("h_un must be provided when LAI_un is provided")
        h_un = _as_array(h_un)
        w_un = _as_array(1.0 if w_un is None else w_un)
        omega0_un = _as_array(0.8 if omega0_un is None else omega0_un)

    if fv_un is None:
        fv_un = 1 - np.exp(-GF.estimate_Kbe(x_LAD=1, sza=0) * omega0_un * LAI_un)
    else:
        fv_un = _as_array(fv_un)

    leaf_width_un = _as_array(leaf_width_un, 0.05)

    mask_2seb = LAI_un <= 1e-6
    mask_3seb = ~mask_2seb

    # ------------------------------------------------------------------
    # 2) Precompute radiation / geometry / roughness / initial states
    #    (put here all your code before the outer loop)
    # ------------------------------------------------------------------
    ########################################################################################################
    ##################################### Add clumping index #####################################
    ########################################################################################################
    omega0_ov = TSEB.CI.calc_omega0_Kustas(
        LAI_ov,
        fv_ov,
        x_LAD=x_LAD,
        isLAIeff=False
    )

    omega_SZA_ov = TSEB.CI.calc_omega_Kustas(
        omega0_ov,
        theta=sza_degrees,
        w_C=w_ov / h_ov
    )

    LAI_eff_ov = LAI_ov * omega_SZA_ov
    LAI_eff_un = LAI_un * omega0_un

    difvis, difnir, fvis, fnir = TSEB.rad.calc_difuse_ratio(
        S_dn=S_dn,
        sza=sza_degrees,
        press=P_atm
    )

    skyl = fvis * difvis + fnir * difnir
    St_dir = (1. - skyl) * S_dn
    St_dif = skyl * S_dn

    rho_soil = np.array((rho_vis_soil, rho_nir_soil))
    albb_sub, albd_sub, taubt_un, taudt_un = rad.calc_spectra_Cambpell(
        lai=LAI_un,
        sza=sza_degrees,
        rho_leaf=np.array((rho_vis_leaf, rho_nir_leaf)),
        tau_leaf=np.array((tau_vis_leaf, tau_nir_leaf)),
        rho_soil=rho_soil,
        x_lad=x_LAD,
        lai_eff=LAI_eff_un)

    _, _, taubt_ov, taudt_ov = rad.calc_spectra_Cambpell(LAI_ov,
                                                         sza_degrees,
                                                         np.array((rho_vis_leaf, rho_nir_leaf)),
                                                         np.array((tau_vis_leaf, tau_nir_leaf)),
                                                         albb_sub,
                                                         x_lad=x_LAD,
                                                         lai_eff=LAI_eff_ov)

    Lt_ov = np.nan_to_num(LAI_eff_ov / fv_ov)
    # Lt_T = np.full_like(Lt_T, 5)
    albb_ov, albd_ov, _, _ = rad.calc_spectra_Cambpell(Lt_ov,
                                                       sza_degrees,
                                                       np.array((rho_vis_leaf, rho_nir_leaf)),
                                                       np.array((tau_vis_leaf, tau_nir_leaf)),
                                                       albb_sub,
                                                       x_lad=x_LAD,
                                                       lai_eff=None)

    Lt_un = np.nan_to_num(LAI_eff_un / fv_un)
    albb_un, albd_un, _, _ = rad.calc_spectra_Cambpell(Lt_un,
                                                       sza_degrees,
                                                       np.array((rho_vis_leaf, rho_nir_leaf)),
                                                       np.array((tau_vis_leaf, tau_nir_leaf)),
                                                       rho_soil,
                                                       x_lad=x_LAD,
                                                       lai_eff=None)

    Sn_ov = ((1.0 - taubt_ov[0]) * (1.0 - albb_ov[0]) * St_dir * fvis
             + (1.0 - taubt_ov[1]) * (1.0 - albb_ov[1]) * St_dir * fnir
             + (1.0 - taudt_ov[0]) * (1.0 - albd_ov[0]) * St_dif * fvis
             + (1.0 - taudt_ov[1]) * (1.0 - albd_ov[1]) * St_dif * fnir)

    # Sn_t = Sn_ov * taubt_ov[0]
    St_dir_vis_ground = taubt_ov[0] * St_dir * fvis
    St_dir_nir_ground = taubt_ov[1] * St_dir * fnir
    St_dif_vis_ground = taudt_ov[0] * St_dif * fvis
    St_dif_nir_ground = taudt_ov[1] * St_dif * fnir

    Sn_un = ((1.0 - taubt_un[0]) * (1.0 - albb_un[0]) * St_dir_vis_ground
             + (1.0 - taubt_un[1]) * (1.0 - albb_un[1]) * St_dir_nir_ground
             + (1.0 - taudt_un[0]) * (1.0 - albd_un[0]) * St_dif_vis_ground
             + (1.0 - taudt_un[1]) * (1.0 - albd_un[1]) * St_dif_nir_ground)

    Sn_soil = (taubt_un[0] * (1.0 - rho_vis_soil) * St_dir_vis_ground
               + taubt_un[1] * (1.0 - rho_nir_soil) * St_dir_nir_ground
               + taudt_un[0] * (1.0 - rho_vis_soil) * St_dif_vis_ground
               + taudt_un[1] * (1.0 - rho_nir_soil) * St_dif_nir_ground)

    Sn_un = np.asarray(Sn_un)
    Sn_soil = np.asarray(Sn_soil)
    Sn_sub = Sn_un + Sn_soil

    T_ov = np.min([Tair, Trad], axis=0)

    T_sub = myTSEB.estimate_Trad_S(
        Trad=Trad,
        Trad_V=T_ov,
        f_theta=fv_ov
    )

    T_un = (T_sub + Tair) / 2
    T_soil = myTSEB.estimate_Trad_S(
        Trad=T_sub,
        Trad_V=T_un,
        f_theta=fv_un
    )

    L_dn_atm = rad.calc_longwave_irradiance(
        ea,
        Tair,
        p=P_atm,
        z_T=z_T,
        h_C=h_ov
    )

    c_p = met.calc_c_p(P_atm, ea)

    z0_soil = np.full(LAI_ov.shape, 0.01)
    L = np.zeros(Trad.shape) + np.inf

    z_0m_ov, d_0_ov = TSEB.res.calc_roughness(
        LAI_ov,
        h_ov,
        w_ov,
        np.full_like(LAI_ov, TSEB.res.CROP),
        f_c=fv_ov
    )
    d_0_ov[d_0_ov < 0] = 0
    z_0m_ov[z_0m_ov < np.min(z0_soil)] = np.mean(z0_soil)

    z_0m_un, d_0_un = TSEB.res.calc_roughness(
        LAI_un,
        h_C=h_un,
        w_C=np.full_like(LAI_un, 1),
        landcover=np.full_like(LAI_un, TSEB.res.GRASS),
        f_c=fv_un
    )
    d_0_un[d_0_un < 0] = 0
    z_0m_un[z_0m_un < np.min(z0_soil)] = np.mean(z0_soil)

    KB_1_DEFAULTC = 0
    z_0H_ov = res.calc_z_0H(
        z_0m_ov,
        kB=KB_1_DEFAULTC)

    z_0H_un = res.calc_z_0H(
        z_0m_un,
        kB=KB_1_DEFAULTC)

    rho = met.calc_rho(
        P_atm,
        ea,
        Tair
    )
    # T_AC = Tair.copy()

    u_friction = MO.calc_u_star(
        u,
        z_u,
        L,
        d_0_ov,
        z_0m_ov
    )
    U_FRICTION_MIN = np.full_like(LAI_ov, 0.01)

    u_friction = np.asarray(np.nanmax([U_FRICTION_MIN, u_friction], axis=0), dtype=np.float32)

    F_un = np.divide(LAI_un, fv_un, out=np.zeros_like(LAI_un, dtype=float), where=fv_un > 0)
    F_ov = np.divide(LAI_ov, fv_ov, out=np.zeros_like(LAI_ov, dtype=float), where=fv_ov > 0)

    T_AC_ov = Tair.copy()

    [LE_ov, LE_sub, LE_un, LE_soil, LE, H_ov, H_sub, H_soil, H, H_un,
     Ln_ov, Ln_sub, Ln_un, Ln_soil, Rn_ov, Rn_sub, Rn_un, Rn_soil,
     T_AC_un, G,
     R_A_ov, R_S_ov, R_A_un, R_x_un, R_x_ov, R_S_un, AELE_soil, AELE_sub] = [np.full_like(LAI_ov, -9999) for i in
                                                                             range(28)]


    # ------------------------------------------------------------------
    # 3) Internal helpers
    # ------------------------------------------------------------------
    # --- helper: recompute LW transmittance & L_dn_sub below overstory using current T_ov
    def _update_Ldn_sub():
        # tau_LW_ov as diffuse LW transmittance of overstory
        _, _, _, tau_LW_ov_local = rad.calc_spectra_Cambpell(
            lai=LAI_eff_ov,
            sza=0,
            rho_leaf=1.0 - emis_leaf,
            tau_leaf=0.0,
            rho_soil=1.0 - emis_soil,
            x_lad=x_LAD,
            lai_eff=None
        )
        tau_LW_ov_local = np.clip(tau_LW_ov_local, 0.0, 1.0)

        # L_dn just below overstory
        L_dn_sub_local = tau_LW_ov_local * L_dn_atm + (1.0 - tau_LW_ov_local) * emis_leaf * sigma * (T_ov ** 4)
        return L_dn_sub_local, tau_LW_ov_local


    # --- helper: loop-1 (parallel ov/sub) with its own alpha reduction
    def loop_ov_parallel(alpha_PT_i1, max_it1=20):
        i1 = np.full_like(LAI_ov, True, dtype=bool)
        it1 = 0

        while np.any(i1) and it1 <= max_it1:

            # Resistances (parallel K-N99 style)
            R_A_ov[i1], _, R_S_ov[i1] = TSEB.calc_resistances(
                resistance_form_ov,
                {
                    "R_A": {
                        "z_T": z_T[i1],
                        "u_friction": u_friction[i1],
                        "L": L[i1],
                        "d_0": d_0_ov[i1],
                        "z_0H": z_0H_ov[i1],
                    },
                    "R_S": {
                        "u_friction": u_friction[i1],
                        "h_C": h_ov[i1],
                        "d_0": d_0_ov[i1],
                        "z_0M": z_0m_ov[i1],
                        "L": L[i1],
                        "F": F_ov[i1],
                        "omega0": omega0_ov[i1],
                        "LAI": LAI_ov[i1],
                        "leaf_width": leaf_width_ov[i1],
                        "z0_soil": z0_soil[i1],
                        "z_u": z_u[i1],
                        "deltaT": T_sub[i1] - T_ov[i1],
                        "u": u[i1],
                        "rho": rho[i1],
                        "c_p": c_p[i1],
                        "f_cover": fv_ov[i1],
                        "w_C": w_ov[i1] / h_ov[i1],
                        "massman_profile": massman_profile,
                        "res_params": {k: res_params_ov[k] for k in res_params_ov.keys()},
                    },
                },
            )

            # Net LW (ov/sub) using L_inc above overstory
            Ln_ov[i1], Ln_sub[i1] = rad.calc_L_n_Campbell(
                T_C=T_ov[i1],
                T_S=T_sub[i1],
                L_dn=L_dn_atm[i1],
                lai=LAI_eff_ov[i1],
                emisVeg=emis_leaf[i1],
                emisGrd=emis_soil[i1],
                x_LAD=x_LAD[i1],
            )

            Rn_ov[i1] = Sn_ov[i1] + Ln_ov[i1]
            Rn_sub[i1] = Sn_sub[i1] + Ln_sub[i1]

            # PT for ov
            LE_ov[i1] = myTSEB.Priestly_Taylor_LE_V(
                fv_g=1,
                Rn_V=Rn_ov[i1],
                alpha_PT=alpha_PT_i1[i1],
                Tair=Tair[i1],
                P_atm=P_atm[i1],
                c_p=c_p[i1],
            )

            H_ov[i1] = Rn_ov[i1] - LE_ov[i1]

            # Parallel canopy temperature (ov)
            T_ov[i1] = (H_ov[i1] * R_A_ov[i1]) / (rho[i1] * c_p[i1]) + Tair[i1]

            # Radiometric partition to substrate
            T_sub[i1] = myTSEB.estimate_Trad_S(
                Trad=Trad[i1],
                Trad_V=T_ov[i1],
                f_theta=fv_ov[i1],
            )

            # Recompute LW with updated temperatures
            Ln_ov[i1], Ln_sub[i1] = rad.calc_L_n_Campbell(
                T_C=T_ov[i1],
                T_S=T_sub[i1],
                L_dn=L_dn_atm[i1],
                lai=LAI_eff_ov[i1],
                emisVeg=emis_leaf[i1],
                emisGrd=emis_soil[i1],
                x_LAD=x_LAD[i1],
            )

            Rn_ov[i1] = Sn_ov[i1] + Ln_ov[i1]
            Rn_sub[i1] = Sn_sub[i1] + Ln_sub[i1]

            # Recompute R_S_sub with updated deltaT
            _, _, R_S_ov[i1] = TSEB.calc_resistances(
                resistance_form_ov,
                {
                    "R_S": {
                        "u_friction": u_friction[i1],
                        "h_C": h_ov[i1],
                        "d_0": d_0_ov[i1],
                        "z_0M": z_0m_ov[i1],
                        "L": L[i1],
                        "F": F_ov[i1],
                        "omega0": omega0_ov[i1],
                        "LAI": LAI_ov[i1],
                        "leaf_width": leaf_width_ov[i1],
                        "z0_soil": z0_soil[i1],
                        "z_u": z_u[i1],
                        "deltaT": T_sub[i1] - T_ov[i1],
                        "u": u[i1],
                        "rho": rho[i1],
                        "c_p": c_p[i1],
                        "f_cover": fv_ov[i1],
                        "w_C": w_ov[i1] / h_ov[i1],
                        "massman_profile": massman_profile,
                        "res_params": {k: res_params_ov[k] for k in res_params_ov.keys()},
                    }
                },
            )

            # Substrate sensible heat (parallel to air, using Tair)
            H_sub[i1] = (rho[i1] * c_p[i1] * (T_sub[i1] - Tair[i1])) / (R_S_ov[i1] + R_A_ov[i1])

            # Your G scheme for loop-1
            G[i1] = Rn_sub[i1] * (1.0 - fv_un[i1]) * 0.3 * np.exp(-0.5 * LAI_eff_un[i1])
            AELE_sub[i1] = Rn_sub[i1] - G[i1]
            LE_sub[i1] = AELE_sub[i1] - H_sub[i1]
            LE_ov[i1] = Rn_ov[i1] - H_ov[i1]

            H[i1] = H_ov[i1] + H_sub[i1]
            LE[i1] = LE_ov[i1] + LE_sub[i1]

            # Reduce alpha only where LE_sub < 0
            i1 = (LE_sub < 0) & (alpha_PT_i1 > 0)
            alpha_PT_i1[i1] = np.maximum(alpha_PT_i1[i1] - 0.1, 0.0)

            it1 += 1

        return alpha_PT_i1


    def loop_ov_series(alpha_PT_i1, max_it1=20):
        i1 = np.full_like(LAI_ov, True, dtype=bool)
        it1 = 0

        while np.any(i1) and it1 <= max_it1:
            # Resistances (parallel K-N99 style)
            R_A_ov[i1], R_x_ov[i1], R_S_ov[i1] = TSEB.calc_resistances(
                resistance_form_ov,
                {
                    "R_A": {
                        "z_T": z_T[i1],
                        "u_friction": u_friction[i1],
                        "L": L[i1],
                        "d_0": d_0_ov[i1],
                        "z_0H": z_0H_ov[i1],
                    },
                    "R_x": {
                        "u_friction": u_friction[i1],
                        "h_C": h_ov[i1],
                        "d_0": d_0_ov[i1],
                        "z_0M": z_0m_ov[i1],
                        "L": L[i1],
                        "F": F_ov[i1],
                        "LAI": LAI_ov[i1],
                        "leaf_width": leaf_width_ov[i1],
                        "z0_soil": z0_soil[i1],
                        "massman_profile": massman_profile,
                        "res_params": {k: res_params_ov[k] for k in res_params_ov.keys()},
                    },
                    "R_S": {
                        "u_friction": u_friction[i1],
                        "h_C": h_ov[i1],
                        "d_0": d_0_ov[i1],
                        "z_0M": z_0m_ov[i1],
                        "L": L[i1],
                        "F": F_ov[i1],
                        "omega0": omega0_ov[i1],
                        "LAI": LAI_ov[i1],
                        "leaf_width": leaf_width_ov[i1],
                        "z0_soil": z0_soil[i1],
                        "z_u": z_u[i1],
                        "deltaT": T_sub[i1] - T_AC_ov[i1],
                        "u": u[i1],
                        "rho": rho[i1],
                        "c_p": c_p[i1],
                        "f_cover": fv_ov[i1],
                        "w_C": w_ov[i1] / h_ov[i1],
                        "massman_profile": massman_profile,
                        "res_params": {k: res_params_ov[k] for k in res_params_ov.keys()},
                    },
                },
            )

            # Net LW (ov/sub) using L_inc above overstory
            Ln_ov[i1], Ln_sub[i1] = rad.calc_L_n_Campbell(
                T_C=T_ov[i1],
                T_S=T_sub[i1],
                L_dn=L_dn_atm[i1],
                lai=LAI_eff_ov[i1],
                emisVeg=emis_leaf[i1],
                emisGrd=emis_soil[i1],
                x_LAD=x_LAD[i1],
            )

            Rn_ov[i1] = Sn_ov[i1] + Ln_ov[i1]
            Rn_sub[i1] = Sn_sub[i1] + Ln_sub[i1]

            # PT for ov
            LE_ov[i1] = myTSEB.Priestly_Taylor_LE_V(
                fv_g=1,
                Rn_V=Rn_ov[i1],
                alpha_PT=alpha_PT_i1[i1],
                Tair=Tair[i1],
                P_atm=P_atm[i1],
                c_p=c_p[i1],
            )

            H_ov[i1] = Rn_ov[i1] - LE_ov[i1]

            # Series canopy temperature (ov)
            T_ov[i1] = TSEB.calc_T_C_series(
                Tr_K=Trad[i1],
                T_A_K=Tair[i1],
                R_A=R_A_ov[i1],
                R_x=R_x_ov[i1],
                R_S=R_S_ov[i1],
                f_theta=fv_ov[i1],
                H_C=H_ov[i1],
                rho=rho[i1],
                c_p=c_p[i1]
            )

            # Radiometric partition to substrate
            T_sub[i1] = myTSEB.estimate_Trad_S(
                Trad=Trad[i1],
                Trad_V=T_ov[i1],
                f_theta=fv_ov[i1],
            )

            # Recompute LW with updated temperatures
            Ln_ov[i1], Ln_sub[i1] = rad.calc_L_n_Campbell(
                T_C=T_ov[i1],
                T_S=T_sub[i1],
                L_dn=L_dn_atm[i1],
                lai=LAI_eff_ov[i1],
                emisVeg=emis_leaf[i1],
                emisGrd=emis_soil[i1],
                x_LAD=x_LAD[i1],
            )

            Rn_ov[i1] = Sn_ov[i1] + Ln_ov[i1]
            Rn_sub[i1] = Sn_sub[i1] + Ln_sub[i1]

            # 2) Air temperature at canopy interface (T_AC), array form
            T_AC_ov[i1] = (
                    (Tair[i1] / R_A_ov[i1] + T_sub[i1] / R_S_ov[i1] + T_ov[i1] / R_x_ov[
                        i1]) /
                    (1.0 / R_A_ov[i1] + 1.0 / R_S_ov[i1] + 1.0 / R_x_ov[i1])
            )

            # Recompute R_S_sub with updated deltaT
            _, _, R_S_ov[i1] = TSEB.calc_resistances(
                resistance_form_ov,
                {
                    "R_S": {
                        "u_friction": u_friction[i1],
                        "h_C": h_ov[i1],
                        "d_0": d_0_ov[i1],
                        "z_0M": z_0m_ov[i1],
                        "L": L[i1],
                        "F": F_ov[i1],
                        "omega0": omega0_ov[i1],
                        "LAI": LAI_ov[i1],
                        "leaf_width": leaf_width_ov[i1],
                        "z0_soil": z0_soil[i1],
                        "z_u": z_u[i1],
                        "deltaT": T_sub[i1] - T_AC_ov[i1],
                        "u": u[i1],
                        "rho": rho[i1],
                        "c_p": c_p[i1],
                        "f_cover": fv_ov[i1],
                        "w_C": w_ov[i1] / h_ov[i1],
                        "massman_profile": massman_profile,
                        "res_params": {k: res_params_ov[k] for k in res_params_ov.keys()},
                    }
                },
            )


            # 3) Soil sensible heat flux H_S, array form
            H_sub[i1] = rho[i1] * c_p[i1] * (T_sub[i1] - T_AC_ov[i1]) / R_S_ov[i1]

            # Your G scheme for loop-1
            G[i1] = Rn_sub[i1] * (1.0 - fv_un[i1]) * 0.3 * np.exp(-0.5 * LAI_eff_un[i1])
            AELE_sub[i1] = Rn_sub[i1] - G[i1]
            LE_sub[i1] = AELE_sub[i1] - H_sub[i1]
            # LE_ov[i1] = Rn_ov[i1] - H_ov[i1]

            H[i1] = H_ov[i1] + H_sub[i1]
            LE[i1] = LE_ov[i1] + LE_sub[i1]

            # Reduce alpha only where LE_sub < 0
            i1 = (LE_sub < 0) & (alpha_PT_i1 > 0)
            alpha_PT_i1[i1] = np.maximum(alpha_PT_i1[i1] - 0.1, 0.0)

            it1 += 1

        return alpha_PT_i1

    # --- helper: loop-2 (series un/soil) with its own alpha reduction
    def loop_un_series(alpha_PT_i2, max_it2=20, mask=None):

        if mask is None:
            i2 = np.ones_like(LAI_un, dtype=bool)
        else:
            i2 = mask.copy()

        # initial guesses from current T_sub / T_ov
        T_un[i2] = 0.5 * (T_sub[i2] + Tair[i2])
        T_soil[i2] = myTSEB.estimate_Trad_S(Trad=T_sub[i2], Trad_V=T_un[i2], f_theta=fv_un[i2])

        L_dn_sub, _ = _update_Ldn_sub()

        # initialize LW partition once (will be updated inside loop too)
        Ln_un[i2], Ln_soil[i2] = rad.calc_L_n_Campbell(
            T_C=T_un[i2],
            T_S=T_soil[i2],
            L_dn=L_dn_sub[i2],
            lai=LAI_eff_un[i2],
            emisVeg=emis_leaf[i2],
            emisGrd=emis_soil[i2],
            x_LAD=x_LAD[i2],
        )

        # NOTE: start guess for interface air temperature (doesn't have to be perfect)
        T_AC_un[i2] = 0.5 * (Tair[i2] + T_ov[i2])

        it2 = 0

        while np.any(i2) and it2 < max_it2:
            # Resistances (series)
            R_A_un[i2], R_x_un[i2], R_S_un[i2] = TSEB.calc_resistances(
                resistance_form_un,
                {
                    "R_A": {
                        "z_T": z_T[i2],
                        "u_friction": u_friction[i2],
                        "L": L[i2],
                        "d_0": d_0_un[i2],
                        "z_0H": z_0H_un[i2],
                    },
                    "R_x": {
                        "u_friction": u_friction[i2],
                        "h_C": h_un[i2],
                        "d_0": d_0_un[i2],
                        "z_0M": z_0m_un[i2],
                        "L": L[i2],
                        "F": F_un[i2],
                        "LAI": LAI_un[i2],
                        "leaf_width": leaf_width_un[i2],
                        "z0_soil": z0_soil[i2],
                        "massman_profile": massman_profile,
                        "res_params": {k: res_params_ov[k] for k in res_params_ov.keys()},
                    },
                    "R_S": {
                        "u_friction": u_friction[i2],
                        "h_C": h_un[i2],
                        "d_0": d_0_un[i2],
                        "z_0M": z_0m_un[i2],
                        "L": L[i2],
                        "F": F_un[i2],
                        "omega0": omega0_un[i2],
                        "LAI": LAI_un[i2],
                        "leaf_width": leaf_width_un[i2],
                        "z0_soil": z0_soil[i2],
                        "z_u": z_u[i2],
                        "deltaT": T_soil[i2] - T_un[i2],
                        "u": u[i2],
                        "rho": rho[i2],
                        "c_p": c_p[i2],
                        "f_cover": fv_un[i2],
                        "w_C": np.full_like(LAI_un[i2], 1),
                        "massman_profile": massman_profile,
                        "res_params": {k: res_params_ov[k] for k in res_params_ov.keys()},
                    },
                },
            )

            # LW forcing below ov depends on current T_ov (which may change in outer loop)
            L_dn_sub, _ = _update_Ldn_sub()

            # Update LW split (un/soil) with current temps
            Ln_un[i2], Ln_soil[i2] = rad.calc_L_n_Campbell(
                T_C=T_un[i2],
                T_S=T_soil[i2],
                L_dn=L_dn_sub[i2],
                lai=LAI_eff_un[i2],
                emisVeg=emis_leaf[i2],
                emisGrd=emis_soil[i2],
                x_LAD=x_LAD[i2],
            )

            Rn_un[i2] = Sn_un[i2] + Ln_un[i2]
            Rn_soil[i2] = Sn_soil[i2] + Ln_soil[i2]

            # PT for understory
            LE_un[i2] = myTSEB.Priestly_Taylor_LE_V(
                fv_g=1,
                Rn_V=Rn_un[i2],
                alpha_PT=alpha_PT_i2[i2],
                Tair=Tair[i2],
                P_atm=P_atm[i2],
                c_p=c_p[i2],
            )
            H_un[i2] = Rn_un[i2] - LE_un[i2]

            # Understory canopy temperature (series)
            T_un[i2] = TSEB.calc_T_C_series(
                Tr_K=T_sub[i2],
                T_A_K=Tair[i2],
                R_A=R_A_un[i2],
                R_x=R_x_un[i2],
                R_S=R_S_un[i2],
                f_theta=fv_un[i2],
                H_C=H_un[i2],
                rho=rho[i2],
                c_p=c_p[i2],
            )

            # Soil radiometric temperature under understory
            T_soil[i2] = myTSEB.estimate_Trad_S(
                Trad=T_sub[i2],
                Trad_V=T_un[i2],
                f_theta=fv_un[i2],
            )

            # Recompute LW with updated temps
            Ln_un[i2], Ln_soil[i2] = rad.calc_L_n_Campbell(
                T_C=T_un[i2],
                T_S=T_soil[i2],
                L_dn=L_dn_sub[i2],
                lai=LAI_eff_un[i2],
                emisVeg=emis_leaf[i2],
                emisGrd=emis_soil[i2],
                x_LAD=x_LAD[i2],
            )
            Rn_un[i2] = Sn_un[i2] + Ln_un[i2]
            Rn_soil[i2] = Sn_soil[i2] + Ln_soil[i2]

            _, _, R_S_un[i2] = TSEB.calc_resistances(
                resistance_form_un, {
                    "R_S": {
                        "u_friction": u_friction[i2],
                        "h_C": h_un[i2],
                        "d_0": d_0_un[i2],
                        "z_0M": z_0m_un[i2],
                        "L": L[i2],
                        "F": F_un[i2],
                        "omega0": omega0_un[i2],
                        "LAI": LAI_un[i2],
                        "leaf_width": leaf_width_un[i2],
                        "z0_soil": z0_soil[i2],
                        "z_u": z_u[i2],
                        "deltaT": T_soil[i2] - T_un[i2],
                        "u": u[i2],
                        "rho": rho[i2],
                        "c_p": c_p[i2],
                        "f_cover": fv_un[i2],
                        "w_C": np.full_like(LAI_un[i2], 1),
                        "massman_profile": massman_profile,
                        "res_params": {k: res_params_ov[k] for k in res_params_ov.keys()},
                    },
                }
            )

            # Air temperature at interface for series network
            T_AC_un[i2] = (
                    (Tair[i2] / R_A_un[i2] + T_soil[i2] / R_S_un[i2] + T_un[i2] / R_x_un[i2]) /
                    (1.0 / R_A_un[i2] + 1.0 / R_S_un[i2] + 1.0 / R_x_un[i2])
            )

            # Soil sensible heat
            H_soil[i2] = rho[i2] * c_p[i2] * (T_soil[i2] - T_AC_un[i2]) / R_S_un[i2]

            # Soil heat flux
            G[i2] = 0.3 * Rn_soil[i2] * np.exp(-0.5 * LAI_eff_un[i2])

            # Soil LE residual
            LE_soil[i2] = Rn_soil[i2] - G[i2] - H_soil[i2]
            LE_un[i2] = np.maximum(Rn_un[i2] - H_un[i2], 0.0)
            # H_un[i2] = Rn_un[i2] - LE_un[i2]
            LE_sub[i2] = LE_un[i2] + LE_soil[i2]

            H[i2] = H_ov[i2] + H_soil[i2] + H_un[i2]
            LE[i2] = LE_ov[i2] + LE_soil[i2] + LE_un[i2]

            AELE_soil[i2] = Rn_soil[i2] - G[i2]  # (available energy for LE + H)

            # bad_rn_soil = mask & (Rn_soil < 0)
            # bad_rn_un = mask & (Rn_un < 0)
            # bad_tsoil = mask & (T_soil < 273.15)
            # bad_view = mask & ((1.0 - fv_un) < 0.05)
            #
            # bad_3seb = bad_rn_soil | bad_rn_un | bad_tsoil | bad_view

            # if np.any(bad_3seb):
            #     # constrain component net radiation
            #     Rn_soil[bad_3seb] = np.maximum(Rn_soil[bad_3seb], 0.0)
            #     Rn_un[bad_3seb] = np.maximum(Rn_un[bad_3seb], 0.0)
            #
            #     # keep radiative consistency if desired
            #     Ln_soil[bad_3seb] = Rn_soil[bad_3seb] - Sn_soil[bad_3seb]
            #     Ln_un[bad_3seb] = Rn_un[bad_3seb] - Sn_un[bad_3seb]
            #
            #     # constrain LE
            #     LE_soil[bad_3seb] = np.maximum(LE_soil[bad_3seb], 0.0)
            #     LE_un[bad_3seb] = np.maximum(LE_un[bad_3seb], 0.0)
            #
            #     # recompute H consistently
            #     AELE_soil[bad_3seb] = Rn_soil[bad_3seb] - G[bad_3seb]
            #     H_soil[bad_3seb] = AELE_soil[bad_3seb] - LE_soil[bad_3seb]
            #     H_un[bad_3seb] = Rn_un[bad_3seb] - LE_un[bad_3seb]
            #
            #     # recompute combined terms
            #     LE_sub[bad_3seb] = LE_un[bad_3seb] + LE_soil[bad_3seb]
            #     H[bad_3seb] = H_ov[bad_3seb] + H_un[bad_3seb] + H_soil[bad_3seb]
            #     LE[bad_3seb] = LE_ov[bad_3seb] + LE_un[bad_3seb] + LE_soil[bad_3seb]

            # Reduce alpha only where LE_soil < 0 and AE positive
            i2 = mask & (LE_soil < 0) & (AELE_soil > 0) & (alpha_PT_i2 > 0)
            alpha_PT_i2[i2] = np.maximum(alpha_PT_i2[i2] - 0.1, 0.0)

            it2 += 1

        return alpha_PT_i2

    sigma = 5.670374419e-8

    # ======================================================================================
    # OUTER COUPLING DRIVER
    # ======================================================================================
    alpha_PT_i1 = _as_array(alpha_PT_ov, 1.26)
    alpha_PT_i2 = _as_array(alpha_PT_un, 1.00) # understory PT (you can start at 1.26 too)

    converged = False

    for outer in range(max_outer):
        # loop-1 for all pixels
        if scheme_ov == 0:
            alpha_PT_i1 = loop_ov_parallel(alpha_PT_i1, max_it1=max_it1)
        elif scheme_ov == 1:
            alpha_PT_i1 = loop_ov_series(alpha_PT_i1, max_it1=max_it1)
        elif scheme_ov not in (0, 1):
            raise ValueError("scheme_ov must be 0 (parallel) or 1 (series)")

        # 2SEB branch
        if np.any(mask_2seb):
            LE_soil[mask_2seb] = LE_sub[mask_2seb]
            H_soil[mask_2seb] = H_sub[mask_2seb]
            Rn_soil[mask_2seb] = Rn_sub[mask_2seb]
            Ln_soil[mask_2seb] = Ln_sub[mask_2seb]
            T_soil[mask_2seb] = T_sub[mask_2seb]
            AELE_soil[mask_2seb] = Rn_soil[mask_2seb] - G[mask_2seb]

            LE_un[mask_2seb] = 0.0
            H_un[mask_2seb] = 0.0
            Rn_un[mask_2seb] = 0.0
            Ln_un[mask_2seb] = 0.0
            T_un[mask_2seb] = np.nan

        # 3SEB branch
        if np.any(mask_3seb):
            alpha_PT_i2 = loop_un_series(alpha_PT_i2, max_it2=max_it2, mask=mask_3seb)

        # feasibility
        bad_soil_3src = mask_3seb & (LE_soil < 0) & (AELE_soil > 0)
        bad_soil_2src = mask_2seb & (LE_soil < 0) & (AELE_soil > 0)
        bad_soil = bad_soil_3src | bad_soil_2src

        # totals
        H = H_ov + H_un + H_soil
        LE = LE_ov + LE_un + LE_soil
        con_LE = LE > 0

        # stability
        if const_L is None and np.any(con_LE):
            L_new = MO.calc_L(u_friction[con_LE], Tair[con_LE], rho[con_LE], c_p[con_LE], H[con_LE], LE[con_LE])
            ustar_new = MO.calc_u_star(u[con_LE], z_u[con_LE], L_new, d_0_ov[con_LE], z_0m_ov[con_LE])
            ustar_new = np.maximum(ustar_new, U_FRICTION_MIN[con_LE])

            # diff[con_LE] = np.nanmax(np.abs((L_new[con_LE] - L[con_LE]) / (L[con_LE] + 1e-6)))
            L[con_LE] = L_new
            u_friction[con_LE] = ustar_new

        if not np.any(bad_soil):
            converged = True
            break
        else:
            can_reduce_un = bad_soil_3src & (alpha_PT_i2 > 0)
            if np.any(can_reduce_un):
                alpha_PT_i2[can_reduce_un] = np.maximum(alpha_PT_i2[can_reduce_un] - step_alpha, 0.0)
                continue

            can_reduce_ov = bad_soil & (alpha_PT_i1 > 0)
            if np.any(can_reduce_ov):
                alpha_PT_i1[can_reduce_ov] = np.maximum(alpha_PT_i1[can_reduce_ov] - step_alpha, 0.0)
                continue
            else:
                converged = False


    # ---------------------------------------------------------------------
    # Fallback: ONLY if still infeasible
    # ---------------------------------------------------------------------
    # Testing radiative invalidity
    invalid_rn = (
            (Rn_soil < 0) |
            (Rn_un < 0) |
            (Rn_ov < 0)
    )

    # Testing temperature invalidity
    invalid_temp = (
            (np.abs(T_soil - Tair) > 25) |
            (np.abs(T_un - Tair) > 25) |
            (np.abs(T_ov - Tair) > 25)
    )

    #Not coverged

    not_coverged = LE_soil < 0
    invalid_partition = invalid_rn | invalid_temp | not_coverged

    if np.any(invalid_partition):
        Rn_ov[invalid_partition] = np.maximum(Rn_ov[invalid_partition], 0)
        Rn_un[invalid_partition] = np.maximum(Rn_un[invalid_partition], 0)
        Rn_soil[invalid_partition] = np.maximum(Rn_soil[invalid_partition], 0)
        Rn_sub[invalid_partition] = Rn_un[invalid_partition] + Rn_soil[invalid_partition]

        Ln_ov[invalid_partition] = Rn_ov[invalid_partition] - Sn_ov[invalid_partition]
        Ln_un[invalid_partition] = Rn_un[invalid_partition] - Sn_un[invalid_partition]
        Ln_soil[invalid_partition] = Rn_soil[invalid_partition] - Sn_soil[invalid_partition]
        Ln_sub[invalid_partition] = Ln_un[invalid_partition] + Ln_soil[invalid_partition]

        LE_ov[invalid_partition] = 0
        LE_un[invalid_partition] = 0
        LE_soil[invalid_partition] = 0
        LE_sub[invalid_partition] = 0
        LE[invalid_partition] = 0

        G[invalid_partition] = np.minimum(G[invalid_partition], Rn_soil[invalid_partition])

        H_ov[invalid_partition] = Rn_ov[invalid_partition]
        H_un[invalid_partition] = Rn_un[invalid_partition]

        H_soil[invalid_partition] = np.maximum(Rn_soil[invalid_partition] - G[invalid_partition], 0)
        H_sub[invalid_partition] = H_un[invalid_partition] + H_soil[invalid_partition]
        LE_sub[invalid_partition] = LE_un[invalid_partition] + LE_soil[invalid_partition]
        H[invalid_partition] = H_ov[invalid_partition] + H_un[invalid_partition] + H_soil[invalid_partition]
        LE[invalid_partition] = LE_ov[invalid_partition] + LE_un[invalid_partition] + LE_soil[invalid_partition]


    L_dn_sub, _ = _update_Ldn_sub()
    return {
        "T_ov": T_ov,
        "T_un": T_un,
        "T_soil": T_soil,
        "T_sub": T_sub,
        "T_AC_ov": T_AC_ov,
        "T_AC_un": T_AC_un,
        "R_S_ov": R_S_ov,
        "R_S_un": R_S_un,
        "LE_ov": LE_ov,
        "LE_un": LE_un,
        "LE_soil": LE_soil,
        "LE_sub": LE_sub,
        "LE": LE,
        "H_ov": H_ov,
        "H_un": H_un,
        "H_soil": H_soil,
        "H_sub": H_sub,
        "H": H,
        "Rn_ov": Rn_ov,
        "Rn_un": Rn_un,
        "Rn_soil": Rn_soil,
        "Rn_sub": Rn_sub,
        "Sn_ov": Sn_ov,
        "Sn_un": Sn_un,
        "Sn_soil": Sn_soil,
        "Sn_sub": Sn_sub,
        "L_dn_sub": L_dn_sub,
        "Ln_ov": Ln_ov,
        "Ln_un": Ln_un,
        "Ln_soil": Ln_soil,
        "Ln_sub": Ln_sub,
        "G": G,
        "L": L,
        "u_friction": u_friction,
        "alpha_PT_ov_final": alpha_PT_i1,
        "alpha_PT_un_final": alpha_PT_i2,
        "mask_2seb": mask_2seb,
        "mask_3seb": mask_3seb,
        "converged": converged,
    }


def enforce_nonnegative_soil_evap(LE_soil, H_soil, AELE_soil):
    """
    Enforce LE_soil >= 0 by clipping and adjusting H_soil to conserve:
        AE_soil = H_soil + LE_soil   (AE_soil = Rn_soil - G = AELE_soil)
    """
    violated = (LE_soil < 0) & (AELE_soil > 0)
    if np.any(violated):
        LE_soil = LE_soil.copy()
        H_soil  = H_soil.copy()
        LE_soil[violated] = 0.0
        H_soil[violated]  = AELE_soil[violated]
    return LE_soil, H_soil, violated


