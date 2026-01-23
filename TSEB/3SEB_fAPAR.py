import numpy as np
import pyTSEB.TSEB as TSEB
from pyTSEB import net_radiation as rad
import functions as myTSEB
from SALib.sample import saltelli
from SALib.analyze import sobol
import pandas as pd


def estimate_lai_2_fv(LAI, fv_var, x_LAD):
    K_be = myTSEB.estimate_Kbe(x_LAD, 0)
    fv02 = np.clip((1 - np.exp(-K_be * LAI)) * fv_var, 1e-6, 1)
    return fv02

def fAPAR_3SEB(LAI_ov, fv_var_ov, h_V_ov, x_LAD_ov,
               LAI_un, fv_var_un, h_V_un, x_LAD_un,
               row_sep, row_azimuth,
               Sdn, sza_degrees, saa_degrees, P_atm,
               rho_vis_leaf_ov, rho_nir_leaf_ov, tau_vis_leaf_ov, tau_nir_leaf_ov,
               rho_vis_leaf_un, rho_nir_leaf_un, tau_vis_leaf_un, tau_nir_leaf_un,
               rho_vis_soil, rho_nir_soil,
               T_MODEL='CN_R'):


    sza_rad =  np.radians(sza_degrees)
    saa_rad = np.radians(saa_degrees)
    row_azimuth_rad = np.radians(row_azimuth)
    # vza_rad = np.radians(vza_degrees)
    # f_un = (1 - fv_ov)
    # fv_un = f_un * fv_un
    K_be_ov = myTSEB.estimate_Kbe(x_LAD_ov, 0)
    fv_ov = np.clip((1 - np.exp(-K_be_ov * LAI_ov)) * fv_var_ov, 1e-6, 1)
    w_V_ov = fv_ov * row_sep

    K_be_un = myTSEB.estimate_Kbe(x_LAD_ov, 0)
    fv_un = np.clip((1 - np.exp(-K_be_un * LAI_un)) * fv_var_un, 1e-6, 1)
    w_V_un = fv_un * row_sep

    # fs_un = 1 - (fv_un + fv_ov)
    # fs_un = np.clip(fs_un, 1e-6, 1)

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
                                                                     rho_leaf=np.array((rho_vis_leaf_un, rho_nir_leaf_un)),
                                                                     tau_leaf=np.array((tau_vis_leaf_un, tau_nir_leaf_un)),
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
    # print(Sn_ov)
    return Sn_ov, Sn_un, Sn_S

# Sn_ov, Sn_un, Sn_S = fAPAR_3SEB(LAI_ov=3, fv_ov=0.4, h_V_ov=3, x_LAD_ov=1,
#            LAI_un=2, fv_un=0.4, h_V_un=0.1, x_LAD_un=0.5,
#            row_sep=7, row_azimuth=0,
#            Sdn=1000, sza_degrees=10, saa_degrees=10, P_atm=1013,
#            rho_vis_leaf=0.07, rho_nir_leaf=0.32, tau_vis_leaf=0.08, tau_nir_leaf=0.33, rho_vis_soil=0.15, rho_nir_soil=0.25,
#            T_MODEL='CN_H')

# Independent base inputs only
names = [
    "LAI_ov", "fv_var_ov", "h_V_ov", "x_LAD_ov",
    "LAI_un", "fv_var_un", "h_V_un", "x_LAD_un",
    "row_sep", "row_azimuth",
    "Sdn", "sza_degrees", "saa_degrees", "P_atm",
    "rho_vis_leaf_ov", "rho_nir_leaf_ov",
    "tau_vis_leaf_ov", "tau_nir_leaf_ov",
    "rho_vis_leaf_un", "rho_nir_leaf_un",
    "tau_vis_leaf_un", "tau_nir_leaf_un",
    "rho_vis_soil", "rho_nir_soil"
]


problem = {
    "num_vars": len(names),
    "names": names,
    "bounds": [
        [0.1, 8.0],     # LAI_ov (m2/m2)
        [0.7, 1.1],   # fv_ov (fraction)
        [1, 8.0],     # h_V_ov (m)
        [0.5, 1.5],     # x_LAD_ov (unitless)

        [0, 7.0],    # LAI_un (m2/m2)
        [0.7, 1.1],   # fv_un (fraction)
        [0.02, 1.0],    # h_V_un (m)
        [0.5, 1.5],     # x_LAD_un (unitless)

        [2.0, 12.0],    # row_sep (m)
        [0.0, 179.0],   # row_azimuth (deg) (0-180 enough due to symmetry)

        [700, 1100.0],# Sdn (W m-2)
        [10, 70],    # sza_degrees (deg)
        [0.0, 360.0],   # saa_degrees (deg)
        [850.0, 1050.0],# P_atm (hPa or mb) (same numeric)

        [0.03, 0.18],   # rho_vis_leaf_ov
        [0.25, 0.60],   # rho_nir_leaf_ov
        [0.01, 0.12],   # tau_vis_leaf_ov
        [0.15, 0.55],   # tau_nir_leaf_ov

        [0.03, 0.18],   # rho_vis_leaf_un
        [0.25, 0.60],   # rho_nir_leaf_un
        [0.01, 0.12],   # tau_vis_leaf_un
        [0.15, 0.55],   # tau_nir_leaf_un
        [0.05, 0.35],   # rho_vis_soil
        [0.10, 0.55],   # rho_nir_soil


    ]
}

# --- 1) Generate samples once
N = 5000
second_order = False
X = saltelli.sample(problem, N, calc_second_order=second_order)

# --- 2) Evaluate model once, capture all outputs
Y_ov = np.empty(X.shape[0], dtype=float)
Y_un = np.empty(X.shape[0], dtype=float)
Y_S  = np.empty(X.shape[0], dtype=float)

T_MODEL = "CN_R"

for i, row in enumerate(X):
    params = dict(zip(problem["names"], row))
    Sn_ov, Sn_un, Sn_S = fAPAR_3SEB(**params, T_MODEL=T_MODEL)
    Y_ov[i] = Sn_ov
    Y_un[i] = Sn_un
    Y_S[i]  = Sn_S

# Optional safety: remove invalid runs (NaN/inf) consistently for each output
def clean_xy(X, Y):
    mask = np.isfinite(Y)
    return X[mask], Y[mask]

X_ov, Y_ov_c = clean_xy(X, Y_ov)
X_un, Y_un_c = clean_xy(X, Y_un)
X_S,  Y_S_c  = clean_xy(X, Y_S)

pd_X_ov = pd.DataFrame(X_ov)
pd_X_ov.columns = names
pd_X_ov.loc[:, 'Sn_ov'] = Y_ov
pd_X_ov.loc[:, 'Sn_un'] = Y_un
pd_X_ov.loc[:, 'Sn_s'] = Y_S

path = r'files/3seb_fapar_{}.csv'.format(T_MODEL)
pd_X_ov.to_csv(path, index=False)
# --- 3) Sobol analyze for each output
# If you filtered, X is no longer Saltelli-structured -> results become invalid.
# So prefer analyzing without filtering unless you know what you're doing.
# Therefore: analyze the original Y vectors if possible.
Si_ov = sobol.analyze(problem, Y_ov, calc_second_order=second_order, print_to_console=False)
Si_un = sobol.analyze(problem, Y_un, calc_second_order=second_order, print_to_console=False)
Si_S  = sobol.analyze(problem, Y_S,  calc_second_order=second_order, print_to_console=False)

path = r'files/sensitivity_analysis_3SEB_fAPAR_{}_5000.xlsx'.format(T_MODEL)

def si_to_table(data, problem):
    df = pd.DataFrame(data)
    df.insert(0, 'params', problem['names'])
    return df
si_tables = [si_to_table(df, problem) for df in [Si_ov, Si_un, Si_S]]

names_outcomes = ['Si_ov', 'Si_un', 'Si_S']
# [si_tables[x].to_excel(path, sheet_name=names[x]) for x in range(0, len(si_tables))]
with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
    si_tables[0].to_excel(writer, sheet_name=names_outcomes[0], index=False)
    si_tables[1].to_excel(writer, sheet_name=names_outcomes[1], index=False)
    si_tables[2].to_excel(writer, sheet_name=names_outcomes[2], index=False)


for n, s1, st, c1, ct in zip(problem["names"], Si_ov["S1"], Si_ov["ST"], Si_ov["S1_conf"], Si_ov["ST_conf"]):
    print(f"{n:>12s} | S1={s1:.3f} (±{c1:.3f})  ST={st:.3f} (±{ct:.3f})")
