import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BinomialModelPrism_DirectandDiffuseRadiation import compute_binomial_prism
import pyTSEB.TSEB as TSEB
import GeneralFunctions as GF
from SALib.sample import saltelli
from SALib.analyze import sobol

# What is Gtheta (fraction of leaf area projected in the direction of the sun)? ESTIMATE, EQUATION 3
# What is nrays and Nbins?
# Which kind of shapes are available

# Determine how soil absorptivity change considering cover crop

def binomial_model(lai, fv_var, h_V, row_sep, plant_sep_var,
                   sza_radians, saa_radians, row_azimuth_radians,
                   Sdn, P_atm,
                   abs_vis_leaf, abs_nir_leaf, rho_vis_soil, rho_nir_soil,
                   Nz_diff=16, Nphi_diff=32, Nbins=40, nrays=5000, shape="prism"):

    sza_degrees = np.degrees(sza_radians)
    difvis, difnir, fvis, fnir = TSEB.rad.calc_difuse_ratio(
        S_dn=Sdn,
        sza=sza_degrees,
        press=P_atm
    )

    skyl = fvis * difvis + fnir * difnir
    Srad_dir = (1. - skyl) * Sdn
    Srad_diff = skyl * Sdn

    phi_radians = saa_radians - row_azimuth_radians

    # saltelli.sample create random values that sometime don't make any sense
    # LAI, FV, wc, sp, sr are intrinsically related, so they can not be analysed independently
    # For instance FV = 0 with LAI = 8. wc = 8, sp and sr = 5
    # They'll be estimated mainly based  on the LAI
    K_be = GF.estimate_Kbe(x_LAD=1, sza=0)
    fv0 = np.clip((1 - np.exp(-K_be * lai)) * fv_var, 1e-6, 1)
    w_V = fv0 * row_sep
    plant_sep = row_sep * plant_sep_var # sp [0.5, 1]

    Gtheta = GF.canopy_gap_fraction(lai, fv0=fv0, w_V=w_V, h_V=h_V, sza=sza_radians, saa=saa_radians,
                                    row_azimuth=row_azimuth_radians, hb_V=0, L=None, x_LAD=1)
    # canopy_width < sr and sp

    try:
        Rc_binomial, Rs_binomial = compute_binomial_prism(
            sr=row_sep, #Row spacing (meters)
            sza=sza_radians, # sun zenith angle (radians)
            psi=phi_radians, # sun azimuth relative to row orientation (radians)
            lai=lai, # Leaf Area Index
            ameanv=abs_vis_leaf, # Leaf absorptivity in the visible (PAR) band
            ameann=abs_nir_leaf, # Leaf absorptivity in the near infra-red band (NIR) band
            rsoilv=rho_vis_soil, # Soil absorptivity in the visible (PAR) band
            rsoiln=rho_nir_soil,  # Soil absorptivity in the near infra-red band (NIR) band
            Srad_dir=Srad_dir,  # Direct-beam incoming radiation (W m-2)
            Srad_diff=Srad_diff,  # Diffuse incoming radiation (W m-2)
            fvis=fvis, # Fraction incoming radiation in the visible part of the spectrum
            fnir=fnir,  # Fraction incoming radiation in the near infra-red part of the spectrum
            CanopyHeight=h_V,# Canopy heigth (meters)
            wc=w_V, # Canopy width (meters)
            sp=plant_sep, #Plant spacing (meters)
            Gtheta=Gtheta, # fraction of leaf area projected in the direction of the sun
            nrays=nrays, #
            Nbins=Nbins,
            shape=shape,   # <-- shape of the canopy
            Nz_diff=Nz_diff,
            Nphi_diff=Nphi_diff,  # sampling for diffuse hemisphere
        )
    except:
        Rc_binomial, Rs_binomial = -9999, -9999

    return Rc_binomial, Rs_binomial

# binomial_model(lai=5.3796875, fv_var=0.7328125, h_V=0.736328125, row_sep=4.6484375, plant_sep_var=0.576171875,
#                sza_radians=0.657566431, saa_radians=2.850477189, row_azimuth_radians=3.111935692,
#                Sdn=956.640625, P_atm=991.328125,
#                abs_vis_leaf=0.848828125, abs_nir_leaf=0.123242188, rho_vis_soil=0.798242188, rho_nir_soil=0.683007813,
#                Nz_diff=16, Nphi_diff=32, Nbins=40, nrays=5000, shape="prism")


names = ['lai', 'fv_var', 'h_V', 'row_sep', 'plant_sep_var', 'sza_radians', 'saa_radians', 'row_azimuth_radians',
         'Sdn', 'P_atm', 'abs_vis_leaf', 'abs_nir_leaf', 'rho_vis_soil', 'rho_nir_soil']

# Independent base inputs only
problem = {
    "num_vars": 14,
    "names": names,
    "bounds": [
        [0.2, 7.0],  # lai: orchards/crops
        [0.7, 1.1],  # fv_var
        [0.5, 6.0],  # CanopyHeight
        # [0.5, 5.0],  # wc
        [2.0, 8.0],  # row_sep: typical orchard rows
        [0.5, 1],  # plant_sep_var: plant space in relation to sr
        [np.deg2rad(10), np.deg2rad(75)],  # sza: avoid near 0 and near 90
        [np.deg2rad(10), np.deg2rad(260)],  # saa: avoid near 0 and near 360
        [0, np.deg2rad(179)],               # row_azimuth: relative azimuth 0..pi

        [700, 1000],  # 9. Sdn (w m-2)
        [990, 1010],  # 11. P_atm (mb)

        [0.80, 0.9],               # abs_vis_leaf: VIS absorpti
        # vity high
        [0.05, 0.20],               # abs_nir_leaf: NIR absorptivity lower
        [0.60, 0.95],               # rho_vis_soil: VIS absorptivity (soil darker -> higher abs)
        [0.40, 0.85],               # rho_nir_soil: NIR absorptivity

        # [8, 24],  # Nz_diff (treat as integer later)
        # [16, 64],  # Nphi_diff (treat as integer later)
        # [10, 80],  # Nbins (treat as integer later)
    ],
}


# --- 1) Generate samples once
N = 1000
second_order = False
X = saltelli.sample(problem, N, calc_second_order=second_order)

# --- 2) Evaluate model once, capture all outputs
Y_Rc = np.empty(X.shape[0], dtype=float)
Y_Rs = np.empty(X.shape[0], dtype=float)

for i, row in enumerate(X):
    params = dict(zip(problem["names"], row))
    Rc, Rs = binomial_model(**params)
    Y_Rc[i] = Rc
    Y_Rs[i] = Rs

pd_X = pd.DataFrame(X)
pd_X.columns = names
pd_X.loc[:, 'Sn_ov'] = Y_Rc
pd_X.loc[:, 'Sn_un'] = Y_Rs

# K_be = GF.estimate_Kbe(x_LAD=1, sza=0)
pd_X.loc[:, 'sza'] = np.rad2deg(pd_X.sza_radians)
pd_X.loc[:, 'saa'] = np.rad2deg(pd_X.saa_radians)
pd_X.loc[:, 'row_azimuth'] = np.rad2deg(pd_X.row_azimuth_radians)

pd_X.loc[:, 'fv0'] = np.clip((1 - np.exp(-0.5 * pd_X.lai)) * pd_X.fv_var, 1e-6, 1)
pd_X.loc[:, 'row_sep'] = pd_X.row_sep * pd_X.plant_sep_var

pd_X.to_csv('files/binomial_outputs.csv', index=False)

Si_Rc = sobol.analyze(problem, Y_Rc, calc_second_order=second_order, print_to_console=False)
Si_Rs = sobol.analyze(problem, Y_Rs, calc_second_order=second_order, print_to_console=False)

for n, s1, st, c1, ct in zip(problem["names"], Si_Rc["S1"], Si_Rc["ST"], Si_Rc["S1_conf"], Si_Rc["ST_conf"]):
    print(f"{n:>12s} | S1={s1:.3f} (±{c1:.3f})  ST={st:.3f} (±{ct:.3f})")
