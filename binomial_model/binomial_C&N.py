import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BinomialModelPrism_DirectandDiffuseRadiation import compute_binomial_prism_manuel
import pyTSEB.TSEB as TSEB
import GeneralFunctions as GF
from SALib.sample import saltelli
from SALib.analyze import sobol


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
        Rc_binomial, Rs_binomial = compute_binomial_prism_manuel(
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

names = ['lai', 'fv_var', 'h_V', 'row_sep', 'plant_sep_var', 'sza_radians', 'saa_radians', 'row_azimuth_radians',
         'Sdn', 'P_atm', 'abs_vis_leaf', 'abs_nir_leaf', 'rho_vis_soil', 'rho_nir_soil']

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

N = 10
second_order = False
X = saltelli.sample(problem, N, calc_second_order=second_order)

binomial_model(lai=X[:, 0], fv_var=X[:, 1], h_V=X[:, 2], row_sep=X[:, 3], plant_sep_var=X[:, 4],
                   sza_radians=X[:, 5], saa_radians=X[:, 6], row_azimuth_radians=X[:, 7],
                   Sdn=X[:, 8], P_atm=X[:, 9],
                   abs_vis_leaf=X[:, 10], abs_nir_leaf=X[:, 11], rho_vis_soil=X[:, 12], rho_nir_soil=X[:, 13],
                   Nz_diff=16, Nphi_diff=32, Nbins=40, nrays=5000, shape="prism")
# binomial_model(lai, fv_var, h_V, row_sep, plant_sep_var,
#                    sza_radians, saa_radians, row_azimuth_radians,
#                    Sdn, P_atm,
#                    abs_vis_leaf, abs_nir_leaf, rho_vis_soil, rho_nir_soil,
#                    Nz_diff=16, Nphi_diff=32, Nbins=40, nrays=5000, shape="prism")
