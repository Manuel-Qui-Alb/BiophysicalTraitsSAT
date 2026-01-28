import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BinomialModelPrism_DirectandDiffuseRadiation import compute_binomial_prism
import pyTSEB.TSEB as TSEB


names =  ["sr", "sza", "psi", "lai", "ameanv", "ameann", "rsoilv", "rsoiln", "Srad_dir", "Srad_diff", "fvis",
          "CanopyHeight", "wc", "sp", "Gtheta", "Nz_diff", "Nphi_diff",  "Nbins"]
# Independent base inputs only
problem = {
    "num_vars": 18,
    "names": names,
    "bounds": [
        [2.0, 8.0],                 # sr: typical orchard rows
        [np.deg2rad(10), np.deg2rad(75)],  # sza: avoid near 0 and near 90
        [0.0, np.pi],               # psi: relative azimuth 0..pi
        [0.2, 7.0],                 # lai: orchards/crops

        [0.85, 0.98],               # ameanv: VIS absorptivity high
        [0.10, 0.40],               # ameann: NIR absorptivity lower
        [0.60, 0.95],               # rsoilv: VIS absorptivity (soil darker -> higher abs)
        [0.40, 0.85],               # rsoiln: NIR absorptivity

        [0.0, 1000.0],              # Srad_dir
        [0.0, 400.0],               # Srad_diff

        [0.45, 0.55],               # fvis (often ~0.5; keep narrow unless you *want* variability)

        [0.5, 6.0],                 # CanopyHeight
        [0.5, 5.0],                 # wc
        [1.0, 8.0],                 # sp

        [0.2, 1.0],                 # Gtheta (0-1, avoid 0)
        [8, 24],                    # Nz_diff (treat as integer later)
        [16, 64],                   # Nphi_diff (treat as integer later)
        [10, 80],                   # Nbins (treat as integer later)
    ],
}

# What is Gtheta (fraction of leaf area projected in the direction of the sun)? ESTIMATE, EQUATION 3
# What is nrays and Nbins?
# Which kind of shapes are available

# Determine how soil absorptivity change considering cover crop

def binomial_model(lai, Gtheta, CanopyHeight, wc, sr, sp, sza_radians, saa_radians, row_azimuth_radians,
                   Sdn, P_atm,
                   ameanv, ameann, rsoilv, rsoiln,
                   Nz_diff=16, Nphi_diff=32, Nbins=40,  nrays=5000, shape="prism"):

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
    compute_binomial_prism(
        sr=sr, #Row spacing (meters)
        sza=sza_radians, # sun zenith angle (radians)
        psi=phi_radians, # sun azimuth relative to row orientation (radians)
        lai=lai, # Leaf Area Index
        ameanv=ameanv, # Leaf absorptivity in the visible (PAR) band
        ameann=ameann, # Leaf absorptivity in the near infra-red band (NIR) band
        rsoilv=rsoilv, # Soil absorptivity in the visible (PAR) band
        rsoiln=rsoiln,  # Soil absorptivity in the near infra-red band (NIR) band
        Srad_dir=Srad_dir,  # Direct-beam incoming radiation (W m-2)
        Srad_diff=Srad_diff,  # Diffuse incoming radiation (W m-2)
        fvis=fvis, # Fraction incoming radiation in the visible part of the spectrum
        fnir=fnir,  # Fraction incoming radiation in the near infra-red part of the spectrum
        CanopyHeight=CanopyHeight,# Canopy heigth (meters)
        wc=wc, # Canopy width (meters)
        sp=sp, #Plant spacing (meters)
        Gtheta=Gtheta, # fraction of leaf area projected in the direction of the sun
        nrays=nrays, #
        Nbins=Nbins,
        shape=shape,   # <-- shape of the canopy
        Nz_diff=Nz_diff,
        Nphi_diff=Nphi_diff,  # sampling for diffuse hemisphere
    )
    return None