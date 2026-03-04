import numpy as np
import pandas as pd
from BinomialModelPrism_DirectandDiffuseRadiation import compute_binomial_prism_manuel
import pyTSEB.TSEB as TSEB
import GeneralFunctions as GF
from SALib.sample import saltelli
from SALib.analyze import sobol
import os

def binomial_model(lai, fv_var, h_V, row_sep, plant_sep_var,
                   sza_radians, phi_radians,
                   abs_vis_leaf, abs_nir_leaf, rho_vis_soil, rho_nir_soil,
                   Nz_diff=16, Nphi_diff=32, Nbins=40, nrays=5000, shape="prism"):


    # abs_vis_leaf = 1 - rho_vis_leaf - tau_vis_leaf
    # abs_nir_leaf = 1 - rho_nir_leaf - tau_nir_leaf

    # vza_rad = np.radians(vza_degrees)
    # Campbell and Norman 1998. Page 172.
    m = 101.3 / (101.3 * np.cos(sza_radians))
    tau_atms = 0.6  # atmospheric transmittance
    Sp0 = 1368  # Extraterrestrial flux density
    Sp = 1368 * (tau_atms ** m)
    Sb = Sp * np.cos(sza_radians)

    Sd = 0.3 * (1 - tau_atms ** m) * Sp0 * np.cos(sza_radians)
    # f_diff = np.clip(0.15 + 0.6 * (1 - mu), 0.15, 0.7)
    Sdn = Sb + Sd

    sza_degrees = np.degrees(sza_radians)
    difvis, difnir, fvis, fnir = TSEB.rad.calc_difuse_ratio(
        S_dn=Sdn,
        sza=sza_degrees,
        press=np.full_like(lai, 1013)
    )

    skyl = fvis * difvis + fnir * difnir
    Srad_dir = (1. - skyl) * Sdn
    Srad_diff = skyl * Sdn

    # phi_radians = saa_radians - row_azimuth_radians

    # saltelli.sample create random values that sometime don't make any sense
    # LAI, FV, wc, sp, sr are intrinsically related, so they can not be analysed independently
    # For instance FV = 0 with LAI = 8. wc = 8, sp and sr = 5
    # They'll be estimated mainly based  on the LAI
    K_be = GF.estimate_Kbe(x_LAD=1, sza=0)
    fv0 = np.clip((1 - np.exp(-K_be * lai)) * fv_var, 1e-6, 1)
    w_V = fv0 * row_sep
    plant_sep = row_sep * plant_sep_var # sp [0.5, 1]

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
        Gtheta=0.5, # fraction of leaf area projected in the direction of the sun
        nrays=nrays, #
        Nbins=Nbins,
        shape=shape,   # <-- shape of the canopy
        Nz_diff=Nz_diff,
        Nphi_diff=Nphi_diff,  # sampling for diffuse hemisphere
    )


    return Sdn, Srad_dir, Srad_diff, Rc_binomial, Rs_binomial

names = ['lai', 'fv_var', 'h_V', 'row_sep', 'plant_sep_var', 'sza_radians', 'phi_radians',
         'abs_vis_leaf', 'abs_nir_leaf', 'rho_vis_soil', 'rho_nir_soil']

problem = {
    "num_vars": 11,
    "names": names,
    "bounds": [
        [0.2, 7.0],  # lai: orchards/crops
        [0.5, 1],  # fv_var
        [0.5, 6.0],  # CanopyHeight
        # [0.5, 5.0],  # wc
        [2.0, 8.0],  # row_sep: typical orchard rows
        [0.5, 1],  # plant_sep_var: plant space in relation to sr
        [np.deg2rad(10), np.deg2rad(75)],  # sza: avoid near 0 and near 90
        # [np.deg2rad(10), np.deg2rad(260)],  # saa: avoid near 0 and near 360
        [0, np.deg2rad(179)], # phi_degrees: relative azimuth 0..pi
        # [700, 1000],  # 9. Sdn (w m-2)
        # [990, 1010],  # 11. P_atm (mb)
        [0.80, 0.90],  # 18. abs_vis_leaf
        [0.05, 0.15],  # 19. abs_nir_leaf
        [0.05, 0.30],  # 22. rho_vis_soil
        [0.20, 0.45],  # 23. rho_nir_soil
    ],
}

N = 100
second_order = False
X = saltelli.sample(problem, N, calc_second_order=second_order)

# abs = 1 - (data.rho_nir_leaf + data.tau_nir_leaf)
# np.where(abs>1)

inputs_dict = {f"{names[i]}": X[:, i:i+1] for i in range(X.shape[1])}

# Testing with 1d array

Sdn, Srad_dir, Srad_diff, Rc_binomial, Rs_binomial = binomial_model(
    lai=inputs_dict['lai'], fv_var=inputs_dict['fv_var'], h_V=inputs_dict['h_V'], row_sep=inputs_dict['row_sep'],
    plant_sep_var=inputs_dict['plant_sep_var'],
    sza_radians=inputs_dict['sza_radians'], phi_radians=inputs_dict['phi_radians'],
    abs_vis_leaf=inputs_dict['abs_vis_leaf'], abs_nir_leaf=inputs_dict['abs_nir_leaf'],
    rho_vis_soil=inputs_dict['rho_vis_soil'], rho_nir_soil=inputs_dict['rho_nir_soil'],
    Nz_diff=16, Nphi_diff=32, Nbins=40, nrays=5000, shape="prism"
)

data = pd.DataFrame(X)
data.columns = names
data.loc[:,'Sdn'] = Sdn
data.loc[:,'Srad_dir'] = Srad_dir
data.loc[:,'Srad_diff'] = Srad_diff
data.loc[:,'Rc_binomial'] = Rc_binomial
data.loc[:,'Rs_binomial'] = Rs_binomial

data.to_csv("files/binomial.csv")

Si_Rc = sobol.analyze(problem, Rc_binomial[:, 0], calc_second_order=second_order, print_to_console=False)
Si_Rs = sobol.analyze(problem, Rs_binomial[:, 0], calc_second_order=second_order, print_to_console=False)

df_Si_Rc = pd.DataFrame(Si_Rc)
df_Si_Rc.insert(0, 'params', problem['names'])

df_Si_Rs = pd.DataFrame(Si_Rs)
df_Si_Rs.insert(0, 'params', problem['names'])

path = 'files/binomial_sensitivity_analysis.xlsx'
with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
    df_Si_Rc.to_excel(writer, sheet_name='Rc_binomial', index=False)
    df_Si_Rs.to_excel(writer, sheet_name='Rs_binomial', index=False)

for n, s1, st, c1, ct in zip(problem["names"], Si_Rc["S1"], Si_Rc["ST"], Si_Rc["S1_conf"], Si_Rc["ST_conf"]):
    print(f"{n:>12s} | S1={s1:.3f} (±{c1:.3f})  ST={st:.3f} (±{ct:.3f})")
