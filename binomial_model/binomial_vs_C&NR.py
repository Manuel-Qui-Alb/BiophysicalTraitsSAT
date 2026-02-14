import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BinomialModelPrism_DirectandDiffuseRadiation import compute_binomial_prism_manuel
import pyTSEB.TSEB as TSEB
import GeneralFunctions as GF
from SALib.sample import saltelli
from SALib.analyze import sobol
import TSEB.functions as myTSEB
import time


def contrasting_binomial_CN(lai, fv_var, h_V, row_sep, plant_sep_var,
                            sza_radians, phi_radians,
                            rho_vis_leaf, tau_vis_leaf, rho_nir_leaf, tau_nir_leaf, rho_vis_soil, rho_nir_soil,
                            Nz_diff=16, Nphi_diff=32, Nbins=40, nrays=5000, shape="prism"):

    sza_degrees = np.degrees(sza_radians)

    # Campbell and Norman 1998. Page 172.
    m = 101.3 / (101.3 * np.cos(sza_radians))
    tau_atms = 0.6  # atmospheric transmittance
    Sp0 = 1368  # Extraterrestrial flux density
    Sp = 1368 * (tau_atms ** m)
    Sb = Sp * np.cos(sza_radians)

    Sd = 0.3 * (1 - tau_atms ** m) * Sp0 * np.cos(sza_radians)
    # f_diff = np.clip(0.15 + 0.6 * (1 - mu), 0.15, 0.7)
    St = Sb + Sd

    difvis, difnir, fvis, fnir = TSEB.rad.calc_difuse_ratio(
        S_dn=St,
        sza=sza_degrees,
        press=np.full_like(St, 1013)
    )

    skyl = fvis * difvis + fnir * difnir
    Srad_dir = (1. - skyl) * St
    Srad_diff = skyl * St

    # phi_radians = saa_radians - row_azimuth_radians

    # saltelli.sample create random values that sometime don't make any sense
    # LAI, FV, wc, sp, sr are intrinsically related, so they can not be analysed independently
    # For instance FV = 0 with LAI = 8. wc = 8, sp and sr = 5
    # They'll be estimated mainly based  on the LAI
    K_be = GF.estimate_Kbe(x_LAD=1, sza=0)
    fv0 = np.clip((1 - np.exp(-K_be * lai)) * fv_var, 1e-6, 1)
    w_V = fv0 * row_sep
    plant_sep = row_sep * plant_sep_var # sp [0.5, 1]

    # Relating absortivity, reflectivity and transmissivity of the leaf
    abs_vis_leaf = 1 - rho_vis_leaf - tau_vis_leaf
    abs_nir_leaf = 1 - rho_nir_leaf - tau_nir_leaf

    x_LAD = 1

    start_time_bin = time.perf_counter()
    Sn_V_bin, Sn_S_bin = compute_binomial_prism_manuel(
        sr=row_sep,  # Row spacing (meters)
        sza=sza_radians,  # sun zenith angle (radians)
        psi=phi_radians,  # sun azimuth relative to row orientation (radians)
        lai=lai,  # Leaf Area Index
        ameanv=abs_vis_leaf,  # Leaf absorptivity in the visible (PAR) band
        ameann=abs_nir_leaf,  # Leaf absorptivity in the near infra-red band (NIR) band
        rsoilv=rho_vis_soil,  # Soil absorptivity in the visible (PAR) band
        rsoiln=rho_nir_soil,  # Soil absorptivity in the near infra-red band (NIR) band
        Srad_dir=Srad_dir,  # Direct-beam incoming radiation (W m-2)
        Srad_diff=Srad_diff,  # Diffuse incoming radiation (W m-2)
        fvis=fvis,  # Fraction incoming radiation in the visible part of the spectrum
        fnir=fnir,  # Fraction incoming radiation in the near infra-red part of the spectrum
        CanopyHeight=h_V,  # Canopy heigth (meters)
        wc=w_V,  # Canopy width (meters)
        sp=plant_sep,  # Plant spacing (meters)
        Gtheta=0.5,  # fraction of leaf area projected in the direction of the sun
        nrays=nrays,  #
        Nbins=Nbins,
        shape=shape,  # <-- shape of the canopy
        Nz_diff=Nz_diff,
        Nphi_diff=Nphi_diff,  # sampling for diffuse hemisphere
    )

    end_time_bin = time.perf_counter()
    elapsed_time_bin = end_time_bin - start_time_bin
    print(f"Execution time BINOMIAL: {elapsed_time_bin} seconds")

    start_time_CN = time.perf_counter()
    omega = myTSEB.rectangular_row_clumping_index_parry(
        LAI=lai,
        fv0=fv0,
        w_V=w_V,
        h_V=h_V,
        sza=sza_radians,
        phi=phi_radians,
        hb_V=0.5, L=None, x_LAD=x_LAD)

    Sn_V_CN, Sn_S_CN = myTSEB.shortwave_transmittance_model_CN(
        Sdn_dir=Srad_dir,
        Sdn_dif=Srad_diff,
        fvis=fvis,
        fnir=fnir,
        sza=sza_degrees,
        LAI=lai,
        omega=omega,
        x_LAD=x_LAD,
        rho_vis_leaf=rho_vis_leaf,
        rho_nir_leaf=rho_nir_leaf,
        tau_vis_leaf=tau_vis_leaf,
        tau_nir_leaf=tau_nir_leaf,
        rho_vis_soil=rho_vis_soil,
        rho_nir_soil=rho_nir_soil)

    end_time_CN = time.perf_counter()
    elapsed_time_CN = end_time_CN - start_time_CN
    print(f"Execution time CN: {elapsed_time_CN} seconds")

    Sn_V_CN = Sn_V_CN.reshape(-1, 1)
    Sn_S_CN = Sn_S_CN.reshape(-1, 1)

    # print(Sn_V_bin.shape)
    # print(Sn_V_CN.shape)
    return Sn_V_bin, Sn_S_bin, Sn_V_CN, Sn_S_CN

names = ['lai', 'fv_var', 'h_V', 'row_sep', 'plant_sep_var', 'sza_radians', 'phi_radians',
         'rho_vis_leaf', 'tau_vis_leaf', 'rho_nir_leaf', 'tau_nir_leaf', 'rho_vis_soil', 'rho_nir_soil']

problem = {
    "num_vars": 13,
    "names": names,
    "bounds": [
        [0.2, 5.0],  # 1. lai: orchards/crops
        [0.7, 1.1],  # 2. fv_var
        [0.5, 6.0],  # 3. CanopyHeight
        [2.0, 8.0],  # 4. row_sep: typical orchard rows
        [0.5, 1],  # 5. plant_sep_var: plant space in relation to sr
        [np.deg2rad(15), np.deg2rad(60)],  # 6. sza: avoid near 0 and near 90
        [np.deg2rad(8), np.deg2rad(180)],  # 7. phi: avoid near 0 and near 360
        [0.03, 0.18],  # 8. rho_vis_leaf
        [0.02, 0.10],  # 9. tau_vis_leaf
        [0.32, 0.55],  # 10. rho_nir_leaf (tightened)
        [0.25, 0.45],  # 11. tau_nir_leaf (tightened)
        [0.08, 0.30],  # 12. rho_vis_soil
        [0.20, 0.45],  # 13. rho_nir_soil
    ],
}

N = 500
N_sim = N * (2 * problem['num_vars'] + 2)
print(f"N_sim: {N_sim}")
second_order = False
X = saltelli.sample(problem, N, calc_second_order=second_order)

# Testing with 1d array
Sn_V_bin, Sn_S_bin, Sn_V_CN, Sn_S_CN = contrasting_binomial_CN(
    lai=X[:, 0], fv_var=X[:, 1], h_V=X[:, 2], row_sep=X[:, 3],
    plant_sep_var=X[:, 4],
    sza_radians=X[:, 5], phi_radians=X[:, 6],
    rho_vis_leaf=X[:, 7], tau_vis_leaf=X[:, 8], rho_nir_leaf=X[:, 9], tau_nir_leaf=X[:, 10], rho_vis_soil=X[:, 11],
    rho_nir_soil=X[:, 12],
    Nz_diff=16, Nphi_diff=32, Nbins=40, nrays=5000, shape="prism"
)

inputs_df = pd.DataFrame(X)
inputs_df.columns = names
inputs_df.loc[:, 'Sn_V_bin'] = Sn_V_bin
inputs_df.loc[:, 'Sn_S_bin'] = Sn_S_bin
inputs_df.loc[:, 'Sn_V_CN'] = Sn_V_CN
inputs_df.loc[:, 'Sn_S_CN'] = Sn_S_CN
# inputs_df.to_csv()

Si_Sn_V_bin = sobol.analyze(problem, Sn_V_bin[:, 0], calc_second_order=second_order, print_to_console=False)
Si_Sn_S_bin = sobol.analyze(problem, Sn_S_bin[:, 0], calc_second_order=second_order, print_to_console=False)

Si_Sn_V_CN = sobol.analyze(problem, Sn_V_CN[:, 0], calc_second_order=second_order, print_to_console=False)
Si_Sn_S_CN = sobol.analyze(problem, Sn_S_CN[:, 0], calc_second_order=second_order, print_to_console=False)

df_Sn_V_bin = pd.DataFrame(Si_Sn_V_bin)
df_Sn_V_bin.insert(0, 'params', problem['names'])

df_Sn_S_bin = pd.DataFrame(Si_Sn_S_bin)
df_Sn_S_bin.insert(0, 'params', problem['names'])

df_Sn_V_CN = pd.DataFrame(Si_Sn_V_CN)
df_Sn_V_CN.insert(0, 'params', problem['names'])

df_Sn_S_CN = pd.DataFrame(Si_Sn_S_CN)
df_Sn_S_CN.insert(0, 'params', problem['names'])

path = 'files/binomial_CN_sensitivity_analysis.xlsx'
with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
    inputs_df.to_excel(writer, sheet_name='inputs', index=False)
    df_Sn_V_bin.to_excel(writer, sheet_name='Sn_V_bin', index=False)
    df_Sn_S_bin.to_excel(writer, sheet_name='Sn_S_bin', index=False)
    df_Sn_V_CN.to_excel(writer, sheet_name='Sn_V_CN', index=False)
    df_Sn_S_CN.to_excel(writer, sheet_name='Sn_S_CN', index=False)

df = pd.DataFrame(X)

df.columns = names
df.loc[:, 'Sn_V_bin'] = Sn_V_bin
df.loc[:, 'Sn_S_bin'] = Sn_S_bin
df.loc[:, 'Sn_V_CN'] = Sn_V_CN
df.loc[:, 'Sn_S_CN'] = Sn_S_CN

import seaborn as sns
from matplotlib import pyplot as plt
sns.scatterplot(df, x='Sn_V_bin', y='Sn_V_CN', hue='lai')
plt.plot([0, 1000], [0, 1000])
plt.show()
sns.scatterplot(df, x='Sn_S_bin', y='Sn_S_CN', hue='lai')
plt.plot([0, 1000], [0, 1000])
plt.show()
# sns.scatterplot(df, x='lai', y='Sn_V_bin')
# plt.show()
# sns.scatterplot(df, x='lai', y='Sn_V_CN')
# plt.show()
# sns.scatterplot(df, x='Sdn', y='Sn_V_bin')
# plt.show()
# sns.scatterplot(df, x='Sdn', y='Sn_V_CN')
# plt.show()