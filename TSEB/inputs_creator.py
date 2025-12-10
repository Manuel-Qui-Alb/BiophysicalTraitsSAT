import numpy as np
import pandas as pd

import functions as myTSEB
import pyTSEB.meteo_utils as met
from pyTSEB import resistances as res
from pyTSEB import MO_similarity as MO
import pyTSEB.TSEB as TSEB


n = 10000
# Convert the angle to radians
Tair = np.random.uniform(283, 313, n)
diff = np.random.uniform(0,15,n)
Trad = Tair + diff
Trad_V_0 = np.min([Trad, Tair], axis=0)

LAI = np.random.uniform(0, 8, n)
fv = np.random.uniform(0, 1, n)
F = np.asarray(LAI / fv, dtype=np.float32)
h_V = np.random.uniform(0, 2, n)
w_V = np.random.uniform(0, 1, n)

ea = np.random.uniform(0,80,n)
P_atm = np.random.uniform(990, 1010, n)
u = np.random.uniform(0, 20, n)
c_p = met.calc_c_p(P_atm, ea)

sza_degrees = np.random.uniform(0, 90,n) #(?) between 0 to 60% does not matter. solar zenith angle
sza_rad = np.radians(sza_degrees)

vza_degrees = np.random.uniform(0, 90,n)  # View zenith angle
vza_rad = np.radians(vza_degrees)
x_LAD = np.round(np.random.default_rng().uniform(0, 2,n), 1)
Sdn = np.random.uniform(0, 1000, n)

z_T = np.full(LAI.shape, 2)
z_u = np.full(LAI.shape, 2)

KB_1_DEFAULT = np.full(LAI.shape, 0.0)
alpha_PT = np.full_like(LAI, 1.7)

# None Parameters
ITERATIONS = 15
L = np.zeros(Trad.shape) + np.inf
max_iterations = ITERATIONS
massman_profile = [0, []]

G_constant = 0.35
calcG_params = [[1], G_constant]
G_ratio = 0.35

(emis_leaf, emis_soil, rho_vis_leaf, tau_vis_leaf, rho_nir_leaf, tau_nir_leaf, rho_vis_soil, rho_nir_soil, z0_soil,
 leaf_width) = myTSEB.get_emiss_rho_tau(LAI)
params = dict(
    Tair=Tair, Trad=Trad, Trad_V_0=Trad_V_0,
    LAI=LAI, fv=fv, F=F, h_V=h_V, w_V=w_V,
    ea=ea, P_atm=P_atm, u=u, c_p=c_p,
    sza_degrees=sza_degrees, sza_rad=sza_rad,
    vza_degrees=vza_degrees, vza_rad=vza_rad, x_LAD=x_LAD, Sdn=Sdn,
    z_T=z_T, z_u=z_u,
    KB_1_DEFAULT=KB_1_DEFAULT,
    alpha_PT=alpha_PT,
    L=L,
    emis_leaf=emis_leaf,
    emis_soil=emis_soil,
    rho_vis_leaf=rho_vis_leaf,
    tau_vis_leaf=tau_vis_leaf,
    rho_nir_leaf=rho_nir_leaf,
    tau_nir_leaf=tau_nir_leaf,
    rho_vis_soil=rho_vis_soil,
    rho_nir_soil=rho_nir_soil,
    z0_soil=z0_soil,
    leaf_width=leaf_width
)

params_df = pd.DataFrame(params)
params_df.to_csv(rf'files/inputs_sensitivity_analysis.csv')