import numpy as np
import pandas as pd

import functions as myTSEB
import pyTSEB.meteo_utils as met
from pyTSEB import resistances as res
from pyTSEB import MO_similarity as MO
from pyTSEB import net_radiation as rad
import pyTSEB.TSEB as TSEB


params = (pd.read_csv(rf'files/inputs_sensitivity_analysis.csv')
          .drop('Unnamed: 0', axis=1))

input_dict = {col: params[col].to_numpy() for col in params.columns}

### Net Radiation and clumping
# calculate diffuse/direct ratio
difvis, difnir, fvis, fnir = TSEB.rad.calc_difuse_ratio(
    input_dict['Sdn'],
    input_dict['sza_degrees'],
    press=input_dict['P_atm']
)
skyl = fvis * difvis + fnir * difnir
Sdn_dir = (1. - skyl) * input_dict['Sdn']
Sdn_dif = skyl * input_dict['Sdn']

# incoming long wave radiation
emisAtm = rad.calc_emiss_atm(input_dict['ea'], input_dict['Tair'])
Lsky = emisAtm * met.calc_stephan_boltzmann(input_dict['Tair'])

# We need to compute SAA
# doy = date_ts.dayofyear
# dec_time = date_ts.hour + date_ts.minute / 60.
# sza, saa = met.calc_sun_angles(center_geo[1], center_geo[0], 0, doy, dec_time)


# to take into account row strucure on vegetation clumping
# row_direction = 90
# psi = row_direction - input_dict['saa']


Omega0 = TSEB.CI.calc_omega0_Kustas(
    input_dict['LAI'],
    input_dict['fv'],
    x_LAD=input_dict['x_LAD'],
    isLAIeff=False
)

Omega = TSEB.CI.calc_omega_Kustas(
    Omega0,
    theta,
    w_C=1
)

Omega = TSEB.CI.calc_omega_rows(
    input_dict['LAI'],
    input_dict['fv'],
    theta=input_dict['sza_degrees'],
    psi=psi,
    w_c=input_dict['w_V'],
    x_lad=input_dict['x_LAD']
)

Omega = Omega0
F = input_dict['LAI']/input_dict['fv']
# effective LAI (tree crop)
input_dict['LAI_EFF'] = input_dict['LAI'] * Omega

sn_veg, sn_soil = TSEB.rad.calc_Sn_Campbell(
    input_dict['LAI'],
    input_dict['sza_degrees'],
    Sdn_dir,
    Sdn_dif,
    fvis,
    fnir,
    input_dict['rho_vis_leaf'],
    input_dict['tau_vis_leaf'],
    input_dict['rho_nir_leaf'],
    input_dict['tau_nir_leaf'],
    input_dict['rho_vis_soil'],
    input_dict['rho_nir_soil'],
    x_LAD=input_dict['x_LAD'],
    LAI_eff=input_dict['LAI_EFF']
)

sn_veg[~np.isfinite(sn_veg)] = 0
sn_soil[~np.isfinite(sn_soil)] = 0

input_dict['SN_VEG'] = sn_veg
input_dict['SN_SOIL'] = sn_soil

### Landscape roughness
z_0m, d_0 = TSEB.res.calc_roughness(input_dict['LAI'],
                                    input_dict['h_V'],
                                    input_dict['w_V'],
                                    np.full_like(input_dict['LAI'], TSEB.res.CROP))
d_0[d_0 < 0] = 0
z_0m[z_0m < np.min(params['z0_soil'])] = np.mean(params['z0_soil'])

# store in input dictionary
input_dict['d_0'] = d_0
input_dict['Z_0M'] = z_0m


# use Norman and Kustas 1995 resistance framework
Resistance_flag=[0,{}]
# using constant ratio appraoch to estimate G
G_constant = 0.35
calcG = [[1], G_constant]

Ln_in = rad.calc_longwave_irradiance(
    input_dict['ea'],
    input_dict['Tair'],
    p=input_dict['P_atm'],
    z_T=2.0,
    h_C=2.0
)
input_dict['LW_IN'] = Ln_in

# Ln_V, Ln_S = rad.calc_L_n_Campbell(input_dict[Trad_V, Trad_S, Ln, LAI, emis_leaf, emis_soil, x_LAD=x_LAD)

[flag_PT_all, T_soil, T_veg, T_AC, Ln_soil, Ln_veg, LE_veg, H_veg,
     LE_soil, H_soil, G_mod, R_S, R_X, R_A, u_friction, L, n_iterations] = TSEB.TSEB_PT(
    input_dict['Trad'],
    input_dict['vza_degrees'],
    input_dict['Tair'],
    input_dict['u'],
    input_dict['ea'],
    input_dict['P_atm'],
    input_dict['SN_VEG'],
    input_dict['SN_SOIL'],
    input_dict['LW_IN'],
    input_dict['LAI'],
    input_dict['h_V'],
    input_dict['emis_leaf'],
    input_dict['emis_soil'],
    input_dict['Z_0M'],
    input_dict['d_0'],
    2,
    2,
    leaf_width=input_dict['leaf_width'],
    alpha_PT=input_dict['alpha_PT'],
    f_c=input_dict['fv'],
    f_g=input_dict['fv'],
    calcG_params=calcG,
    resistance_form=Resistance_flag)
print('Done!')

# T_soil, T_veg, T_AC, Ln_soil, Ln_veg, LE_veg, H_veg,
#      LE_soil, H_soil, G_mod
output_dict = {'Rn_V_pyTSEB': sn_veg + Ln_veg,
               'Rn_S_pyTSEB': sn_soil + Ln_soil,
               'LE_V_pyTSEB': LE_veg,
               'LE_S_pyTSEB': LE_soil,
               'H_V_pyTSEB': H_veg,
               'H_S_pyTSEB': H_soil,
               'G_pyTSEB': G_mod,
               'Trad_V_pyTSEB': T_veg,
               'Trad_S_pyTSEB': T_soil}
output_df = pd.DataFrame(output_dict)
output_df.to_csv(rf'files/pyTSEB_outputs.csv')