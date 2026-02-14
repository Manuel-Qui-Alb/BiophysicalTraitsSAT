import numpy as np
import pandas as pd

import functions as myTSEB
import pyTSEB.meteo_utils as met
from pyTSEB import resistances as res
from pyTSEB import MO_similarity as MO
from pyTSEB import net_radiation as rad
import pyTSEB.TSEB as TSEB
import GeneralFunctions as GF


params = pd.read_csv(rf'files/TSEB_CN.csv' )

input_dict = {col: params[col].to_numpy() for col in params.columns}
input_dict.update({'sza_radians': np.deg2rad(input_dict['sza_degrees'])})
# input_dict

m = 101.3 / (101.3 * np.cos(input_dict['sza_radians']))
tau_atms = 0.6  # atmospheric transmittance
Sp0 = 1368  # Extraterrestrial flux density
Sp = 1368 * (tau_atms ** m)
Sb = Sp * np.cos(input_dict['sza_radians'])

Sd = 0.3 * (1 - tau_atms ** m) * Sp0 * np.cos(input_dict['sza_radians'])
    # f_diff = np.clip(0.15 + 0.6 * (1 - mu), 0.15, 0.7)
St = Sb + Sd
input_dict.update({'Sdn': St})

K_be = GF.estimate_Kbe(x_LAD=input_dict['x_LAD'], sza=0)
fv0 = np.clip((1 - np.exp(-K_be * input_dict['LAI'])) * input_dict['fv_var'], 1e-6, 1)
input_dict.update({'fv': fv0})

w_V = fv0 * input_dict['row_sep']
input_dict.update({'w_V': w_V})
# plant_sep = input_dict['row_sep'] * input_dict['plant_sep_var']  # sp [0.5, 1]

input_dict.update({'Trad': input_dict['Tair'] + input_dict['Trad_var']})
input_dict.update({'z_u': input_dict['h_V'] + 2})
input_dict.update({'z_T': input_dict['h_V'] + 2})


### Net Radiation and clumping
# calculate diffuse/direct ratio
difvis, difnir, fvis, fnir = TSEB.rad.calc_difuse_ratio(
    S_dn=St,
    sza=input_dict['sza_degrees'],
    press=np.full_like(St, 1013)
)

skyl = fvis * difvis + fnir * difnir
Srad_dir = (1. - skyl) * St
Srad_diff = skyl * St

# incoming long wave radiation
emisAtm = rad.calc_emiss_atm(input_dict['ea'], input_dict['Tair'])
Lsky = emisAtm * met.calc_stephan_boltzmann(input_dict['Tair'])

# We need to compute SAA
# doy = date_ts.dayofyear
# dec_time = date_ts.hour + date_ts.minute / 60.
# sza, saa = met.calc_sun_angles(center_geo[1], center_geo[0], 0, doy, dec_time)


# to take into account row strucure on vegetation clumping
# row_direction = 90
# psi = np.arccos(np.cos(row_direction - input_dict['saa']))


Omega0 = TSEB.CI.calc_omega0_Kustas(
    input_dict['LAI'],
    input_dict['fv'],
    x_LAD=input_dict['x_LAD'],
    isLAIeff=False
)

Omega_H = TSEB.CI.calc_omega_Kustas(
    Omega0,
    input_dict['sza_degrees'],
    w_C=input_dict['w_V'] / input_dict['h_V']
)

Omega_R = TSEB.CI.calc_omega_rows(
    input_dict['LAI'],
    input_dict['fv'],
    theta=input_dict['sza_degrees'],
    psi=input_dict['phi_degrees'],
    w_c=input_dict['w_V'] / input_dict['h_V'],
    x_lad=input_dict['x_LAD'],
    is_lai_eff=False
)

Omega = Omega_R
F = input_dict['LAI'] / input_dict['fv']
# effective LAI (tree crop)
input_dict['LAI_EFF'] = input_dict['LAI'] * Omega

sn_veg, sn_soil = TSEB.rad.calc_Sn_Campbell(
    lai=input_dict['LAI'],
    sza=input_dict['sza_degrees'],
    S_dn_dir=Srad_dir,
    S_dn_dif=Srad_diff,
    fvis=fvis,
    fnir=fnir,
    rho_leaf_vis=input_dict['rho_vis_leaf'],
    tau_leaf_vis=input_dict['tau_vis_leaf'],
    rho_leaf_nir=input_dict['rho_nir_leaf'],
    tau_leaf_nir=input_dict['tau_nir_leaf'],
    rsoilv=input_dict['rho_vis_soil'],
    rsoiln=input_dict['rho_nir_soil'],
    x_LAD=input_dict['x_LAD'],
    LAI_eff=input_dict['LAI_EFF'])

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
z0_soil = np.full(input_dict['LAI'].shape, 0.01)
z_0m[z_0m < np.min(z0_soil)] = np.mean(z0_soil)

# store in input dictionary
input_dict['d_0'] = d_0
input_dict['Z_0M'] = z_0m


# use Norman and Kustas 1995 resistance framework
Resistance_flag=[0,{}]
# using constant ratio appraoch to estimate G
G_constant = 0.35
calcG = [[1], G_constant]

Ln_in = rad.calc_longwave_irradiance(
    ea=input_dict['ea'],
    t_a_k=input_dict['Tair'],
    p=input_dict['P_atm'],
    z_T=input_dict['z_T'],
    h_C=input_dict['h_V']
)
input_dict['LW_IN'] = Ln_in

# Ln_V, Ln_S = rad.calc_L_n_Campbell(input_dict[Trad_V, Trad_S, Ln, LAI, emis_leaf, emis_soil, x_LAD=x_LAD)
[flag_PT_all, f_theta, T_soil, T_veg, T_AC, Ln_soil, Ln_veg, LE_veg, H_veg,
 LE_soil, H_soil, G_mod, R_S, R_X, R_A, u_friction, L, n_iterations] = TSEB.TSEB_PT(
    Tr_K=input_dict['Trad'],
    vza=input_dict['sza_degrees'],
    T_A_K=input_dict['Tair'],
    u=input_dict['u'],
    ea=input_dict['ea'],
    p=input_dict['P_atm'],
    Sn_C=input_dict['SN_VEG'],
    Sn_S=input_dict['SN_SOIL'],
    L_dn=input_dict['LW_IN'],
    LAI=input_dict['LAI'],
    h_C=input_dict['h_V'],
    emis_C=input_dict['emis_leaf'],
    emis_S=input_dict['emis_soil'],
    z_0M=input_dict['Z_0M'],
    d_0=input_dict['d_0'],
    z_u=input_dict['z_u'],
    z_T=input_dict['z_T'],
    leaf_width=input_dict['leaf_width'],
    z0_soil=0.01,
    alpha_PT=1.26,
    x_LAD=input_dict['x_LAD'],
    f_c=input_dict['fv'],
    f_g=1,
    w_C=input_dict['w_V'] / input_dict['h_V'],
    resistance_form=Resistance_flag,
    calcG_params=calcG,
            # const_L=None,
            # kB=KB_1_DEFAULT,
            # massman_profile=None,
            # verbose=True
             )
print('Done!')

# T_soil, T_veg, T_AC, Ln_soil, Ln_veg, LE_veg, H_veg,
#      LE_soil, H_soil, G_mod

output_df = pd.DataFrame(input_dict)
output_df.loc[:, 'omega_pyTSEB'] = Omega
output_df.loc[:, 'f_theta_pyTSEB'] = f_theta
output_df.loc[:, 'SN_V_pyTSEB'] = sn_veg
output_df.loc[:, 'SN_S_pyTSEB'] = sn_soil
output_df.loc[:, 'Trad_V_pyTSEB'] = T_veg
output_df.loc[:, 'Trad_S_pyTSEB'] = T_soil
output_df.loc[:, 'Ln_V_pyTSEB'] = Ln_veg
output_df.loc[:, 'Ln_S_pyTSEB'] = Ln_soil
output_df.loc[:, 'Rn_V_pyTSEB'] = sn_veg + Ln_veg
output_df.loc[:, 'Rn_S_pyTSEB'] = sn_soil + Ln_soil
output_df.loc[:, 'LE_V_pyTSEB'] = LE_veg
output_df.loc[:, 'LE_S_pyTSEB'] = LE_soil
output_df.loc[:, 'flag_pyTSEB'] = flag_PT_all
# output_df = pd.DataFrame(output_dict)
output_df.to_csv(rf'files/pyTSEB_outputs.csv')