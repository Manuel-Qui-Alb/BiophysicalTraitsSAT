import numpy as np
import pandas as pd

import functions as myTSEB
import pyTSEB.meteo_utils as met
from pyTSEB import resistances as res
from pyTSEB import MO_similarity as MO
import pyTSEB.TSEB as TSEB


params = (pd.read_csv(rf'files/inputs_sensitivity_analysis.csv')
          .drop('Unnamed: 0', axis=1))


params = {col: params[col].to_numpy() for col in params.columns}

# None Parameters
ITERATIONS = 15
L = np.zeros(np.array(params['LAI']).shape) + np.inf
max_iterations = ITERATIONS
massman_profile = [0, []]

G_constant = 0.35
calcG_params = [[1], G_constant]
G_ratio = 0.35


########################################################################################################################
# f_theta = fraction of incident beam radiation intercepted by the plant
# f_theta correspond to the radiation available for scattering, transpiration, and photosynthesis.
########################################################################################################################
omega = myTSEB.estimate_omega0(
    params['LAI'],
    params['fv'],
    params['x_LAD'],
    params['sza_rad']
)
params.update({'omega': omega})

f_theta = myTSEB.estimate_f_theta(
    LAI=params['LAI'],
    x_LAD=params['x_LAD'],
    omega=params['omega'],
    sza=params['sza_rad']
)
params.update({'f_theta': f_theta})

########################################################################################################################
# R_x = boundary layer resistance of the complete canopy of leaves
# R_S = soil-surface resistance
# R_A = aerodynamic resistance above the canopy
########################################################################################################################
z_0m, d_0 = TSEB.res.calc_roughness(
    params['LAI'],
    params['h_V'],
    params['w_V'],
    np.full_like(params['LAI'], TSEB.res.CROP),
    f_c=params['fv']
)
d_0[d_0 < 0] = 0
z_0m[z_0m < np.min(params['z0_soil'])] = np.mean(params['z0_soil'])
params.update({'d_0': d_0, 'z_0m': z_0m})

KB_1_DEFAULTC = 0
z_0H = res.calc_z_0H(
    params['z_0m'],
    kB=KB_1_DEFAULTC)
params.update({'z_0H': z_0H})

res_params = [0,{}]
rho = met.calc_rho(
    params['P_atm'],
    params['ea'],
    params['Tair']
)
T_AC = params['Tair'].copy()
params.update({'rho': rho, 'T_AC': T_AC})

u_friction = MO.calc_u_star(
    params['u'],
    params['z_u'],
    params['L'],
    params['d_0'],
    params['z_0m']
)
U_FRICTION_MIN = np.full_like(params['LAI'], 0.01)
# params.update({'u_friction': u_friction, 'U_FRICTION_MIN': U_FRICTION_MIN})

u_friction = np.asarray(np.nanmax([U_FRICTION_MIN, u_friction], axis=0), dtype=np.float32)
params.update({'u_friction': u_friction})

resistance_form=[0,{}]
res_params = resistance_form[1]
resistance_form = 0 #KUSTAS_NORMAN_1999 (0), CHOUDHURY_MONTEITH_1988 (1), MCNAUGHTON_VANDERHURK (2), CHOUDHURY_MONTEITH_ALPHA_1988(3)

########################################################################################################################
# Fist Net Radiation estimation assuming Trad_V = Tair
# This first estimation of Trad_S will be used to estimate R_S, Ln_S and R_S
########################################################################################################################
Trad_S_0 = myTSEB.estimate_Trad_S(
    Trad=params['Trad'],
    Trad_V=params['Trad_V_0'],
    f_theta=params['f_theta']
)
params.update({'Trad_S_0': Trad_S_0})

R_A, R_x, R_S = TSEB.calc_resistances(
    resistance_form,
    {
        "R_A": {
            "z_T": params['z_T'],
            "u_friction": params['u_friction'],
            "L": params['L'],
            "d_0": params['d_0'],
            "z_0H": params['z_0H']
        },
        "R_x": {
            "u_friction": params['u_friction'],
            "h_C": params['h_V'],
            "d_0": params['d_0'],
            "z_0M": params['z_0m'],
            "L": params['L'],
            "F": params['F'],
            "LAI": params['LAI'],
            "leaf_width": params['leaf_width'],
            "z0_soil": params['z0_soil'],
            "massman_profile": massman_profile,
            "res_params": {k: res_params[k] for k in res_params.keys()}
        },
        "R_S": {
            "u_friction": params['u_friction'],
            "h_C": params['h_V'],
            "d_0": params['d_0'],
            "z_0M": params['z_0m'],
            "L": params['L'],
            "F": params['F'],
            "omega0": params['omega'],
            "LAI": params['LAI'],
            "leaf_width": params['leaf_width'],
            "z0_soil": params['z0_soil'],
            "z_u": params['z_u'],
            "deltaT": params['Trad_S_0'] - params['T_AC'],
            "u": params['u'],
            "rho": params['rho'],
            "c_p": params['c_p'],
            "f_cover": params['fv'],
            "w_C": params['w_V'],
            "massman_profile": massman_profile,
            "res_params": {k: res_params[k] for k in res_params.keys()}}
     }
)
params.update({'R_A': R_A, 'R_x': R_x, 'R_S': R_S})

# con = Trad_S<350
Rn_V, Rn_S = myTSEB.estimate_Rn(
    S_dn=params['Sdn'],
    sza=params['sza_degrees'],
    P_atm=params['P_atm'],
    LAI=params['LAI'],
    x_LAD=params['x_LAD'],
    omega=params['omega'],
    Tair=params['Tair'],
    ea=params['ea'],
    Trad_S=params['Trad_S_0'],
    Trad_V=params['Trad_V_0'],
    rho_vis_leaf=params['rho_vis_leaf'],
    rho_nir_leaf=params['rho_nir_leaf'],
    tau_vis_leaf=params['tau_vis_leaf'],
    tau_nir_leaf=params['tau_nir_leaf'],
    rho_vis_soil=params['rho_vis_soil'],
    rho_nir_soil=params['rho_nir_soil'],
    emis_leaf=params['emis_leaf'],
    emis_soil=params['emis_soil']
)
# params.update({'Rn_V': Rn_S, 'Rn_S': Rn_S})

# Eliminate values without sense Rn_V < 0
idx = np.where(Rn_V>0)
Rn_V = Rn_V[idx]
Rn_S = Rn_S[idx]
dict_keys = list(params.keys())
params_values = [params[x][idx] for x in dict_keys]
params = dict(zip(dict_keys, params_values))


[LE_V, LE_S, H_V, H_S, G, Trad_V, Trad_S] = [np.full_like(params['LAI'], -9999)  for i in range(7)]


def iteration_func(loop_con):
# while len(loop_con)>0:
    print(len(loop_con))
    params['alpha_PT'][loop_con]-= 0.1
    params['alpha_PT'][loop_con] = np.where(params['alpha_PT'][loop_con] < 0, 0, params['alpha_PT'][loop_con])

    ########################################################################################################################
    # Maximum Vegetation Latent Heat Flux (LE_V) and its respective Latent Heat Flux (H_V) using Priestly-Taylor
    ########################################################################################################################
    # print(np.max(alpha_PT[idx]))
    LE_V[loop_con] = myTSEB.Priestly_Taylor_LE_V(
        fv_g=params['f_theta'][loop_con],
        Rn_V=Rn_V[loop_con],
        alpha_PT=params['alpha_PT'][loop_con],
        Tair=params['Tair'][loop_con],
        P_atm=params['P_atm'][loop_con],
        c_p=params['c_p'][loop_con]
    )

    H_V[loop_con] = Rn_V[loop_con] - LE_V[loop_con]

    ########################################################################################################################
    # Reestimate of Trad_V and Trad_S using Sensible Heat Flux
    ########################################################################################################################
    Trad_V[loop_con] = TSEB.calc_T_C_series(
        Tr_K=params['Trad'][loop_con],
        T_A_K=params['Tair'][loop_con],
        R_A=params['R_A'][loop_con],
        R_x=params['R_x'][loop_con],
        R_S=params['R_S'][loop_con],
        f_theta=params['f_theta'][loop_con],
        H_C=H_V[loop_con],
        rho=params['rho'][loop_con],
        c_p=params['c_p'][loop_con]
    )

    Trad_S[loop_con] = myTSEB.estimate_Trad_S(
        Trad=params['Trad'][loop_con],
        Trad_V=Trad_V[loop_con],
        f_theta=params['f_theta'][loop_con]
    )

    ########################################################################################################################
    # Reestimate Soil Sensible Heat Flux (H_S) because it depends on Trad_S
    ########################################################################################################################

    _, _, R_S[loop_con] = TSEB.calc_resistances(
                    resistance_form,
                    {"R_S": {
                        "u_friction": params['u_friction'][loop_con],
                        "h_C": params['h_V'][loop_con],
                        "d_0": params['d_0'][loop_con],
                        "z_0M": params['z_0m'][loop_con],
                        "L": params['L'][loop_con],
                        "F": params['F'][loop_con],
                        "omega0": params['omega'][loop_con],
                        "LAI": params['LAI'][loop_con],
                        "leaf_width": params['leaf_width'][loop_con],
                        "z0_soil": params['z0_soil'][loop_con],
                        "z_u": params['z_u'][loop_con],
                        "deltaT": Trad_S[loop_con] - params['T_AC'][loop_con], # Trad_S
                        "u": params['u'][loop_con],
                        "rho": params['rho'][loop_con],
                        "c_p": params['c_p'][loop_con],
                        "f_cover": params['fv'][loop_con],
                        "w_C": params['w_V'][loop_con],
                        "massman_profile": massman_profile,
                        "res_params": {k: res_params[k] for k in res_params.keys()}}
                     }
                )

    # Get air temperature at canopy interface
    params['T_AC'][loop_con] = ((params['Tair'][loop_con] /
                                 params['R_A'][loop_con] +
                                 Trad_S[loop_con] /
                                 params['R_S'][loop_con] +
                                 Trad_V[loop_con] /
                                 params['R_x'][loop_con])
                                /
                                (1.0 / params['R_A'][loop_con] +
                                 1.0 / params['R_S'][loop_con] +
                                 1.0 / params['R_x'][loop_con]))

    # Calculate soil fluxes
    H_S[loop_con] = (params['rho'][loop_con] * params['c_p'][loop_con] *
                     (Trad_S[loop_con] - params['T_AC'][loop_con]) /
                     params['R_S'][loop_con])

    ########################################################################################################################
    # Compute Soil Heat Flux Ratio as a Ratio of Rn_S
    ########################################################################################################################
    G[loop_con] = G_ratio * Rn_S[loop_con]

    ########################################################################################################################
    # Reestimate Net Radiation estimation with Trad_V and Trad_S from Sensible Heat Flux
    ########################################################################################################################
    Rn_V[loop_con], Rn_S[loop_con] = myTSEB.estimate_Rn(
        S_dn=params['Sdn'][loop_con],
        sza=params['sza_degrees'][loop_con],
        P_atm=params['P_atm'][loop_con],
        LAI=params['LAI'][loop_con],
        x_LAD=params['x_LAD'][loop_con],
        omega=params['omega'][loop_con],
        Tair=params['Tair'][loop_con],
        ea=params['ea'][loop_con],
        Trad_S=Trad_S[loop_con],
        Trad_V=Trad_V[loop_con],
        rho_vis_leaf=params['rho_vis_leaf'][loop_con],
        rho_nir_leaf=params['rho_nir_leaf'][loop_con],
        tau_vis_leaf=params['tau_vis_leaf'][loop_con],
        tau_nir_leaf=params['tau_nir_leaf'][loop_con],
        rho_vis_soil=params['rho_vis_soil'][loop_con],
        rho_nir_soil=params['rho_nir_soil'][loop_con],
        emis_leaf=params['emis_leaf'][loop_con],
        emis_soil=params['emis_soil'][loop_con]
    )


    LE_S[loop_con] = Rn_S[loop_con] - G[loop_con] - H_S[loop_con]
    LE_V[loop_con] = Rn_V[loop_con] - H_V[loop_con]
    return Rn_V, Rn_S, LE_V, LE_S, H_V, H_S, G, Trad_V, Trad_S

loop_con = np.where(LE_S < 0)[0]
max_iterations = 50
iterations = 1
while (len(loop_con)>0 and max_iterations>=iterations):
    print(iterations)
    iterations += 1
    Rn_V, Rn_S, LE_V, LE_S, H_V, H_S, G, Trad_V, Trad_S = iteration_func(loop_con)
    loop_con = np.where(LE_S < 0)[0]

output_dict = {'Rn_V_myTSEB': Rn_V,
               'Rn_S_myTSEB': Rn_S,
               'LE_V_myTSEB': LE_V,
               'LE_S_myTSEB': LE_S,
               'H_V_myTSEB': H_V,
               'H_S_myTSEB': H_S,
               'G_myTSEB': G,
               'Trad_V_myTSEB': Trad_V,
               'Trad_S_myTSEB': Trad_S}
output_df = pd.DataFrame(output_dict)
output_df.to_csv(rf'files/myTSEB_outputs.csv')