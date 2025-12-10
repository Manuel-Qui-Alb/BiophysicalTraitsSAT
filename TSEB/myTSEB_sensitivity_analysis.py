import numpy as np
import pandas as pd

import functions as myTSEB
import pyTSEB.meteo_utils as met
from pyTSEB import resistances as res
from pyTSEB import MO_similarity as MO
import pyTSEB.TSEB as TSEB


# params = (pd.read_csv(rf'files/inputs_sensitivity_analysis.csv')
#           .drop('Unnamed: 0', axis=1))
#
#
# params = {col: params[col].to_numpy() for col in params.columns}

# None Parameters
ITERATIONS = 15
# L = np.zeros(np.array(params['LAI']).shape) + np.inf
max_iterations = ITERATIONS
massman_profile = [0, []]

G_constant = 0.1
calcG_params = [[1], G_constant]
G_ratio = 0.1


########################################################################################################################
# f_theta = fraction of incident beam radiation intercepted by the plant
# f_theta correspond to the radiation available for scattering, transpiration, and photosynthesis.
########################################################################################################################
# myTSEB(LAI, fv, x_LAD, sza_rad)
def sensitivity_analysis_myTSEB(LAI, fv, h_V, row_sep, x_LAD, leaf_width, Trad,
                                Tair, Sdn, u, P_atm, ea, sza_degrees, saa_degrees, vza_degrees,
                                row_azimuth=None):
    Trad = Tair + Trad
    z_u = h_V + 2
    z_T = h_V + 2
    # print(Trad)
    w_V = fv * row_sep
    sza_rad =  np.radians(sza_degrees)
    vza_rad = np.radians(vza_degrees)
    saa_rad = np.radians(saa_degrees)
    row_azimuth = np.radians(row_azimuth)
    F = np.asarray(LAI / fv, dtype=np.float32)
    c_p = met.calc_c_p(P_atm, ea)

    KB_1_DEFAULT = np.full(LAI.shape, 0.0)
    alpha_PT = np.full_like(LAI, 1.7)
    L = np.zeros(Trad.shape) + np.inf

    (emis_leaf, emis_soil, rho_vis_leaf, tau_vis_leaf, rho_nir_leaf, tau_nir_leaf, rho_vis_soil, rho_nir_soil,
     z0_soil) = myTSEB.get_emiss_rho_tau(LAI)

    omega = myTSEB.rectangular_row_clumping_index_parry(
        LAI=LAI,
        fv0=fv,
        w_V=w_V,
        h_V=h_V,
        sza=sza_rad,
        saa=saa_rad,
        row_azimuth=row_azimuth,
        hb_V=0,
        L=None,
        x_LAD=1
    )
    # params.update({'omega': omega})

    f_theta = myTSEB.estimate_f_theta(
        LAI=LAI,
        x_LAD=x_LAD,
        omega=omega,
        sza=sza_rad
    )
    # params.update({'f_theta': f_theta})

    ########################################################################################################################
    # R_x = boundary layer resistance of the complete canopy of leaves
    # R_S = soil-surface resistance
    # R_A = aerodynamic resistance above the canopy
    ########################################################################################################################
    z_0m, d_0 = TSEB.res.calc_roughness(
        LAI,
        h_V,
        w_V,
        np.full_like(LAI, TSEB.res.CROP),
        f_c=fv
    )
    d_0[d_0 < 0] = 0
    z_0m[z_0m < np.min(z0_soil)] = np.mean(z0_soil)
    # params.update({'d_0': d_0, 'z_0m': z_0m})

    KB_1_DEFAULTC = 0
    z_0H = res.calc_z_0H(
        z_0m,
        kB=KB_1_DEFAULTC)
    # params.update({'z_0H': z_0H})

    res_params = [0,{}]
    rho = met.calc_rho(
        P_atm,
        ea,
        Tair
    )
    T_AC = Tair.copy()
    # params.update({'rho': rho, 'T_AC': T_AC})

    u_friction = MO.calc_u_star(
        u,
        z_u,
        L,
        d_0,
        z_0m
    )
    U_FRICTION_MIN = np.full_like(LAI, 0.01)
    # params.update({'u_friction': u_friction, 'U_FRICTION_MIN': U_FRICTION_MIN})

    u_friction = np.asarray(np.nanmax([U_FRICTION_MIN, u_friction], axis=0), dtype=np.float32)
    # params.update({'u_friction': u_friction})

    resistance_form=[0,{}]
    res_params = resistance_form[1]
    resistance_form = 0 #KUSTAS_NORMAN_1999 (0), CHOUDHURY_MONTEITH_1988 (1), MCNAUGHTON_VANDERHURK (2), CHOUDHURY_MONTEITH_ALPHA_1988(3)

    ########################################################################################################################
    # Fist Net Radiation estimation assuming Trad_V = Tair
    # This first estimation of Trad_S will be used to estimate R_S, Ln_S and R_S
    ########################################################################################################################
    Trad_V0 = np.min([Tair, Trad], axis=0) # We set the first Trad_V equal to Tair, under potential ET conditions.
    Trad_S_0 = myTSEB.estimate_Trad_S(
        Trad=Trad,
        Trad_V=Trad_V0,
        f_theta=f_theta
    )
    # params.update({'Trad_S_0': Trad_S_0})

    R_A, R_x, R_S = TSEB.calc_resistances(
        resistance_form,
        {
            "R_A": {
                "z_T": z_T,
                "u_friction": u_friction,
                "L": L,
                "d_0": d_0,
                "z_0H": z_0H,
            },
            "R_x": {
                "u_friction": u_friction,
                "h_C": h_V,
                "d_0": d_0,
                "z_0M": z_0m,
                "L": L,
                "F": F,
                "LAI": LAI,
                "leaf_width": leaf_width,
                "z0_soil": z0_soil,
                "massman_profile": massman_profile,
                "res_params": {k: res_params[k] for k in res_params.keys()},
            },
            "R_S": {
                "u_friction": u_friction,
                "h_C": h_V,
                "d_0": d_0,
                "z_0M": z_0m,
                "L": L,
                "F": F,
                "omega0": omega,
                "LAI": LAI,
                "leaf_width": leaf_width,
                "z0_soil": z0_soil,
                "z_u": z_u,
                "deltaT": Trad_S_0 - T_AC,
                "u": u,
                "rho": rho,
                "c_p": c_p,
                "f_cover": fv,
                "w_C": w_V,
                "massman_profile": massman_profile,
                "res_params": {k: res_params[k] for k in res_params.keys()},
            },
        },
    )

    if np.any(np.array([R_A, R_x, R_S])<=0):
        print([R_A, R_x, R_S])

    # params.update({'R_A': R_A, 'R_x': R_x, 'R_S': R_S})

    # con = Trad_S<350
    Rn_V0, Rn_S0 = myTSEB.estimate_Rn(
        S_dn=Sdn,
        sza=sza_degrees,
        P_atm=P_atm,
        LAI=LAI,
        x_LAD=x_LAD,
        omega=omega,
        Tair=Tair,
        ea=ea,
        Trad_S=Trad_S_0,
        Trad_V=Trad_V0,
        rho_vis_leaf=rho_vis_leaf,
        rho_nir_leaf=rho_nir_leaf,
        tau_vis_leaf=tau_vis_leaf,
        tau_nir_leaf=tau_nir_leaf,
        rho_vis_soil=rho_vis_soil,
        rho_nir_soil=rho_nir_soil,
        emis_leaf=emis_leaf,
        emis_soil=emis_soil
    )
    # idx = np.where(Rn_V > 0)
    # Rn_V0 = np.where(Rn_V0 < 0, 0, Rn_V0)

    # if (Rn_V > 1000 or Rn_V < 0):
        # print(Rn_V)# unrealistic combination
        # return np.nan
    # Rn_V = np.where(Rn_V > 1000, 1000, Rn_V)
    # print(Rn_V)
    Rn_V0 = np.array(np.clip(Rn_V0, -150, 1000))
    Rn_S0 = np.array(np.clip(Rn_S0, -150, 1000))

    [LE_V, LE_S, H_V, H_S, G, Trad_V, Trad_S] = [np.full_like(LAI, -9999) for i in range(7)]

    loop_con = True
    # max_iterations = 17
    iterations = 1
    alpha_condition = True

    while (loop_con and alpha_condition):
        # while len(loop_con)>0:
        # print(loop_con))
        # print('Iteration', iterations)
        iterations += 1
        alpha_PT -= 0.1
        alpha_condition = alpha_PT > 0

        alpha_PT = np.where(alpha_PT < 0, 0, alpha_PT)

        try:
            Rn_V0 = Rn_V
            Rn_S0 = Rn_S
        except:
            None

        LE_V = myTSEB.Priestly_Taylor_LE_V(
            fv_g=f_theta,
            Rn_V=Rn_V0,
            alpha_PT=alpha_PT,
            Tair=Tair,
            P_atm=P_atm,
            c_p=c_p
        )


        H_V = Rn_V0 - LE_V
        # print(rf'Alpha: {alpha_PT}, Rn_V0: {Rn_V0}, H_V: {H_V}')

        # print(alpha_PT)
        # print(LE_V)

        ########################################################################################################################
        # Reestimate of Trad_V and Trad_S using Sensible Heat Flux
        ########################################################################################################################
        Trad_V = TSEB.calc_T_C_series(
            Tr_K=Trad,
            T_A_K=Tair,
            R_A=R_A,
            R_x=R_x,
            R_S=R_S,
            f_theta=f_theta,
            H_C=H_V,
            rho=rho,
            c_p=c_p
        )
        # print(Trad_V)

        Trad_S = myTSEB.estimate_Trad_S(
            Trad=Trad,
            Trad_V=Trad_V,
            f_theta=f_theta
        )
        # print(rf'Alpha: {alpha_PT}, Trad_S: {Trad_S}, Trad_V: {Trad_V}')
        # print(Trad_S)
        ########################################################################################################################
        # Reestimate Soil Sensible Heat Flux (H_S) because it depends on Trad_S
        ########################################################################################################################
        _, _, R_S = TSEB.calc_resistances(
            resistance_form,
            {
                "R_S": {
                    "u_friction": u_friction,
                    "h_C": h_V,
                    "d_0": d_0,
                    "z_0M": z_0m,
                    "L": L,
                    "F": F,
                    "omega0": omega,
                    "LAI": LAI,
                    "leaf_width": leaf_width,
                    "z0_soil": z0_soil,
                    "z_u": z_u,
                    "deltaT": Trad_S - T_AC,  # Trad_S - T_AC
                    "u": u,
                    "rho": rho,
                    "c_p": c_p,
                    "f_cover": fv,
                    "w_C": w_V,
                    "massman_profile": massman_profile,
                    "res_params": {k: res_params[k] for k in res_params.keys()},
                }
            }
        )

        # 2) Air temperature at canopy interface (T_AC), array form
        T_AC = (
                (Tair / R_A + Trad_S / R_S + Trad_V / R_x) /
                (1.0 / R_A + 1.0 / R_S + 1.0 / R_x)
        )

        # 3) Soil sensible heat flux H_S, array form
        H_S = rho * c_p * (Trad_S - T_AC) / R_S

        ########################################################################################################################
        # Compute Soil Heat Flux Ratio as a Ratio of Rn_S
        ########################################################################################################################
        G = G_ratio * Rn_S0

        ########################################################################################################################
        # Reestimate Net Radiation estimation with Trad_V and Trad_S from Sensible Heat Flux
        ########################################################################################################################
        Rn_V, Rn_S = myTSEB.estimate_Rn(
            S_dn=Sdn,
            sza=sza_degrees,
            P_atm=P_atm,
            LAI=LAI,
            x_LAD=x_LAD,
            omega=omega,
            Tair=Tair,
            ea=ea,
            Trad_S=Trad_S,
            Trad_V=Trad_V,
            rho_vis_leaf=rho_vis_leaf,
            rho_nir_leaf=rho_nir_leaf,
            tau_vis_leaf=tau_vis_leaf,
            tau_nir_leaf=tau_nir_leaf,
            rho_vis_soil=rho_vis_soil,
            rho_nir_soil=rho_nir_soil,
            emis_leaf=emis_leaf,
            emis_soil=emis_soil
        )
        # print(rf'Alpha: {alpha_PT}, Rn_S0: {Rn_S0}, H_S: {H_S}')
        # print(Rn_V, Rn_S)
        LE_S = Rn_S - G - H_S
        LE_V = Rn_V - H_V
        print(rf'Alpha: {alpha_PT}, LAI: {LAI}, fv: {fv}, Sdn: {Sdn}, Rn_S: {Rn_S}, G: {G}, H_S: {H_S}, LE_S: {LE_S}')
        loop_con = LE_S < 0


    return alpha_PT


# Independent base inputs only
problem = {
    "num_vars": 16,
    "names":  [
        "LAI", "fv", "h_V", "row_sep", "x_LAD", "leaf_width", "Trad", "Tair",
        "Sdn", "u", "P_atm", "ea", "sza_degrees", "saa_degrees", "vza_degrees", 'row_azimuth'],
    "bounds": [
        [0.5, 8],  # LAI
        [0.01, 1],  # fv
        [0.5, 5],  # h_V (m)
        [1, 7],  # row_sep (m)
        [0.5, 1.5],  # x_LAD
        [0.005, 0.2],  # leaf_width (NOT DEFINED in screenshot - fill if needed) (m)
        [-0.5, 15],  # Trad = Tair + diff  (Tair 283–313, diff 0–15 → max = 328)
        [283, 313],  # Tair
        [100, 1000],  # Sdn
        [0, 20],  # u
        [990, 1010],  # P_atm
        [0, 80],  # ea
        [5, 80],  # sza_degrees
        [5, 80], # vza_degrees
        [0, 180], # saa_degrees
        [0, 179] # row_azimuth
        ]
}

from SALib.sample import saltelli
from SALib.analyze import sobol

# 2) Saltelli sampling
N = 10  # or 2000, 5000… (trade-off cost vs accuracy)
second_order = False

X = saltelli.sample(problem, N, calc_second_order=second_order)

# 3) Evaluate the model for each row of X
def model_from_iso(row):
    LAI, fv, h_V, w_V, x_LAD, leaf_width, Trad, Tair, Sdn, u, P_atm, ea, sza_degrees, saa_degrees, vza_degrees, row_azimuth= row
    return sensitivity_analysis_myTSEB(LAI, fv, h_V, w_V, x_LAD, leaf_width,
                                       Trad, Tair, Sdn, u, P_atm, ea, sza_degrees, saa_degrees, vza_degrees, row_azimuth)

Y = np.array([model_from_iso(row) for row in X], dtype=float)

con = Y==0
X_bad = X[con]
test = pd.DataFrame(X_bad)
test.columns = problem["names"]
test.loc[:, 'values'] = Y[con]

# Y = np.clip(Y , 0, 1000)
# print("Y shape:", Y.shape)
# print("min Y:", np.min(Y))
# print("max Y:", np.max(Y))
# print("mean Y:", np.mean(Y))
# print("var Y:", np.var(Y))

# quantiles to see if there are huge outliers
# for q in [0, 1, 5, 50, 95, 99, 100]:
#     print(f"{q}th pct:", np.percentile(Y, q))

# look at the largest absolute values and their inputs
# idx = np.argsort(np.abs(Y))[-100:]
# print("Extreme Y values:", Y[idx])
# print("Inputs for extreme Y:")
# print(X[idx])

# bad = ~np.isfinite(Y)   # True where Y is nan or ±inf
# print("Bad samples:", bad.sum(), "out of", Y.size)
#
# Inspect a few problematic input combinations:
# print("Example bad rows:")
# print(X[bad][:10])   # first 10 bad rows
# print("Corresponding Y:", Y[bad][:10])

#
# 4) Sanity checks
assert Y.ndim == 1, f"Y must be 1-D; got {Y.shape}"
assert Y.shape[0] == X.shape[0], f"len(Y)={Y.shape[0]} vs nsamples={X.shape[0]}"

# 5) Sobol analysis
Si = sobol.analyze(problem, Y, calc_second_order=second_order, print_to_console=False)

si_df = pd.DataFrame(Si)
si_df.loc[:, 'params'] = problem["names"]

for n, s1, st, c1, ct in zip(problem["names"], Si["S1"], Si["ST"], Si["S1_conf"], Si["ST_conf"]):
    print(f"{n:>12s} | S1={s1:.3f} (±{c1:.3f})  ST={st:.3f} (±{ct:.3f})")
# params.update({'Rn_V': Rn_S, 'Rn_S': Rn_S})

# Eliminate values without sense Rn_V < 0
# idx = np.where(Rn_V>0)
# Rn_V = Rn_V[idx]
# Rn_S = Rn_S[idx]
# dict_keys = list(params.keys())
# params_values = [params[x][idx] for x in dict_keys]
# params = dict(zip(dict_keys, params_values))
