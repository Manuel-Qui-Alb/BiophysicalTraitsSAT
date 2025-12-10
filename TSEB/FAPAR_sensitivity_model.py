import numpy as np
import pandas as pd

import functions as myTSEB
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np


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

params['saa_rad'] = np.radians(np.random.uniform(0, 180, 10000))
params['row_azimuth'] = np.radians(np.random.uniform(0, 180, 10000))

########################################################################################################################
# f_theta = fraction of incident beam radiation intercepted by the plant
# f_theta correspond to the radiation available for scattering, transpiration, and photosynthesis.
########################################################################################################################
# omega0 = myTSEB.nadir_clumpling_index_Kustas_Norman(
#     params['LAI'],
#     params['fv'],
#     params['x_LAD'],
#     params['sza_rad']
# )
#
# omega_iso = myTSEB.off_nadir_clumpling_index_Kustas_Norman(
#     LAI=params['LAI'],
#     fv=params['fv'],
#     h_V=params['h_V'],
#     w_V=params['w_V'],
#     x_LAD=params['x_LAD'],
#     sza=params['sza_rad']
# )
#
# omega_row = myTSEB.rectangular_row_clumping_index_parry(
#     LAI=params['LAI'],
#     fv0=params['fv'],
#     w_V=params['w_V'],
#     h_V=params['h_V'],
#     sza=params['sza_rad'],
#     saa=params['saa_rad'],
#     row_azimuth=params['row_azimuth'],
#     hb_V=0,
#     L=None,
#     x_LAD=params['x_LAD']
# )
#
# params.update({'omega_iso': omega_iso, 'omega_row': omega_row})
#
#
# f_theta = myTSEB.estimate_f_theta(
#     LAI=params['LAI'],
#     x_LAD=params['x_LAD'],
#     omega=params['omega'],
#     sza=params['sza_rad']
# )

#
#

def sensitivity_model_iso(LAI, fv, w_V, h_V, sza, x_LAD):
    omega_iso = myTSEB.off_nadir_clumpling_index_Kustas_Norman(
        LAI=LAI,
        fv=fv,
        h_V=h_V,
        w_V=w_V,
        x_LAD=x_LAD,
        sza=sza
    )

    f_theta = myTSEB.estimate_f_theta(
        LAI=LAI,
        x_LAD=x_LAD,
        omega=omega_iso,
        sza=sza
    )

    return f_theta


def sensitivity_model_row(LAI, fv, w_V, h_V, sza, saa, row_azimuth, x_LAD):
    omega_row = myTSEB.rectangular_row_clumping_index_parry(
        LAI=LAI,
        fv0=fv,
        w_V=w_V,
        h_V=h_V,
        sza=sza,
        saa=saa,
        row_azimuth=row_azimuth,
        hb_V=0,
        L=None,
        x_LAD=x_LAD
    )

    f_theta = myTSEB.estimate_f_theta(
        LAI=LAI,
        x_LAD=x_LAD,
        omega=omega_row,
        sza=sza
    )

    return f_theta

# Independent base inputs only
problem = {
    "num_vars": 6,
    "names": ["LAI", "fv", "w_V", "h_V", "sza", 'x_LAD'],
    "bounds": [
        [np.min(params['LAI']), np.max(params['LAI'])],   # LAI
        [np.min(params['fv']), np.max(params['fv'])],   # FracCover
        [np.min(params['w_V']), np.max(params['w_V'])],     # w_V
        [np.min(params['h_V']), np.max(params['h_V'])],     # h_V
        [np.min(params['sza_rad']), np.max(params['sza_rad'])],     # SolarAngle (deg)
        [np.min(params['x_LAD']), np.max(params['x_LAD'])],   # X_LAD
    ],
}

# 2) Saltelli sampling
N = 10000   # or 2000, 5000… (trade-off cost vs accuracy)
second_order = False

X = saltelli.sample(problem, N, calc_second_order=second_order)   # shape (nsamples, 8)

# 3) Evaluate the model for each row of X
def model_from_iso(row):
    LAI, fv, w_V, h_V, sza, x_LAD = row
    return sensitivity_model_iso(LAI, fv, w_V, h_V, sza, x_LAD)

Y = np.array([model_from_iso(row) for row in X], dtype=float)

# 4) Sanity checks
assert Y.ndim == 1, f"Y must be 1-D; got {Y.shape}"
assert Y.shape[0] == X.shape[0], f"len(Y)={Y.shape[0]} vs nsamples={X.shape[0]}"

# 5) Sobol analysis
Si = sobol.analyze(problem, Y, calc_second_order=second_order, print_to_console=False)

si_df = pd.DataFrame(Si)
si_df.loc[:, 'params'] = problem["names"]

for n, s1, st, c1, ct in zip(problem["names"], Si["S1"], Si["ST"], Si["S1_conf"], Si["ST_conf"]):
    print(f"{n:>12s} | S1={s1:.3f} (±{c1:.3f})  ST={st:.3f} (±{ct:.3f})")

si_df.to_csv('files/FAPAR_iso_sensitivity_analysis.csv', index=False)
print(si_df)

# Independent base inputs only
problem = {
    "num_vars": 8,
    "names": ["LAI", "fv", "w_V", "h_V", "sza", "saa", "row_azimuth", 'x_LAD'],
    "bounds": [
        [np.min(params['LAI']), np.max(params['LAI'])],   # LAI
        [np.min(params['fv']), np.max(params['fv'])],   # FracCover
        [np.min(params['w_V']), np.max(params['w_V'])],     # w_V
        [np.min(params['h_V']), np.max(params['h_V'])],     # h_V
        [np.min(params['sza_rad']), np.max(params['sza_rad'])],     # SolarAngle (deg)
        [np.min(params['saa_rad']), np.max(params['saa_rad'])],     # SolarAngle (deg)
        [np.min(params['row_azimuth']), np.max(params['row_azimuth'])],     # SolarAngle (deg)
        [np.min(params['x_LAD']), np.max(params['x_LAD'])],   # X_LAD
    ],
}

# 2) Saltelli sampling
N = 10000   # or 2000, 5000… (trade-off cost vs accuracy)
second_order = False

X = saltelli.sample(problem, N, calc_second_order=second_order)   # shape (nsamples, 8)

# 3) Evaluate the model for each row of X
def model_from_row(row):
    LAI, fv, w_V, h_V, SZA, SAA, row_azimuth, x_LAD = row
    return sensitivity_model_row(LAI, fv, w_V, h_V, SZA, SAA, row_azimuth, x_LAD)

Y = np.array([model_from_row(row) for row in X], dtype=float)

# 4) Sanity checks
assert Y.ndim == 1, f"Y must be 1-D; got {Y.shape}"
assert Y.shape[0] == X.shape[0], f"len(Y)={Y.shape[0]} vs nsamples={X.shape[0]}"

# 5) Sobol analysis
Si = sobol.analyze(problem, Y, calc_second_order=second_order, print_to_console=False)

si_df = pd.DataFrame(Si)
si_df.loc[:, 'params'] = problem["names"]

for n, s1, st, c1, ct in zip(problem["names"], Si["S1"], Si["ST"], Si["S1_conf"], Si["ST_conf"]):
    print(f"{n:>12s} | S1={s1:.3f} (±{c1:.3f})  ST={st:.3f} (±{ct:.3f})")

si_df.to_csv('files/FAPAR_row_sensitivity_analysis.csv', index=False)
print(si_df)