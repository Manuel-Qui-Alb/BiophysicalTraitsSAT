import pandas as pd
import numpy as np
import glob
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve, KFold


"""
# Train RTMI
params_prosail = (pd.read_csv('C:/Users\mqalborn\Desktop\BiophysicalTraits/results/prosail.csv')
                  .drop(['Unnamed: 0'], axis=1))
rho_prosail = (pd.read_csv(rf'C:/Users\mqalborn\Desktop\BiophysicalTraits/results/rho_canopy_vec.csv')
               .drop('Unnamed: 0', axis=1))


X = rho_prosail[['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']]
y = params_prosail[['LAI']]


X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42
)

best_params_cv = {
    'bootstrap': False,
    'ccp_alpha': 1.5575316820286568e-05,
    'max_depth': 16,
    'max_features': 'sqrt',
    'max_samples': None,
    'min_samples_leaf': 12,
    'min_samples_split': 13, 'n_estimators': 1156
}
RANDOM_STATE = 42
N_JOBS = -1

# Option A: use the CV-best directly on the validation set
rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS, **best_params_cv)

rf.fit(X_train, y_train)
"""

# Read raster images
path_dir = rf'C:\Users\mqalborn\Desktop\ET_3SEB\PISTACHIO\Sentinel2'
paths = glob.glob(f"{path_dir}/*")
path = paths[0]
import rasterio

with rasterio.open(path) as src:
    array = src.read()
    profile = src.profile
    # profile['count'] = 1
    bands = {}
