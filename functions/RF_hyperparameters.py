import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy.stats import randint, uniform

params_prosail = (pd.read_csv(rf'C:\Users\mqalborn\Desktop\BiophysicalTraits\results/prosail.csv')
                  .drop(['Unnamed: 0'], axis=1))
rho_prosail = (pd.read_csv(rf'C:\Users\mqalborn\Desktop\BiophysicalTraits\results/rho_canopy_vec.csv')
               .drop('Unnamed: 0', axis=1))


X = rho_prosail[['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']]
y = params_prosail[['LAI']]


X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


# ------------------------
# Hyperparameter search on training split
# ------------------------
RANDOM_STATE = 42
N_JOBS = -1

rf_base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS)

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

param_distributions = {
    "n_estimators": randint(300, 1201),          # 300–1200 trees
    "max_depth": [None, 8, 12, 16, 24, 32],      # try None + mid depths
    "min_samples_split": randint(2, 21),         # 2–20
    "min_samples_leaf": randint(1, 21),          # 1–20
    "max_features": ["sqrt", "log2", uniform(0.1, 0.7)],  # float means fraction of features
    "bootstrap": [True, False],
    "ccp_alpha": uniform(0.0, 0.02),             # light cost-complexity pruning
    "max_samples": [None] + list(np.linspace(0.5, 1.0, 6))  # only used if bootstrap=True
}

search = RandomizedSearchCV(
    rf_base,
    param_distributions=param_distributions,
    n_iter=75,                          # increase for a more exhaustive search
    scoring="neg_root_mean_squared_error",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    verbose=1,
    refit=False                         # we'll choose by val set, then refit manually
)

search.fit(X_train, y_train)

print("Top 5 configs by CV score (lower RMSE is better):")
results = (
    np.argsort(-search.cv_results_["mean_test_score"])[:5]
)
for rank_idx in results:
    rmse = -search.cv_results_["mean_test_score"][rank_idx]
    params = search.cv_results_["params"][rank_idx]
    print(f"RMSE={rmse:.4f} | params={params}")

best_params_cv = search.best_params_
print("\nBest by CV (for reference):")
print(best_params_cv)




"""
# ------------------------
# Build simple Random Forest
# ------------------------

rf = RandomForestRegressor(
    n_estimators=200,        # number of trees
    max_depth=None,          # let trees expand fully
    random_state=42,
    n_jobs=-1
)

# Fit on training data
rf.fit(X_train, y_train)

# ------------------------
# Validation performance
# ------------------------
y_val_pred = rf.predict(X_val)
val_rmse = mean_squared_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)

print(f"Validation RMSE: {val_rmse:.3f}")
print(f"Validation R²: {val_r2:.3f}")

# ------------------------
# Final test evaluation
# ------------------------
y_test_pred = rf.predict(X_test)
test_rmse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Test RMSE: {test_rmse:.3f}")
print(f"Test R²: {test_r2:.3f}")

"""