import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve, KFold


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
best_params_cv = {
    'bootstrap': True,
    'ccp_alpha': 1.5575316820286568e-05,
    'max_depth': 12, #16,
    'max_features': 0.2, #'sqrt',
    'max_samples': 0.6,
    'min_samples_leaf': 20, #12,
    'min_samples_split': 20, # 13,
    'n_estimators': 1156
}


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
rf_val = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS, **best_params_cv)

# === Choose the data to analyze ===
# Use train+val to understand how performance scales before touching test:
X_lc = np.vstack([X_train, X_val])
y_lc = np.concatenate([y_train, y_val])


# === Learning curve ===
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

train_sizes, train_scores, val_scores, fit_times, _ = learning_curve(
    rf_val,
    X_lc, y_lc,
    train_sizes=np.linspace(0.1, 1.0, 8),
    cv=cv,
    scoring=rmse_scorer,          # returns negative (because greater_is_better=False)
    return_times=True,
    n_jobs=N_JOBS,
    shuffle=True,
    random_state=RANDOM_STATE
)

# Convert from negative RMSE to positive RMSE
train_rmse = -train_scores
val_rmse = -val_scores

train_mean = train_rmse.mean(axis=1)
train_std  = train_rmse.std(axis=1)
val_mean   = val_rmse.mean(axis=1)
val_std    = val_rmse.std(axis=1)

# === Plot ===
plt.figure(figsize=(7,5))
plt.plot(train_sizes, train_mean, marker="o", label="Train RMSE")
plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.2)
plt.plot(train_sizes, val_mean, marker="s", label="CV RMSE")
plt.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.2)
plt.xlabel("Training samples")
plt.ylabel("RMSE")
plt.title("Learning Curve — RandomForestRegressor")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.ylim([0, 0.2])
plt.show()


# (Optional) Fit-time curve to gauge scaling
# plt.figure(figsize=(7,5))
# plt.plot(train_sizes, fit_times.mean(axis=1), marker="o")
# plt.xlabel("Training samples")
# plt.ylabel("Fit time (s)")
# plt.title("Fit Time vs Training Size")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#







"""
# Predictions on training set
y_train_pred = rf_val.predict(X_train)

# Training metrics
train_rmse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f"Training RMSE: {train_rmse:.3f}")
print(f"Training R²: {train_r2:.3f}")


val_rmse = mean_squared_error(y_val, rf_val.predict(X_val))
val_r2 = r2_score(y_val, rf_val.predict(X_val))
print(f"\nValidation RMSE (CV-best): {val_rmse:.4f}, R²: {val_r2:.4f}")

y_test_pred = rf_val.predict(X_test)
test_rmse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n=== Final Test Performance ===")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test R²:   {test_r2:.4f}")
"""