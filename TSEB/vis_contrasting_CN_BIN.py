import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde

folder = rf'C:\Users\mqalborn\Desktop\BiophysicalTraitsSAT\TSEB\files'
df_CN = pd.read_csv(folder + '/TSEB_CN.csv')
df_BIN = pd.read_csv(folder + '/TSEB_BINOMIAL.csv')

df_CN.rename(columns={'Unnamed: 0':'index', 'LAI':'LAI_CN'}, inplace=True)
df_BIN.rename(columns={'Unnamed: 0':'index', 'LAI':'LAI_BIN'}, inplace=True)

df_CN = df_CN[['index','LAI_CN', 'omega_CN', 'f_theta_CN', 'Sn_V_CN', 'Sn_S_CN', 'Trad_V_CN', 'Trad_S_CN', 'Ln_V_CN',
               'Ln_S_CN','Rn_V_CN', 'Rn_S_CN', 'LE_V_CN', 'LE_S_CN']]
df_CN.loc[:, 'LE_CN'] = df_CN['LE_V_CN'] + df_CN['LE_S_CN']
df_CN.loc[:, 'Sn_CN'] = df_CN['Sn_V_CN'] + df_CN['Sn_S_CN']

df_BIN = df_BIN[['index', 'LAI_BIN', 'omega_BIN', 'f_theta_BIN', 'Sn_V_BIN', 'Sn_S_BIN', 'Trad_V_BIN', 'Trad_S_BIN',
                 'Ln_V_BIN', 'Ln_S_BIN', 'Rn_V_BIN', 'Rn_S_BIN', 'LE_V_BIN', 'LE_S_BIN']]
df_BIN.loc[:, 'LE_BIN'] = df_BIN['LE_V_BIN'] + df_BIN['LE_S_BIN']
df_BIN.loc[:, 'Sn_BIN'] = df_BIN['Sn_V_BIN'] + df_BIN['Sn_S_BIN']
df = pd.merge(df_CN, df_BIN, on='index', how='left')

# sns.scatterplot(data=df, x='LAI_BIN', y='LAI_CN')
# plt.show()


def error_metrics(y_model1, y_model2):
    y1 = np.asarray(y_model1, dtype=float)
    y2 = np.asarray(y_model2, dtype=float)

    mask = np.isfinite(y1) & np.isfinite(y2)
    y1 = y1[mask]
    y2 = y2[mask]

    diff = y1 - y2
    mbe = np.mean(diff)
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))

    # Linear regression for R²
    m, b = np.polyfit(y1, y2, 1)
    y_pred = m * y1 + b
    ss_res = np.sum((y2 - y_pred) ** 2)
    ss_tot = np.sum((y2 - np.mean(y2)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    return mbe, mae, rmse, r2

def graph_comparison(ax, x, y, xlabel="C&N-R", ylabel="BINOMIAL",
                     cmap="viridis", point_size=6, show_colorbar=False, ln=False):
    """
    Scatter comparison with density coloring, 1:1 line, regression fit, and stats box.
    Draws on the provided matplotlib axis (ax), so it works in subplots.
    """
    y_model1 = np.asarray(x, dtype=float)
    y_model2 = np.asarray(y, dtype=float)

    # Remove NaNs/Infs consistently
    mask = np.isfinite(y_model1) & np.isfinite(y_model2)
    y_model1 = y_model1[mask]
    y_model2 = y_model2[mask]

    # Density (guard against very small sample sizes)
    if y_model1.size >= 3:
        xy = np.vstack([y_model1, y_model2])
        z = gaussian_kde(xy)(xy)
    else:
        z = np.ones_like(y_model1)

    # Metrics (your function; assumes it exists)
    mbe, mae, rmse, r2 = error_metrics(y_model2, y_model1)

    # Scatter
    sc = ax.scatter(y_model1, y_model2, c=z, s=point_size, cmap=cmap)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # lims = [
    #     np.min([y_model1.min(), y_model2.min()]),
    #     np.max([y_model1.max(), y_model2.max()])
    # ]

    lims = [
        np.min([0, 0]),
        np.max([y_model1.max(), y_model2.max()])
    ]

    # if ln:
    #
    #
    # else:
    #     lims = [
    #         np.max([np.min([y_model1.min(), y_model2.min()]), 0]),
    #         np.max([y_model1.max(), y_model2.max()])
    #     ]

    # Limits + 1:1 line


    ax.plot(lims, lims, "k--", lw=1, label="1:1")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")

    # Regression line
    m, b = np.polyfit(y_model1, y_model2, 1)
    ax.plot(lims, m*np.array(lims) + b, color="red", lw=1, label="Fit")
    ax.legend(frameon=False, loc="lower right")

    # Stats box
    stats_text = (
        f"N = {len(y_model1)}\n"
        f"MBE = {mbe:.1f}\n"
        f"MAE = {mae:.1f}\n"
        f"RMSE = {rmse:.1f}\n"
        f"Slope = {m:.2f}\n"
        f"$R^2$ = {r2:.3f}"
    )

    ax.text(
        0.05, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="square", facecolor="white", alpha=0.8)
    )

    # Optional colorbar
    if show_colorbar:
        plt.colorbar(sc, ax=ax, label="Point density")

    return sc  # useful if you want one shared colorbar


# ---- Example: 2 columns subplot ----
fig, axes = plt.subplots(2, 3,
                         figsize=(18, 11)
                         )
fig.subplots_adjust(wspace=0.05)

graph_comparison(axes[0, 0], df['Sn_V_CN'], df['Sn_V_BIN'], xlabel="C&N-R", ylabel="BINOMIAL")
graph_comparison(axes[0, 1], df['Sn_S_CN'], df['Sn_S_BIN'], xlabel="C&N-R", ylabel="BINOMIAL")
graph_comparison(axes[0, 2], df['Sn_CN'], df['Sn_BIN'], xlabel="C&N-R", ylabel="BINOMIAL")

graph_comparison(axes[1, 0], df['LE_V_CN'], df['LE_V_BIN'], xlabel="C&N-R", ylabel="BINOMIAL")
graph_comparison(axes[1, 1], df['LE_S_CN'], df['LE_S_BIN'], xlabel="C&N-R", ylabel="BINOMIAL")
graph_comparison(axes[1, 2], df['LE_CN'], df['LE_BIN'], xlabel="C&N-R", ylabel="BINOMIAL")
#
# graph_comparison(axes[0, 0], df['Sn_V_CN'], df['Sn_V_BIN'], xlabel="C&N-R", ylabel="BINOMIAL")
# graph_comparison(axes[0, 1], df['f_theta_CN'], df['f_theta_BIN'], xlabel="C&N-R", ylabel="BINOMIAL")
# graph_comparison(axes[0, 2], df['Trad_V_CN'], df['Trad_V_BIN'], xlabel="C&N-R", ylabel="BINOMIAL")
#
# graph_comparison(axes[1, 0], df['Ln_V_CN'], df['Ln_V_BIN'], xlabel="C&N-R", ylabel="BINOMIAL", ln=True)
# graph_comparison(axes[1, 1], df['Rn_V_CN'], df['Rn_V_BIN'], xlabel="C&N-R", ylabel="BINOMIAL")
# graph_comparison(axes[1, 2], df['LE_V_CN'], df['LE_V_BIN'], xlabel="C&N-R", ylabel="BINOMIAL")
#
# axes[0, 0].set_title("Canopy Shortwave Net Radiation")
# axes[0, 1].set_title("f theta")
# axes[0, 2].set_title("Canopy Temperature (K)")
#
# axes[1, 0].set_title("Canopy Longwave Net Radiation")
# axes[1, 1].set_title("Canopy Longwave Net Radiation")
# axes[1, 2].set_title("Canopy Latent Heat Flux")
#
# axes[0].set_title("Canopy Shortwave Net Radiation")
# axes[1].set_title("Canopy Net Radiation")
# axes[2].set_title("Canopy Latent Heat Flux")

# plt.savefig('files/comparison_TSEB_C&N_BINOMIAL.png', dpi=300)
plt.show()