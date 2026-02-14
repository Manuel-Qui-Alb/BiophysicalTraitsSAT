import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde


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

    return mbe, mae, rmse


data = pd.read_excel('files/binomial_CN_sensitivity_analysis.xlsx', sheet_name='inputs')

def graph_comparison(ax, x, y, xlabel="C&N-R", ylabel="BINOMIAL",
                     cmap="viridis", point_size=6, show_colorbar=False):
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
    mbe, mae, rmse = error_metrics(y_model2, y_model1)

    # Scatter
    sc = ax.scatter(y_model1, y_model2, c=z, s=point_size, cmap=cmap)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Limits + 1:1 line
    lims = [
        np.min([y_model1.min(), y_model2.min()]),
        np.max([y_model1.max(), y_model2.max()])
    ]
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
        f"Slope = {m:.2f}"
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
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.subplots_adjust(wspace=0.05)

graph_comparison(axes[0], data['Sn_V_CN'], data['Sn_V_bin'], xlabel="C&N-R", ylabel="BINOMIAL")
graph_comparison(axes[1], data['Sn_S_CN'], data['Sn_S_bin'], xlabel="C&N-R", ylabel="BINOMIAL")

axes[0].set_title("Canopy Shortwave Net Radiation")
axes[1].set_title("Soil Shortwave Net Radiation")
plt.savefig('files/comparision_C&N_BINOMIAL.png', dpi=300)
plt.show()