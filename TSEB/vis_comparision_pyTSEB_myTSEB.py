import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

# N = 10000
df_myTSEB = pd.read_csv(rf'files/TSEB_CN.csv' )
df_pyTSEB = pd.read_csv(rf'files/pyTSEB_outputs.csv').drop('Unnamed: 0', axis=1)
# print(df_myTSEB.columns)
# print(df_pyTSEB.columns)
# nan_values = np.isnan(df_pyTSEB['LE_V_pyTSEB'])
df_myTSEB = df_myTSEB[['omega_CN', 'f_theta_CN', 'Trad_V_CN', 'Trad_S_CN', 'Ln_V_CN', 'Sn_V_CN', 'Rn_V_CN', 'LE_V_CN']]
df_pyTSEB = df_pyTSEB[['omega_pyTSEB', 'f_theta_pyTSEB', 'Trad_V_pyTSEB', 'Trad_S_pyTSEB', 'Ln_V_pyTSEB','SN_V_pyTSEB',
                       'Rn_V_pyTSEB', 'LE_V_pyTSEB']]
#


# nan_values = np.isnan(df_pyTSEB['LE_V_pyTSEB'])
# print(np.any(nan_values))
x = df_pyTSEB['omega_pyTSEB']
y = df_myTSEB['omega_CN']

# xy = np.vstack([x,y])
# z = gaussian_kde(xy)(xy)
plt.scatter(x, y,
           # c=z,
           s=10)
plt.plot([0, 1], [0, 1], c='grey')
# plt.set_ylim([0, 1])
# plt.set_xlim([0, 1])
plt.ylabel(r'omega_pyTSEB')
plt.ylabel(r'omega_myTSEB')

# plt.savefig('files/f_theta_myTSEB_pyTSEB.png', dpi=300)
plt.show()

x = df_pyTSEB['f_theta_pyTSEB']
y = df_myTSEB['f_theta_CN']
# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

plt.scatter(x, y,
           c=z,
           s=10)
# plt.plot([0, 1000], [0, 1000], c='grey')
# plt.ylim([0, 1000])
# plt.xlim([0, 1000])
plt.xlabel(r'$f(\theta)$ pyTSEB')
plt.ylabel(r'$f(\theta)$ myTSEB')
plt.savefig('files/f_theta_myTSEB_pyTSEB.png', dpi=300)
plt.show()

x = df_pyTSEB['SN_V_pyTSEB']
y = df_myTSEB['Sn_V_CN']
# Calculate the point density
# xy = np.vstack([x,y])
# z = gaussian_kde(xy)(xy)

plt.scatter(x, y,
           # c=z,
           s=10)
plt.plot([0, 1000], [0, 1000], c='grey')
plt.ylim([0, 1000])
plt.xlim([0, 1000])
plt.xlabel(r'Sn_V_pyTSEB')
plt.ylabel(r'Sn_V_myTSEB')
plt.savefig('files/Sn_V_myTSEB_pyTSEB.png', dpi=300)
plt.show()

x = df_pyTSEB['Ln_V_pyTSEB']
y = df_myTSEB['Ln_V_CN']
# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

plt.scatter(x, y,
           c=z,
           s=10)
plt.plot([-600, 300], [-600, 300], c='grey')
plt.ylim([-600, 300])
plt.xlim([-600, 300])
plt.xlabel(r'Ln_V_pyTSEB')
plt.ylabel(r'Ln_V_myTSEB')
plt.savefig('files/Ln_V_myTSEB_pyTSEB.png', dpi=300)
plt.show()


x = df_pyTSEB['Rn_V_pyTSEB']
y = df_myTSEB['Rn_V_CN']
# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

plt.scatter(x, y,
           c=z,
           s=10)
# plt.plot([0, 1000], [0, 1000], c='grey')
# plt.ylim([0, 1000])
# plt.xlim([0, 1000])
plt.xscale('log')
plt.xlabel(r'Rn_V_pyTSEB')
plt.ylabel(r'Rn_V_myTSEB')
plt.savefig('files/Rn_V_myTSEB_pyTSEB.png', dpi=300)
plt.show()

x = df_pyTSEB['Rn_V_pyTSEB']
y = df_myTSEB['Rn_V_CN']
# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

plt.scatter(x, y,
           c=z,
           s=10)
plt.plot([0, 1000], [0, 1000], c='grey')
plt.ylim([0, 1000])
plt.xlim([0, 1000])
plt.xlabel(r'Rn_V_pyTSEB')
plt.ylabel(r'Rn_V_myTSEB')
plt.savefig('files/Rn_V_myTSEB_pyTSEB2.png', dpi=300)
plt.show()

x = df_pyTSEB['Trad_V_pyTSEB']
y = df_myTSEB['Trad_V_CN']
# Calculate the point density
xy = np.vstack([x[~np.isnan(x)],y[~np.isnan(x)]])
z = gaussian_kde(xy)(xy)

x = df_pyTSEB['Trad_V_pyTSEB']
y = df_myTSEB['Trad_V_CN']
# Calculate the point density
xy = np.vstack([x[~np.isnan(x)],y[~np.isnan(x)]])
z = gaussian_kde(xy)(xy)

plt.scatter(x[~np.isnan(x)], y[~np.isnan(x)],
           c=z,
           s=10)
# plt.plot([280, 320], [280, 320], c='grey')
plt.xscale('log')
# plt.ylim([280, 320])
# plt.xlim([280, 320])
plt.xlabel(r'Trad_V_pyTSEB')
plt.ylabel(r'Trad_V_myTSEB')
plt.savefig('files/Trad_V_myTSEB_pyTSEB.png', dpi=300)
plt.show()

plt.scatter(x[~np.isnan(x)], y[~np.isnan(x)],
           c=z,
           s=10)
plt.plot([280, 320], [280, 320], c='grey')
# plt.xscale('log')
plt.ylim([280, 320])
plt.xlim([280, 320])
plt.xlabel(r'Trad_V_pyTSEB')
plt.ylabel(r'Trad_V_CN')
plt.savefig('files/Trad_V_myTSEB_pyTSEB2.png', dpi=300)
plt.show()



x = df_pyTSEB['LE_V_pyTSEB']
y = df_myTSEB['LE_V_myTSEB']
# Calculate the point density
xy = np.vstack([x[~np.isnan(x)],y[~np.isnan(x)]])
z = gaussian_kde(xy)(xy)

plt.scatter(x[~np.isnan(x)], y[~np.isnan(x)],
           c=z,
           s=10)
plt.plot([0, 1000], [0, 1000], c='grey')
plt.ylim([0, 1000])
plt.xlim([0, 1000])
plt.xlabel(r'LE_V_pyTSEB')
plt.ylabel(r'LE_V_CN')
plt.savefig('files/LE_V_myTSEB_pyTSEB.png', dpi=300)
plt.show()
#

# x = df_myTSEB['LAI_CNLAI_CN']
# y = df_pyTSEB['LAI_pyTSEB']
# ax = axes[1]
# ax.scatter(x, y,
#            # c=z,
#            s=10)
# ax.plot([0, 1], [0, 1], c='grey')
# ax.set_ylim([0, 1])
# ax.set_xlim([0, 1])
# ax.set_xlabel('myTSEB')
# ax.set_ylabel('pyTSEB')
#
# x = df_myTSEB['fv_CN']
# y = df_pyTSEB['fv_pyTSEB']
# ax = axes[2]
# ax.scatter(x, y,
#            # c=z,
#            s=10)
# ax.plot([0, 1], [0, 1], c='grey')
# ax.set_ylim([0, 1])
# ax.set_xlim([0, 1])
# ax.set_xlabel('myTSEB')
# ax.set_ylabel('pyTSEB')
# plt.show()plt.show()


"""
x = df_myTSEB['f_theta_CN']
y = df_pyTSEB['f_theta_pyTSEB']

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
ax = axes[1]
ax.scatter(x=x, y=y, c=z, s=10)
ax.plot([0, 1], [0, 1], c='grey')
ax.set_ylim([0, 1])
ax.set_xlim([0, 1])

ax.set_xlabel('f_theta_CN')
ax.set_ylabel('f_theta_pyTSEB')

x = df_myTSEB['LE_V_CN']
y = df_pyTSEB['LE_V_pyTSEB']

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
ax = axes[2]
ax.scatter(x=x, y=y, c=z, s=10)
# ax.plot([0, 700], [0, 700], c='grey')
# ax.set_ylim([0, 700])
# ax.set_xlim([0, 700])
ax.set_xlabel('LE_V')
ax.set_ylabel('LE_V_pyTSEB')

plt.show()
# import seaborn as sns
# sns.jointplot(x=x, y=y, kind='kde', fill=True, cmap='Blues')
# plt.show()
# plt.ylim([0, 700])
# plt.xlim([0, 700])
# plt.show()
"""