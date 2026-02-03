import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import GeneralFunctions as GF


N = 1000
lai = np.random.uniform(0, 7, N)
fv_var = np.random.uniform(0.7, 1.1, N)

# lai = np.arange(0, 7, 0.01)
K_be = GF.estimate_Kbe(x_LAD=1, sza=0)
fv_base = 1 - np.exp(-K_be * lai)
fv0 = np.clip(fv_base * fv_var, 1e-6, 1)

sns.scatterplot(x=lai, y=fv0)
sns.lineplot(x=lai, y=fv_base, color='r')
plt.ylabel('Fractional Vegetation Cover (fv)')
plt.xlabel('Leaf Area Index (LAI)')
plt.show()

row_sep = np.random.uniform(2, 8, N)
w_V = fv0 * row_sep

sns.scatterplot(x=fv0, y=w_V, hue=row_sep)
# sns.lineplot(x=lai, y=fv_base, color='r')
plt.ylabel('Canopy Width (w_V, m)')
# plt.xlabel('Row Separation (m)')
plt.xlabel('Fractional Vegetation Cover (fv)')
plt.show()

plant_sep_var = np.random.uniform(0.5, 1, N)
sp = row_sep * plant_sep_var
sns.scatterplot(x=row_sep, y=sp)
# sns.lineplot(x=lai, y=fv_base, color='r')
plt.ylabel('Plant Separation (m)')
# plt.xlabel('Row Separation (m)')
plt.xlabel('Row Separation (m)')
plt.show()
