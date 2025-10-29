import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

path = rf'C:\Users\mqalborn\Desktop\ET_3SEB\results/LAI_TRAD_L08_LAI_MODIS_RIP720.csv'

data = pd.read_csv(path)
data.loc[:, 'date'] = pd.to_datetime(data['id'].str.split('_').str[-1])
# print(data.head())

data = data[data.LAI_count >= 25]
sns.set_palette("viridis")
sns.lineplot(data, x='date', y='TRAD', hue='block', marker='o', markersize=10)
plt.ylim(300, 320)
plt.show()