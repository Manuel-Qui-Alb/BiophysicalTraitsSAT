import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import functions as myTSEB
# data_cn = pd.read_csv('files/TSEB_CN.csv').rename(columns={'Unnamed: 0': 'index'})
# data_bn = pd.read_csv('files/TSEB_BINOMIAL.csv').rename(columns={'Unnamed: 0': 'index'})
# pd.merge()
# print(data_cn.columns)
# print(data_bn.columns)

# sns.regplot(data=data_cn, x='fv_var', y='Sn_V_CN', order=2, line_kws=dict(color="r"))
# plt.show()

LAI = random_array_range = np.random.uniform(low=0, high=7, size=1000)
k_omega = random_array_range = np.random.uniform(low=0.4, high=1, size=1000)
x_LAD = random_array_range = np.random.uniform(low=0.5, high=1.5, size=1000)
K_be = myTSEB.estimate_Kbe(x_LAD, 0)
fv = 1 - np.exp(-K_be * k_omega * LAI)

plt.scatter(LAI, fv)
plt.show()
# print(fv)
