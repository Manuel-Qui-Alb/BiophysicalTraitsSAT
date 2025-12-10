import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data_pytseb = pd.read_csv('files/pyTSEB_outputs.csv').rename(columns={'Unnamed: 0': 'index'})
data_mytseb = pd.read_csv('files/myTSEB_outputs.csv').rename(columns={'Unnamed: 0': 'index'})

data = pd.merge(data_pytseb, data_mytseb)

sns.scatterplot(data, x='LE_V_pyTSEB', y='LE_V_myTSEB')
# plt.ylim([200, 500])
# plt.xlim([200, 500])
plt.show()


print(data_pytseb.head())
print(data_mytseb.head())
print(data.head())
