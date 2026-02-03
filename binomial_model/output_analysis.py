import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

data = pd.read_csv('files/binomial_outputs.csv')

data.loc[:, 'flag'] = 0
con = data.Sn_ov == -9999
data.loc[con, 'flag'] = 1
con = np.isnan(data.Sn_ov)
data.loc[con, 'flag'] = 2

names = ['lai', 'fv0', 'CanopyHeight', 'sr', 'sp', 'sza', 'saa', 'row_azimuth', 'Sdn',
         'P_atm', 'ameanv', 'ameann', 'rsoilv', 'rsoiln']
Xfeat = data[names].values
y = data["flag"].astype(int).values

clf = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=0
)
clf.fit(Xfeat, y)

print(export_text(clf, feature_names=names))


# print(data.head())
