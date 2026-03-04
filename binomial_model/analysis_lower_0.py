import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text



df = pd.read_csv("files/SA_Binomial.csv")

df.loc[:, 'Rns'] = df.Rc_binomial + df.Rs_binomial
con = df.Rns > df.St
df.loc[con, 'flag'] = 1
df.loc[~con, 'flag'] = 0

inputs = ['lai', 'fv_var', 'h_V', 'row_sep',
          # 'plant_sep_var',
          'doy', 'hour',
          'row_radians', 'abs_vis_leaf', 'abs_nir_leaf', 'rho_vis_soil',
          'rho_nir_soil']
X = df[inputs].values
y = df["flag"].values  # 1 = failure

clf = DecisionTreeClassifier(max_depth=4, random_state=0)
clf.fit(X, y)

print(export_text(clf, feature_names=inputs))

print(df[con].shape)