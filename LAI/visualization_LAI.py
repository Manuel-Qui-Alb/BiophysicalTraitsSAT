import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

path = rf'C:\Users\mqalborn\Box\DATA_CUBBIES\Manuel\BLS\Defoliation\scanner\20260121/leaf_area_scanner.csv'
df = pd.read_csv(path)
df_grouped = (df.groupby(['sample'])
      .agg(leaf_area_scanner_cm2=('leaf_area_scanner_cm2', 'sum'))
      .reset_index())

r = 10
area_ref = np.pi * (r ** 2)
df_grouped.loc[:, 'leaf_area_index_scanner'] = df_grouped.leaf_area_scanner_cm2 / area_ref

path = rf'C:\Users\mqalborn\Box\DATA_CUBBIES\Manuel\BLS\Defoliation\scanner\20260121/leaf_area_scanner.xlsx'
with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='leaf_area', index=False)
    df_grouped.to_excel(writer, sheet_name='LAI', index=False)

print(df_grouped.head())