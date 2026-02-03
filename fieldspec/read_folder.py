import numpy as np
from fiona.features import length
from specdal import Collection
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import glob
import pandas as pd


folder = os.path.normpath(rf'C:\Users\mqalborn\Box\DATA_CUBBIES\Manuel\BLS\FieldSpec\raw/280126')
files = glob.glob(f"{folder}/*.asd")

c = Collection(name='fieldspec')
c.read(folder, ext=[".asd"], recursive=False)

plt.figure()

docker = []
for x in range(0, len(c.spectra)):
    filename = os.path.basename(files[x])
    sample = filename.split('.')[0].split('#')[2]
    date = files[x].split('\\')[-2]

    sp0 = c.spectra[x]
    wl = sp0.measurement.index
    rho = sp0.measurement.values

    data_pd = pd.DataFrame({'wl': wl, 'rho': rho})
    data_pd.loc[:, 'sample'] = sample
    data_pd.loc[:, 'date'] = date

    docker.append(data_pd)

docker = pd.concat(docker)
docker = docker[['wl', 'rho', 'sample', 'date']]
path_output = rf'C:\Users\mqalborn\Box\DATA_CUBBIES\Manuel\BLS\FieldSpec/processed/rho_280126.csv'
docker.to_csv(path_output)
# sns.lineplot(docker, x='wl', y='rho', hue='sample', legend=None)
# plt.ylim([0, 1])
# plt.legend(None)
# plt.show()

    # lw_1300 = wl <= 1300
    # gt_1500= wl >= 1500
    #
    # lw_1800 = wl <= 1800
    # gt_1950 = wl >= 2000
    #
    # lw_2400 = wl <= 2400


    # wl_plot = np.concatenate([wl[lw_1300],
    #                           [np.nan],
    #                           wl[(gt_1500) & (lw_1800)],
    #                           [np.nan],
    #                           wl[(gt_1950)  & (lw_2400)]])
    # rho_plot = np.concatenate([rho[lw_1300],
    #                            [np.nan],
    #                            rho[(gt_1500) & (lw_1800)],
    #                            [np.nan],
    #                            rho[(gt_1950) & (lw_2400)]])
    #
    #
    # plt.plot(wl_plot, rho_plot, alpha=0.8)

# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Reflectance (%)")
# plt.title("ASD spectra")
# plt.ylim([0, 1])
# # plt.xlim([2400, 2600])
#
# plt.grid(True, alpha=0.3)
# plt.show()