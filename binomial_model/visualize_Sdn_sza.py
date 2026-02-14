import numpy as np
from matplotlib import pyplot as plt
import pyTSEB.TSEB as TSEB

"""
Vocabulary.

Partitions the incoming solar radiation into PAR and non-PR and 
diffuse and direct beam component of the solar spectrum.

Parameters
----------
    sza : float
        Solar Zenith Angle (degrees).
    Wv : float, optional
        Total column precipitable water vapour (g cm-2), default 1 g cm-2.
    press : float, optional
        atmospheric pressure (mb), default at sea level (1013mb).
    m : 
        optical air mass number
    tau_atms:
        Atmospheric Transmittance. 
        t values between 0.7 for typical clear sky conditions. 0.75 for the clearest day.
        Values lower than 0.4 should be considered as a overcast sky. 
    Sp : Float 
        Direct irradiance on a surface perpendicular to the beam
    Sp0 : float
        Extraterrestrial flus density (1368 - 1468)
    Sb :   
        Beam irradiance on a horizontal surface 
    Sd : 
        Diffuse irrandianc
References
----------
"""
N = 100
rng = np.random.default_rng()
sza_degrees = rng.uniform(low=0, high=90, size=N)
sza_radians = np.deg2rad(sza_degrees)

# Campbell and Norman 1998. Page 172.
m = 101.3 / (101.3 * np.cos(sza_radians))
tau_atms = 0.6  # atmospheric transmittance
Sp0 = 1368 # Extraterrestrial flux density
Sp = 1368 * (tau_atms ** m)
Sb = Sp * np.cos(sza_radians)

Sd = 0.3 * (1 - tau_atms ** m) * Sp0 * np.cos(sza_radians)
# f_diff = np.clip(0.15 + 0.6 * (1 - mu), 0.15, 0.7)
St = Sb + Sd

difvis, difnir, fvis, fnir = TSEB.rad.calc_difuse_ratio(
    S_dn=St,
    sza=sza_degrees,
    press=np.full_like(St, 1013)
)
print(difvis)
skyl = fvis * difvis + fnir * difnir
Srad_dir = (1. - skyl) * St
Srad_diff = skyl * St


plt.plot(sza_degrees, St, 'o')
plt.plot(sza_degrees, Srad_dir, 'o')
plt.plot(sza_degrees, Srad_diff, 'o')
plt.show()


# sza_degrees = np.arange(15, 65, 1)
# sza_radians = np.deg2rad(sza_degrees)
#
# sza_degrees = np.degrees(sza_radians)
#
# mu = np.cos(sza_radians)
# mu = np.clip(mu, 0.05, 1.0)  # avoid horizon singularities
# air_mass = 1.0 / mu
# T0 = 0.8  # atmospheric transmittance
# S_dir = 1000 * mu * (T0 ** air_mass)
#
# f_diff = np.clip(0.15 + 0.6 * (1 - mu), 0.15, 0.7)
#
# S_diff = f_diff * S_dir / (1 - f_diff)
# Sdn = S_dir + S_diff
#
# from matplotlib import pyplot as plt
#
# plt.scatter(sza_degrees, Sdn)
# plt.ylabel('Sdn')
# plt.xlabel('sza')
# plt.show()