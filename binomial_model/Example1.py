import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BinomialModelPrism_DirectandDiffuseRadiation import compute_binomial_prism

IncomingRad = np.array([362.5, 563.1, 743.7, 887.7, 982.7, 1021.0, 999.6, 920.2])  # W/m²
azimuth = np.array([1.3665, 1.5139, 1.6817, 1.9052, 2.2789, 3.0089, 3.8417, 4.2938])
zenith  = np.array([1.1520, 0.9485, 0.7433, 0.5430, 0.3639, 0.2616, 0.3226, 0.4886])

# ------------------------------
# SENSITIVITY TO FDIFF
# ------------------------------
fdiff_values = np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0
Rc_mean, Rs_mean = [], []

for fd in fdiff_values:
    Rc_list, Rs_list = [], []
    sza_i = float(zenith[6])
    ssa_i = float(azimuth[6])
    Rc, Rs = compute_binomial_prism(
        sr=3.35,
        sza=sza_i,
        psi=ssa_i,
        lai=1.7923,
        ameanv=0.8,
        ameann=0.34,
        rsoilv=0.12,
        rsoiln=0.24,
        Srad_dir=IncomingRad[6] * (1-fd),
        Srad_diff=IncomingRad[6] * fd,
        fvis=0.47,
        fnir=0.53,
        CanopyHeight=1,
        wc=0.5,
        sp=100,
        Gtheta=0.5,
        nrays=5000,
        Nbins=40,
        shape="prism",
        Nz_diff=8,
        Nphi_diff=16,
    )
    Rc_list.append(Rc[0] if hasattr(Rc, '__len__') else Rc)
    Rs_list.append(Rs[0] if hasattr(Rs, '__len__') else Rs)

    Rc_mean.append(np.mean(Rc_list))
    Rs_mean.append(np.mean(Rs_list))
# ------------------------------
# PLOT EXPECTED BEHAVIOR
# ------------------------------
plt.figure(figsize=(6,5))
plt.plot(fdiff_values, Rc_mean, marker='o', label="Rc (Canopy absorbed)")
plt.plot(fdiff_values, Rs_mean, marker='s', label="Rs (Soil absorbed)")
plt.xlabel("Diffuse fraction (fdiff)")
plt.ylabel("Binomial - absorbed radiation at 13:00 (W/m²)")
plt.legend()
plt.grid(True)
plt.savefig("fdiff_sensitivity.png", dpi=300)
plt.show()
