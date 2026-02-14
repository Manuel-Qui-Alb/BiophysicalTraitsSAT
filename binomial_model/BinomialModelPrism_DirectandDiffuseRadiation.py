import os
import numpy as np
from numpy import sqrt, sin, arcsin, cos, arccos, exp, pi, linspace, ceil
from plyfile import PlyData, PlyElement

def intersectBBox(ox, oy, oz, dx, dy, dz, sizex, sizey, sizez):

    # Intersection code below is adapted from Suffern (2007) Listing 19.1
    x0 = -0.5 * sizex
    x1 = 0.5 * sizex
    y0 = -0.5 * sizey
    y1 = 0.5 * sizey
    z0 = -1e-6
    z1 = sizez
    if dx == 0:
        a = 1e6
    else:
        a = 1.0 / dx
    if a >= 0:
        tx_min = (x0 - ox) * a
        tx_max = (x1 - ox) * a
    else:
        tx_min = (x1 - ox) * a
        tx_max = (x0 - ox) * a
    if dy == 0:
        b = 1e6
    else:
        b = 1.0 / dy
    if b >= 0:
        ty_min = (y0 - oy) * b
        ty_max = (y1 - oy) * b
    else:
        ty_min = (y1 - oy) * b
        ty_max = (y0 - oy) * b
    if dz == 0:
        c = 1e6
    else:
        c = 1.0 / dz
    if c >= 0:
        tz_min = (z0 - oz) * c
        tz_max = (z1 - oz) * c
    else:
        tz_min = (z1 - oz) * c
        tz_max = (z0 - oz) * c

    # find largest entering t value
    if tx_min > ty_min:
        t0 = tx_min
    else:
        t0 = ty_min
    if tz_min > t0:
        t0 = tz_min

    # find smallest exiting t value
    if tx_max < ty_max:
        t1 = tx_max
    else:
        t1 = ty_max
    if tz_max < t1:
        t1 = tz_max
    if t0 < t1 and t1 > 1e-6:
        if t0 > 1e-6:
            dr = t1-t0
        else:
            dr = t1
    else:
        dr = 0
    xe = ox + t1 * dx
    ye = oy + t1 * dy
    ze = oz + t1 * dz

    if dr == 0:
         # raise Exception('Shouldnt be here')
         dr, xe, ye, ze = np.nan, np.nan, np.nan, np.nan
    return dr, xe, ye, ze

def pathlengths(shape, scale_x, scale_y, scale_z, ray_zenith, ray_azimuth, nrays, outputfile=''):

    kEpsilon = 1e-5
    N = int(ceil(sqrt(nrays)))
    # Ray direction Cartesian unit vector
    dx = sin(ray_zenith) * cos(ray_azimuth)
    dy = sin(ray_zenith) * sin(ray_azimuth)
    dz = cos(ray_zenith)
    path_length = np.zeros(N*N)

    # Define bounding box depending on shape
    bbox_sizex = scale_x * (1.0 + kEpsilon)
    bbox_sizey = scale_y * (1.0 + kEpsilon)
    z_min = 0
    z_max = scale_z * (1.0 + kEpsilon)
    sx = bbox_sizex / N
    sy = bbox_sizey / N

    # loop over all rays, which originate at the bottom of the box
    for j in range(0, N):
        for i in range(0, N):
            # ray origin point
            ox = -0.5*bbox_sizex + (i+0.5)*sx
            oy = -0.5*bbox_sizey + (j+0.5)*sy
            oz = z_min - kEpsilon
            ze = 0
            dr = 0
            while ze <= z_max:

                # Intersect shape
                if shape == 'prism':
                    dr, _, _, _, = intersectBBox(ox, oy, oz, dx, dy, dz, scale_x, scale_y, scale_z)
                # Intersect bounding box walls
                _, xe, ye, ze = intersectBBox(ox, oy, oz, dx, dy, dz, bbox_sizex, bbox_sizey, 1e6)
                if ze <= z_max:  # intersection below object height -> record path length and cycle ray
                    path_length = np.append(path_length, dr)

                    ox = xe
                    oy = ye
                    oz = ze
                    if abs(ox-0.5*bbox_sizex) < kEpsilon:  # hit +x wall
                        ox = ox - bbox_sizex + kEpsilon
                    elif abs(ox+0.5*bbox_sizex) < kEpsilon:  # hit -x wall
                        ox = ox + bbox_sizex - kEpsilon
                    if abs(oy-0.5*bbox_sizey) < kEpsilon:  # hit +y wall
                        oy = oy - bbox_sizey + kEpsilon
                    elif abs(oy + 0.5 * bbox_sizey) < kEpsilon:  # hit -y wall
                        oy = oy + bbox_sizey - kEpsilon
            path_length[i+j*N] = dr
    if outputfile != '':
        np.savetxt(outputfile, path_length, delimiter=',')
    return path_length[path_length > kEpsilon]

def pathlengthdistribution(
    shape,
    scale_x,
    scale_y,
    scale_z,
    ray_zenith,
    ray_azimuth,
    nrays,
    plyfile="",
    bins=10,
    normalize=True,
):

    pl = pathlengths(shape, scale_x, scale_y, scale_z, ray_zenith, ray_azimuth, nrays, plyfile)

    hist, bin_edges = np.histogram(pl, bins=bins, density=normalize)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return {"hist": hist, "bin_centers": bin_centers}

import time
# ---------- Binomial radiation ----------
def compute_binomial_prism(
    sr, #Row spacing (meters)
    sza, # sun zenith angle (radians)
    psi, # sun azimuth relative to row orientation (radians)
    lai, # Leaf Area Index
    ameanv, # Leaf absorptivity in the visible (PAR) band
    ameann, # Leaf absorptivity in the near infra-red band (NIR) band
    rsoilv, # Soil absorptivity in the visible (PAR) band
    rsoiln,  # Soil absorptivity in the near infra-red band (NIR) band
    Srad_dir,  # Direct-beam incoming radiation (W m-2)
    Srad_diff,  # Diffuse incoming radiation (W m-2)
    fvis, # Fraction incoming radiation in the visible part of the spectrum
    fnir,  # Fraction incoming radiation in the near infra-red part of the spectrum
    CanopyHeight,# Canopy heigth (meters)
    wc, # Canopy width (meters)
    sp, #Plant spacing (meters)
    Gtheta, # fraction of leaf area projected in the direction of the sun
    nrays,
    Nbins,
    shape="prism",   # <-- shape of the canopy
    Nz_diff=16,
    Nphi_diff=32,  # sampling for diffuse hemisphere
):
    """
    Returns:
        Rc_binomial: canopy absorbed radiation (W m-2),
        Rs_binomial: soil absorbed radiation (W m-2)
    Both include direct + isotropic diffuse contributions.


    References:
    - Bailey, B.N., Ponce de León, M.A., and Krayenhoff, E.S., 2020. One-dimensional models of radiation transfer in homogeneous canopies: A review, re-evaluation, and improved model. Geoscientific Model Development 13:4789:4808
    - Bailey, B.N. and Fu, K., 2022. The probability distribution of absorbed direct, diffuse, and scattered radiation in plant canopies with varying structure. Agricultural and Forest Meteorology, 322, p.109009.
    - Ponce de León, M.A., Alfieri, J.G., Prueger, J.H., Hipps, L., Kustas, W.P., Agam, N., Bambach, N., McElrone, A.J., Knipper, K., Roby, M.C. and Bailey, B.N., 2025.
      One-dimensional modeling of radiation absorption by vine canopies: evaluation of existing model assumptions, and development of an improved generalized model.
      Agricultural and Forest Meteorology, 373, p.110706 (https://doi.org/10.1016/j.agrformet.2025.110706)
    - Path length distribution code: https://github.com/PlantSimulationLab/pathlengthdistribution
    """

    IncPAR_dir  = Srad_dir * fvis
    IncNIR_dir  = Srad_dir * fnir
    IncPAR_diff = Srad_diff * fvis
    IncNIR_diff = Srad_diff * fnir

    # Within-prism leaf area density
    a = lai * sr * sp / (sp * wc * CanopyHeight)

    # Adjusted effective spacing for row-oriented canopies
    s = sr * np.sin(psi) ** 2 + sp * np.cos(psi) ** 2
    s2 = s**2

    # Area of a single prism shadow at solar zenith of 0
    S0 = sp * wc
    #Area of a single prism shadow at solar zenith (θs) and relative azimuth angle
    # between the incident beam and the row direction (ϕ)
    Stheta = (
        sp * wc
        + sp * CanopyHeight * np.tan(sza) * np.abs(np.sin(psi))
        + wc * CanopyHeight * np.tan(sza) * np.abs(np.cos(psi))
    )
    #Number of prisms intersected by a beam of radiation
    N_crown = Stheta / S0

    # start_time_ale = time.perf_counter()
    dist = pathlengthdistribution(
        shape=shape,
        scale_x=wc,
        scale_y=sp,
        scale_z=CanopyHeight,
        ray_zenith=sza,
        ray_azimuth=psi,
        nrays=int(nrays),
        bins=int(Nbins),
    )
    N = dist["hist"] / (np.sum(dist["hist"]))
    S = dist["bin_centers"]

    # end_time_ale = time.perf_counter()
    # elapsed_time_ale = end_time_ale - start_time_ale
    # print(f"Execution time Ale: {elapsed_time_ale} seconds")

    #Probability of intersecting a leaf within a prism during first and second order scattering
    PlOne_PAR = np.sum(N * (1.0 - np.exp(-Gtheta * ameanv * a * S)))
    PlOne_NIR = np.sum(N * (1.0 - np.exp(-Gtheta * ameann * a * S)))
    PlTwo_PAR = np.sum(N * (1.0 - np.exp(-Gtheta * 2.0 * ameanv * a * S)))
    PlTwo_NIR = np.sum(N * (1.0 - np.exp(-Gtheta * 2.0 * ameann * a * S)))

    ####################################################################################################################
    ##################################################### CHANGE #######################################################
    ####################################################################################################################
    # Sometime S0 / s2 retrives values greater than 1. So It should be clipped
    S0_over_s2 = np.clip(np.where(s2 > 0, S0 / s2, 0.0), 0, 1)

    #Canopy-level probability ofinterception
    Pc1_PAR = (s2 / (sr * sp)) * (1.0 - (1.0 - PlOne_PAR * S0_over_s2) ** N_crown)
    Pc1_NIR = (s2 / (sr * sp)) * (1.0 - (1.0 - PlOne_NIR * S0_over_s2) ** N_crown)
    Pc2_PAR = (s2 / (sr * sp)) * (1.0 - (1.0 - PlTwo_PAR * S0_over_s2) ** N_crown)
    Pc2_NIR = (s2 / (sr * sp)) * (1.0 - (1.0 - PlTwo_NIR * S0_over_s2) ** N_crown)

    # Direct radiation absorbed by the soil
    Rs_dir = IncPAR_dir  * (1.0 - Pc1_PAR) * (1-rsoilv) + IncNIR_dir  * (1.0 - Pc1_NIR)* (1-rsoiln)

    soil_term_PAR = (1.0 - Pc1_PAR) * rsoilv * S0_over_s2 * PlOne_PAR
    soil_term_NIR = (1.0 - Pc1_NIR) * rsoiln * S0_over_s2 * PlOne_NIR

    # Direct radiation absorbed by the canopy
    Rc_dir = (Pc2_PAR + soil_term_PAR) * IncPAR_dir  + (Pc2_NIR + soil_term_NIR) * IncNIR_dir

    # ---- Diffuse sky part: integrate over hemisphere ----
    # rectangular integration over zenith and azimuth:
    dtheta = (0.5 * np.pi) / Nz_diff
    dphi = (2.0 * np.pi) / Nphi_diff

    # Accumulators for per-crown interception averages (used in soil terms)
    PlOne_PAR_diff_tot = 0.0
    PlOne_NIR_diff_tot = 0.0
    PlTwo_PAR_diff_tot = 0.0
    PlTwo_NIR_diff_tot = 0.0

    # Accumulators for canopy-level interception (crown overlap included)
    Pc1_PAR_diff = 0.0
    Pc1_NIR_diff = 0.0
    Pc2_PAR_diff = 0.0
    Pc2_NIR_diff = 0.0

    for i in range(Nz_diff):
        # midpoint zenith
        theta = (i + 0.5) * dtheta
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        for j in range(Nphi_diff):
            # midpoint azimuth
            phi = (j + 0.5) * dphi
            # directional weight for isotropic diffuse sky
            w_dir = (cos_theta * sin_theta * dtheta * dphi) / np.pi
            # pathlength distribution for this direction
            dist_d = pathlengthdistribution(
                shape=shape,
                scale_x=wc,
                scale_y=sp,
                scale_z=CanopyHeight,
                ray_zenith=theta,
                ray_azimuth=phi,
                nrays=int(nrays // (Nz_diff * Nphi_diff)
                          if nrays >= Nz_diff * Nphi_diff
                          else max(100, int(nrays / (Nz_diff * Nphi_diff)))),
                bins=int(Nbins),
            )

            N_d = dist_d["hist"] / np.sum(dist_d["hist"])
            S_d = dist_d["bin_centers"]

            # projection function
            G_dir = Gtheta

            # per-crown interception fractions (per beam, per spectral band)
            PlOne_PAR_d = np.sum(N_d * (1.0 - np.exp(-G_dir * ameanv * a * S_d)))
            PlOne_NIR_d = np.sum(N_d * (1.0 - np.exp(-G_dir * ameann * a * S_d)))
            PlTwo_PAR_d = np.sum(N_d * (1.0 - np.exp(-G_dir * 2.0 * ameanv * a * S_d)))
            PlTwo_NIR_d = np.sum(N_d * (1.0 - np.exp(-G_dir * 2.0 * ameann * a * S_d)))

            PlOne_PAR_diff_tot += w_dir * PlOne_PAR_d
            PlOne_NIR_diff_tot += w_dir * PlOne_NIR_d
            PlTwo_PAR_diff_tot += w_dir * PlTwo_PAR_d
            PlTwo_NIR_diff_tot += w_dir * PlTwo_NIR_d

            S0 = sp * wc
            Sthetadiff = (
                    sp * wc
                    + sp * CanopyHeight * np.tan(theta) * np.abs(np.sin(phi))
                    + wc * CanopyHeight * np.tan(theta) * np.abs(np.cos(phi))
            )
            N_crowndiff = Sthetadiff / S0

            Pc1_PAR_diff += w_dir * (s2 / (sr * sp)) * (1.0 - (1.0 - PlOne_PAR_d * S0_over_s2) ** N_crowndiff)
            Pc1_NIR_diff += w_dir * (s2 / (sr * sp)) * (1.0 - (1.0 - PlOne_NIR_d * S0_over_s2) ** N_crowndiff)
            Pc2_PAR_diff += w_dir * (s2 / (sr * sp)) * (1.0 - (1.0 - PlTwo_PAR_d * S0_over_s2) ** N_crowndiff)
            Pc2_NIR_diff += w_dir * (s2 / (sr * sp)) * (1.0 - (1.0 - PlTwo_NIR_d * S0_over_s2) ** N_crowndiff)

    # ---- Diffuse contributions to soil and canopy ----
    Rs_diff = (IncPAR_diff * (1.0 - Pc1_PAR_diff) * (1.0 - rsoilv) +
               IncNIR_diff * (1.0 - Pc1_NIR_diff) * (1.0 - rsoiln))

    soil_term_PAR_diff = (1.0 - Pc1_PAR_diff) * rsoilv * S0_over_s2 * PlOne_PAR_diff_tot
    soil_term_NIR_diff = (1.0 - Pc1_NIR_diff) * rsoiln * S0_over_s2 * PlOne_NIR_diff_tot

    Rc_diff = ((Pc2_PAR_diff + soil_term_PAR_diff) * IncPAR_diff +
               (Pc2_NIR_diff + soil_term_NIR_diff) * IncNIR_diff)

    # ---- Total (direct + diffuse) ----
    Rs_binomial = Rs_dir + Rs_diff
    Rc_binomial = Rc_dir + Rc_diff

    return Rc_binomial, Rs_binomial


from GeneralFunctions import _to_2d, normalize_inputs_flat


def hist_divide_sumhist(dist):
    N = dist["hist"] / (np.sum(dist["hist"]))
    S = dist["bin_centers"]
    return dict(N=N, S=S)


def calculate_p_lone_p_ltwo(N, G, abs_leaf, a, S):
    PlOne_d = np.sum(N * (1.0 - np.exp(-G * abs_leaf * a * S)), axis=1)
    PlTwo_d = np.sum(N * (1.0 - np.exp(-G * 2.0 * abs_leaf * a * S)), axis=1)
    return _to_2d(PlOne_d), _to_2d(PlTwo_d)

Pc_function = lambda s2, sr, sp, Pl, S0_over_s2, N_crown: (s2 / (sr * sp)) * (1.0 - (1.0 - Pl * S0_over_s2)
                                                                                  ** N_crown)

Pc_diff_function = lambda w_dir, s2, sr, sp, Pl, S0_over_s2, N_crowndiff:  (
                    w_dir * (s2 / (sr * sp)) * (1.0 - (1.0 - Pl * S0_over_s2) ** N_crowndiff))

from joblib import Parallel, delayed


# ---------- Binomial radiation ----------
def compute_binomial_prism_manuel(
    sr, #Row spacing (meters)
    sza, # sun zenith angle (radians)
    psi, # sun azimuth relative to row orientation (radians)
    lai, # Leaf Area Index
    ameanv, # Leaf absorptivity in the visible (PAR) band
    ameann, # Leaf absorptivity in the near infra-red band (NIR) band
    rsoilv, # Soil absorptivity in the visible (PAR) band
    rsoiln,  # Soil absorptivity in the near infra-red band (NIR) band
    Srad_dir,  # Direct-beam incoming radiation (W m-2)
    Srad_diff,  # Diffuse incoming radiation (W m-2)
    fvis, # Fraction incoming radiation in the visible part of the spectrum
    fnir,  # Fraction incoming radiation in the near infra-red part of the spectrum
    CanopyHeight,# Canopy heigth (meters)
    wc, # Canopy width (meters)
    sp, #Plant spacing (meters)
    Gtheta, # fraction of leaf area projected in the direction of the sun
    nrays,
    Nbins,
    shape="prism",   # <-- shape of the canopy
    Nz_diff=16,
    Nphi_diff=32,  # sampling for diffuse hemisphere
):
    """
    Returns:
        Rc_binomial: canopy absorbed radiation (W m-2),
        Rs_binomial: soil absorbed radiation (W m-2)
    Both include direct + isotropic diffuse contributions.


    References:
    - Bailey, B.N., Ponce de León, M.A., and Krayenhoff, E.S., 2020. One-dimensional models of radiation transfer in homogeneous canopies: A review, re-evaluation, and improved model. Geoscientific Model Development 13:4789:4808
    - Bailey, B.N. and Fu, K., 2022. The probability distribution of absorbed direct, diffuse, and scattered radiation in plant canopies with varying structure. Agricultural and Forest Meteorology, 322, p.109009.
    - Ponce de León, M.A., Alfieri, J.G., Prueger, J.H., Hipps, L., Kustas, W.P., Agam, N., Bambach, N., McElrone, A.J., Knipper, K., Roby, M.C. and Bailey, B.N., 2025.
      One-dimensional modeling of radiation absorption by vine canopies: evaluation of existing model assumptions, and development of an improved generalized model.
      Agricultural and Forest Meteorology, 373, p.110706 (https://doi.org/10.1016/j.agrformet.2025.110706)
    - Path length distribution code: https://github.com/PlantSimulationLab/pathlengthdistribution
    """

    (sr, sza, psi, lai, ameanv, ameann, rsoilv, rsoiln, Srad_dir, Srad_diff, fvis, fnir, CanopyHeight,
     wc, sp, Gtheta, ref_shape) = normalize_inputs_flat(sr, sza, psi, lai, ameanv, ameann, rsoilv, rsoiln, Srad_dir,
                                                        Srad_diff, fvis, fnir, CanopyHeight, wc, sp, Gtheta)

    IncPAR_dir  = Srad_dir * fvis
    IncNIR_dir  = Srad_dir * fnir
    IncPAR_diff = Srad_diff * fvis
    IncNIR_diff = Srad_diff * fnir

    # Within-prism leaf area density
    a = lai * sr * sp / (sp * wc * CanopyHeight)

    # Adjusted effective spacing for row-oriented canopies
    s = sr * np.sin(psi) ** 2 + sp * np.cos(psi) ** 2
    s2 = s**2

    # Area of a single prism shadow at solar zenith of 0
    S0 = sp * wc
    #Area of a single prism shadow at solar zenith (θs) and relative azimuth angle
    # between the incident beam and the row direction (ϕ)
    Stheta = (
        sp * wc
        + sp * CanopyHeight * np.tan(sza) * np.abs(np.sin(psi))
        + wc * CanopyHeight * np.tan(sza) * np.abs(np.cos(psi))
    )
    #Number of prisms intersected by a beam of radiation
    N_crown = Stheta / S0

    # start_time_man = time.perf_counter()

    dists = Parallel(n_jobs= max(1, os.cpu_count() - 2), prefer="processes", verbose=0)(
        delayed(pathlengthdistribution)(
            shape=shape, scale_x=wx, scale_y=spy, scale_z=hz,
            ray_zenith=sz, ray_azimuth=az, nrays=nrays, bins=Nbins
        )
        for wx, spy, hz, sz, az in zip(wc[:, 0], sp[:, 0], CanopyHeight[:, 0], sza[:, 0], psi[:, 0])
    )
    # #
    # dists = [pathlengthdistribution(shape=shape, scale_x=wx, scale_y=spy, scale_z=hz, ray_zenith=sz, ray_azimuth=az,
    #                                 nrays=nrays, bins=Nbins)
    #          for wx, spy, hz, sz, az in zip(wc[:, 0], sp[:, 0], CanopyHeight[:, 0], sza[:, 0], psi[:, 0])]
    # end_time_man = time.perf_counter()
    #
    # elapsed_time_man = end_time_man - start_time_man
    # print(f"Execution time Manuel: {elapsed_time_man} seconds")

    dict_N_S = list(map(hist_divide_sumhist, dists))

    N = np.array([dict_N_S[x]['N'] for x in np.arange(len(dict_N_S))])
    S = np.array([dict_N_S[x]['S'] for x in np.arange(len(dict_N_S))])

    #Probability of intersecting a leaf within a prism during first and second order scattering
    PlOne_PAR, PlTwo_PAR = calculate_p_lone_p_ltwo(N, G=Gtheta, abs_leaf=ameanv, a=a, S=S)
    PlOne_NIR, PlTwo_NIR = calculate_p_lone_p_ltwo(N, G=Gtheta, abs_leaf=ameann, a=a, S=S)

    ####################################################################################################################
    ##################################################### CHANGE #######################################################
    ####################################################################################################################
    # Sometime S0 / s2 retrives values greater than 1. So It should be clipped
    S0_over_s2 = np.clip(np.where(s2 > 0, S0 / s2, 0.0), 0, 1)

    #Canopy-level probability of interception
    Pc1_PAR = Pc_function(s2, sr, sp, PlOne_PAR, S0_over_s2, N_crown)
    Pc1_NIR = Pc_function(s2, sr, sp, PlOne_NIR, S0_over_s2, N_crown)
    Pc2_PAR = Pc_function(s2, sr, sp, PlTwo_PAR, S0_over_s2, N_crown)
    Pc2_NIR = Pc_function(s2, sr, sp, PlTwo_NIR, S0_over_s2, N_crown)

    # Direct radiation absorbed by the soil
    Rs_dir = IncPAR_dir  * (1.0 - Pc1_PAR) * (1-rsoilv) + IncNIR_dir  * (1.0 - Pc1_NIR)* (1-rsoiln)

    soil_term_PAR = (1.0 - Pc1_PAR) * rsoilv * S0_over_s2 * PlOne_PAR
    soil_term_NIR = (1.0 - Pc1_NIR) * rsoiln * S0_over_s2 * PlOne_NIR

    # Direct radiation absorbed by the canopy
    Rc_dir = (Pc2_PAR + soil_term_PAR) * IncPAR_dir  + (Pc2_NIR + soil_term_NIR) * IncNIR_dir

    # ---- Diffuse sky part: integrate over hemisphere ----
    # rectangular integration over zenith and azimuth:
    dtheta = (0.5 * np.pi) / Nz_diff
    dphi = (2.0 * np.pi) / Nphi_diff

    # Accumulators for per-crown interception averages (used in soil terms)
    PlOne_PAR_diff_tot = 0.0
    PlOne_NIR_diff_tot = 0.0
    PlTwo_PAR_diff_tot = 0.0
    PlTwo_NIR_diff_tot = 0.0

    # Accumulators for canopy-level interception (crown overlap included)
    Pc1_PAR_diff = 0.0
    Pc1_NIR_diff = 0.0
    Pc2_PAR_diff = 0.0
    Pc2_NIR_diff = 0.0

    for i in range(Nz_diff):
        # midpoint zenith
        theta = (i + 0.5) * dtheta
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        for j in range(Nphi_diff):
            # midpoint azimuth
            phi = (j + 0.5) * dphi
            # directional weight for isotropic diffuse sky
            w_dir = (cos_theta * sin_theta * dtheta * dphi) / np.pi

            # pathlength distribution for this direction
            nray = int(nrays // (Nz_diff * Nphi_diff)
                       if nrays >= Nz_diff * Nphi_diff
                       else max(100, int(nrays / (Nz_diff * Nphi_diff)))
                       )

            dist_d_N_S = [pathlengthdistribution(shape=shape, scale_x=wx, scale_y=spy, scale_z=hz,
                                                 ray_zenith=theta, ray_azimuth=phi, nrays=nray, bins=int(Nbins))
                              for wx, spy, hz in zip(wc[:, 0], sp[:, 0], CanopyHeight[:, 0])]
            #
            # dist_d_N_S = Parallel(n_jobs=-1, prefer="processes")(
            #     delayed(pathlengthdistribution)(
            #         shape=shape, scale_x=wx, scale_y=spy, scale_z=hz,
            #         ray_zenith=theta, ray_azimuth=phi, nrays=nrays, bins=Nbins
            #     )
            #     for wx, spy, hz in zip(wc[:, 0], sp[:, 0], CanopyHeight[:, 0]))

            dist_d_N_S = list(map(hist_divide_sumhist, dist_d_N_S))
            N_d = np.array([dist_d_N_S[x]['N'] for x in np.arange(len(dist_d_N_S))])
            S_d = np.array([dist_d_N_S[x]['S'] for x in np.arange(len(dist_d_N_S))])

            # projection function
            G_dir = Gtheta

            # per-crown interception fractions (per beam, per spectral band)
            PlOne_PAR_d, PlTwo_PAR_d = calculate_p_lone_p_ltwo(N=N_d, G=G_dir, abs_leaf=ameanv, a=a, S=S_d)
            PlOne_NIR_d, PlTwo_NIR_d = calculate_p_lone_p_ltwo(N=N_d, G=G_dir, abs_leaf=ameann, a=a, S=S_d)

            PlOne_PAR_diff_tot += w_dir * PlOne_PAR_d
            PlOne_NIR_diff_tot += w_dir * PlOne_NIR_d
            PlTwo_PAR_diff_tot += w_dir * PlTwo_PAR_d
            PlTwo_NIR_diff_tot += w_dir * PlTwo_NIR_d

            S0 = sp * wc
            Sthetadiff = (
                    sp * wc
                    + sp * CanopyHeight * np.tan(theta) * np.abs(np.sin(phi))
                    + wc * CanopyHeight * np.tan(theta) * np.abs(np.cos(phi))
            )
            N_crowndiff = Sthetadiff / S0


            Pc1_PAR_diff += Pc_diff_function(w_dir, s2, sr, sp, PlOne_PAR_d, S0_over_s2, N_crowndiff)
            Pc1_NIR_diff += Pc_diff_function(w_dir, s2, sr, sp, PlOne_NIR_d, S0_over_s2, N_crowndiff)
            Pc2_PAR_diff += Pc_diff_function(w_dir, s2, sr, sp, PlTwo_PAR_d, S0_over_s2, N_crowndiff)
            Pc2_NIR_diff += Pc_diff_function(w_dir, s2, sr, sp, PlTwo_NIR_d, S0_over_s2, N_crowndiff)


    # ---- Diffuse contributions to soil and canopy ----
    Rs_diff = (IncPAR_diff * (1.0 - Pc1_PAR_diff) * (1.0 - rsoilv) +
               IncNIR_diff * (1.0 - Pc1_NIR_diff) * (1.0 - rsoiln))

    soil_term_PAR_diff = (1.0 - Pc1_PAR_diff) * rsoilv * S0_over_s2 * PlOne_PAR_diff_tot
    soil_term_NIR_diff = (1.0 - Pc1_NIR_diff) * rsoiln * S0_over_s2 * PlOne_NIR_diff_tot

    Rc_diff = ((Pc2_PAR_diff + soil_term_PAR_diff) * IncPAR_diff +
               (Pc2_NIR_diff + soil_term_NIR_diff) * IncNIR_diff)

    # ---- Total (direct + diffuse) ----
    Rs_binomial = Rs_dir + Rs_diff
    Rc_binomial = Rc_dir + Rc_diff

    Rs_binomial = np.asarray(Rs_binomial).reshape(ref_shape)
    Rc_binomial = np.asarray(Rc_binomial).reshape(ref_shape)
    return Rc_binomial, Rs_binomial

