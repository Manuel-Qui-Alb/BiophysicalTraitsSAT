import numpy as np


def estimate_Kbe(x_LAD, sza):
    """
    Estimate the beam extinction coefficient (Kbe) using the
    ellipsoidal leaf angle distribution model from Campbell (1986).

    Parameters
    ----------
    x_LAD : float
        Ellipsoidal leaf angle distribution parameter (dimensionless).
        x = 1 for spherical LAD, x < 1 for more vertical leaves,
        and x > 1 for more horizontal leaves.
    sza : float
        Solar zenith angle in radians.

    Returns
    -------
    float
        Extinction coefficient for direct beam radiation (dimensionless).
    """
    K_be = np.sqrt(x_LAD ** 2 + np.tan(sza) ** 2) / (x_LAD + 1.774 * (x_LAD + 1.182) ** -0.733)
    K_be = np.clip(K_be, 1e-6, None)
    return K_be

def canopy_gap_fraction(LAI, fv0, w_V, h_V, sza, saa, row_azimuth, hb_V=0, L=None, x_LAD=1 ):
    """
    Estimate the canopy gap fraction from Campbell and Norman (1998), Parry (2019), and Ponce de Leon (2025).

    Parameters
    ----------
    LAI : float
        Leaf Area Index.
    f_V0:
        Apparent nadir fractional cover
    w_V: float
        Vegetation width.
    h_V: float
        Vegetation Height.
    hb_V: float
        The height of the first living branch.
    L: float
        row separation (m)
    x_LAD : float
        Ellipsoidal leaf angle distribution parameter (dimensionless).
        x = 1 for spherical LAD, x < 1 for more vertical leaves,
        and x > 1 for more horizontal leaves.
    sza : float
        Solar zenith angle in radians.
    saa : float
        Solar Azimuth angle in radians
    row_azimuth : float
        row azimuth angle in radians
    phi : float
        relative azimuth angle between the incident beam and the row direction (radians)
        row direction - solar azimuth

    Returns
    -------
    float
        Omega: Off-Nadir clumping index (dimensionless).
    """
    phi = saa - row_azimuth

    K_be = estimate_Kbe(x_LAD, sza)

    # Solar canopy view factor f_sc(theta, phi) Eq. 15
    alpha = np.tan(sza) * np.abs(np.sin(phi))

    try:
        f_sc = (w_V + (h_V - hb_V) * alpha) / L
    except:
        f_sc = fv0 * (1 + ( (h_V - hb_V) * alpha ) / w_V )

    f_sc = np.clip(f_sc, 0.0, 1.0)

    # The gap fraction of the real-world canopy Eq 13.
    gap_phi = f_sc * np.exp(-K_be * LAI) + (1 - f_sc)
    gap_phi = np.clip(gap_phi, 0, 1.0)
    return gap_phi

def rectangular_row_clumping_index_parry(LAI, fv0, w_V, h_V, sza, saa, row_azimuth, hb_V=0, L=None, x_LAD=1 ):
    """
    Estimate the off nadir clumpling index from Campbell and Norman (1998) and Parry (2019).

    Parameters
    ----------
    LAI : float
        Leaf Area Index.
    f_V0:
        Apparent nadir fractional cover
    w_V: float
        Vegetation width.
    h_V: float
        Vegetation Height.
    hb_V: float
        The height of the first living branch.
    L: float
        row separation (m)
    x_LAD : float
        Ellipsoidal leaf angle distribution parameter (dimensionless).
        x = 1 for spherical LAD, x < 1 for more vertical leaves,
        and x > 1 for more horizontal leaves.
    sza : float
        Solar zenith angle in radians.
    saa : float
        Solar Azimuth angle in radians
    row_azimuth : float
        row azimuth angle in radians
    phi : float
        relative azimuth angle between the incident beam and the row direction (radians)
        row direction - solar azimuth

    Returns
    -------
    float
        Omega: Off-Nadir clumping index (dimensionless).
    """

    K_be = estimate_Kbe(x_LAD, sza)

    gap_phi = canopy_gap_fraction(LAI, fv0, w_V, h_V, sza, saa, row_azimuth, hb_V=0, L=None, x_LAD=1)
    omega_row = -np.log(gap_phi) / (K_be * LAI)
    omega_row = np.clip(omega_row, 0.05, 2)

    return omega_row