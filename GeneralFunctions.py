import numpy as np

import numpy as np

def _to_2d(a, name="var"):
    """
    Convert scalar / 1D / 2D input to a 2D numpy array of shape (n, c).
    - scalar -> (1, 1)
    - (n,)   -> (n, 1)
    - (n, c) -> (n, c)
    """
    a = np.asarray(a)

    if a.ndim == 0:              # scalar
        return a.reshape(1, 1)
    if a.ndim == 1:              # (n,)
        return a.reshape(-1, 1)
    if a.ndim == 2:              # (n, c)
        return a

    raise ValueError(f"{name} must be scalar, 1D, or 2D. Got shape={a.shape} (ndim={a.ndim})")


def _broadcast_to_n(a2d, n, name="var"):
    """
    Ensure first dimension is n by broadcasting if possible.
    Allows:
      (n, c) stays
      (1, c) -> broadcast to (n, c)
    Rejects:
      (m, c) where m not in {1, n}
    """
    if a2d.shape[0] == n:
        return a2d
    if a2d.shape[0] == 1:
        return np.broadcast_to(a2d, (n, a2d.shape[1]))
    raise ValueError(f"{name} has incompatible n: got {a2d.shape[0]}, expected {n} (or 1 for broadcasting).")


def _broadcast_to_shape(a2d, ref_shape, name="var"):
    """
    Broadcast a 2D array to a reference shape (n, c) when possible.

    Allows:
      (n, c) stays
      (n, 1) -> broadcast to (n, c)
      (1, c) -> broadcast to (n, c)
      (1, 1) -> broadcast to (n, c)

    Rejects any other incompatible shape.
    """
    n, c = ref_shape
    r, cc = a2d.shape

    # already OK
    if (r, cc) == (n, c):
        return a2d

    # broadcastable cases
    if r in (1, n) and cc in (1, c):
        return np.broadcast_to(a2d, (n, c))

    raise ValueError(
        f"{name} has incompatible shape {a2d.shape}; expected broadcastable to {ref_shape}."
    )


def normalize_inputs_flat(
    sr, sza, psi, lai, ameanv, ameann, rsoilv, rsoiln,
    Srad_dir, Srad_diff, fvis, fnir, CanopyHeight, wc, sp, Gtheta
):
    """
    Normalize inputs and return FLATTENED 1D arrays + reference shape for reshaping outputs.

    Steps:
      1) Convert each input to 2D (n, c) using _to_2d
      2) Determine reference shape (n, c) as the "largest" 2D among inputs:
         - n = max n across inputs
         - c = max c across inputs
      3) Broadcast every input to (n, c) when possible:
         - (n,1) or (1,c) or (1,1) are expanded to (n,c)
      4) Flatten to 1D of length n*c
    """

    # 1) Convert to 2D arrays
    arrs = {
        "sr": _to_2d(sr, "sr"),
        "sza": _to_2d(sza, "sza"),
        "psi": _to_2d(psi, "psi"),
        "lai": _to_2d(lai, "lai"),
        "ameanv": _to_2d(ameanv, "ameanv"),
        "ameann": _to_2d(ameann, "ameann"),
        "rsoilv": _to_2d(rsoilv, "rsoilv"),
        "rsoiln": _to_2d(rsoiln, "rsoiln"),
        "Srad_dir": _to_2d(Srad_dir, "Srad_dir"),
        "Srad_diff": _to_2d(Srad_diff, "Srad_diff"),
        "fvis": _to_2d(fvis, "fvis"),
        "fnir": _to_2d(fnir, "fnir"),
        "CanopyHeight": _to_2d(CanopyHeight, "CanopyHeight"),
        "wc": _to_2d(wc, "wc"),
        "sp": _to_2d(sp, "sp"),
        "Gtheta": _to_2d(Gtheta, "Gtheta"),
    }

    # 2) Determine reference shape (n, c)
    n = max(a.shape[0] for a in arrs.values())
    c = max(a.shape[1] for a in arrs.values())
    ref_shape = (n, c)

    # 3) Broadcast all to (n, c)
    for k in arrs:
        arrs[k] = _broadcast_to_shape(arrs[k], ref_shape, k)

    # 4) Flatten to 1D
    return (
        arrs["sr"].reshape(-1, 1),
        arrs["sza"].reshape(-1, 1),
        arrs["psi"].reshape(-1, 1),
        arrs["lai"].reshape(-1, 1),
        arrs["ameanv"].reshape(-1, 1),
        arrs["ameann"].reshape(-1, 1),
        arrs["rsoilv"].reshape(-1, 1),
        arrs["rsoiln"].reshape(-1, 1),
        arrs["Srad_dir"].reshape(-1, 1),
        arrs["Srad_diff"].reshape(-1, 1),
        arrs["fvis"].reshape(-1, 1),
        arrs["fnir"].reshape(-1, 1),
        arrs["CanopyHeight"].reshape(-1, 1),
        arrs["wc"].reshape(-1, 1),
        arrs["sp"].reshape(-1, 1),
        arrs["Gtheta"].reshape(-1, 1),
        ref_shape,
    )


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