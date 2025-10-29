import numpy as np


def e_T(T):
    """
    Calculates Saturation Vapor Pressure Enviromental biophysics (page 42).

    Parameters
    ----------
    T : float
        air temperature in Celsius

    Returns
    ---------
    es : Saturation vapor pressure [kPa]
    """
    A = 0.611
    B = 17.502
    C = 240.97
    es = A * np.exp((B * T) / (T + C))
    return es


def calculate_vpd(Tmean_C, RH, Tmax_C=None, Tmin_C=None):
    """
    Calculates Vapor Pressure Deficit (VPD) According FAO 56.

    Parameters
    ----------
    Tmean_C : float
            air temperature in Celsius
    Tmax_C : float
            max air temperature in Celsius
    Tmin_C : float
            mion air temperature in Celsius
    RH : float
            Relative Humidity

    Returns
    ---------
    et : Saturation vapor pressure [kPa]
    es : Mean saturation vapor pressure [kPa]
    ea : Real vapor pressure [kPa]
    vpd : vapor pressure deficit [kPa]
    """

    try:
        es = (e_T(Tmax_C) + e_T(Tmin_C)) / 2
    except:
        print('e°(t) is using for calculating vpd instead es (see FAO 56 page 36)')
        es = e_T(Tmean_C)
    ea = RH * es
    vpd = es - ea

    return es, ea, vpd