import datetime as dt
import math

def equation_of_time_minutes(utc_dt: dt.datetime) -> float:
    """
    NOAA formulation (minutes). Works best with UTC datetime (timezone-aware or naive as UTC).
    """
    # Ensure we use UTC date/time
    if utc_dt.tzinfo is not None:
        utc_dt = utc_dt.astimezone(dt.timezone.utc).replace(tzinfo=None)

    year_start = dt.datetime(utc_dt.year, 1, 1)
    day_of_year = (utc_dt - year_start).days + 1
    # fractional hour in UTC
    frac_hour = utc_dt.hour + utc_dt.minute/60 + utc_dt.second/3600
    # fractional year (radians); NOAA variant
    gamma = 2*math.pi/365 * (day_of_year - 1 + (frac_hour - 12)/24)

    eot = (229.18 * (0.000075
                     + 0.001868*math.cos(gamma)
                     - 0.032077*math.sin(gamma)
                     - 0.014615*math.cos(2*gamma)
                     - 0.040849*math.sin(2*gamma)))
    return eot  # minutes (can be negative)


def utc_to_local_solar_time(utc_millis: int, longitude_deg: float):
    utc =  dt.datetime.fromtimestamp(utc_millis/1000, tz=dt.timezone.utc)
    lon_offset_min = 4.0 * longitude_deg
    lmst = utc + dt.timedelta(minutes=lon_offset_min)

    eot_min = equation_of_time_minutes(utc)
    ast = lmst + dt.timedelta(minutes=eot_min)
    return ast
