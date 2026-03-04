import numpy as np
import pvlib
import pandas as pd
from matplotlib import pyplot as plt

site = pvlib.location.Location(
    latitude=38.5449,
    longitude=-121.7405,
    tz='US/Pacific',
    altitude=18,      # meters; small effect, 15–25 m is fine
    name='Davis_CA'
)

def solar_doy_hour_to_times(doys, solar_hours, year=2024,
                            lon_deg=-121.7405,
                            tz='US/Pacific',
                            lon_std_deg=-120.0):
    """
    doys: array-like, 1..365
    solar_hours: array-like, local solar time in decimal hours (e.g., 12.5 = 12:30 solar)
    Returns: tz-aware DatetimeIndex in civil time that corresponds to those solar hours.
    """
    doys = np.asarray(doys).astype(int).ravel()
    solar_hours = np.asarray(solar_hours).astype(float).ravel()

    # equation of time (minutes), Spencer 1971 (pvlib returns minutes)
    # pvlib expects dayofyear array
    eot_min = pvlib.solarposition.equation_of_time_spencer71(doys)

    # time correction (minutes): EoT + 4*(lon - lon_std)
    tc_min = eot_min + 4.0 * (lon_deg - lon_std_deg)

    # convert solar hour -> civil hour
    civil_hours = solar_hours - tc_min / 60.0

    base = pd.Timestamp(f'{year}-01-01', tz=tz)
    times = base + pd.to_timedelta(doys - 1, unit='D') + pd.to_timedelta(civil_hours, unit='h')
    return pd.DatetimeIndex(times)

doy = np.random.randint(1, 365, 1000)
hour = np.random.randint(10, 16, 1000)
times = solar_doy_hour_to_times(doy, hour, year=2024, lon_deg=site.longitude, tz=site.tz)

solpos = site.get_solarposition(times)
sza_degrees = solpos.zenith
saa_degrees = solpos.azimuth

irradiance_cs = site.get_clearsky(times)
St = irradiance_cs.ghi

plt.scatter(sza_degrees, St)
plt.show()

plt.scatter(doy, St)
plt.show()

plt.scatter(hour, St)
plt.show()