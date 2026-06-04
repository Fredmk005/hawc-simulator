from __future__ import annotations

import numpy as np
import pandas as pd
import sasktran2 as sk
from hamilton.function_modifiers import config

from hawcsimulator.datastructures.viewinggeo import ObservationContainer
from hawcsimulator.geometry.observation import SimulatedObservationGeometry


@config.when(observation_method="limb")
def observation__limb(
    viewing_tangent_altitudes: np.ndarray,
    time: pd.Timestamp,
    tangent_latitude: float,
    tangent_longitude: float,
    observer_altitude: float,
    sample_wavelengths: np.ndarray,
    observer_longitude: float | None = None,
    observer_latitude: float | None = None,
    tangent_solar_zenith_angle: float | None = None,
    tangent_solar_azimuth_angle: float | None = None,
    maximum_allowed_sza: float = 88,
) -> ObservationContainer:
    """
    Creates an idealized limb viewing observation based on a set of viewing tangent altitudes and
    optionally solar angles at the tangent point.  The observation is created assuming the solar angles
    are the same for every tangent altitude.


    Parameters
    ----------
    viewing_tangent_altitudes : np.array
        Tangent altitudes for the observation in [m], assuming no refraction
    time : pd.Timestamp
        Time of the observation.  Primarily used when the solar angles are not specified to calculate
        the sun position
    tangent_latitude : float
        Tangent latitude in [degrees]
    tangent_longitude : float
        Tangent longitude in [degrees]
    observer_altitude : float
        Altitude of the observer in [m]
    sample_wavelengths : np.ndarray
        Observation sample wavelengths for the instrument in [nm]
    observer_latitude: float
        Latitude of the observer in [degrees], optional, only required if time based solar angles are used
    observer_longitude: float
        Longitude of the observer in [degrees], optional, only required if time based solar angles are used
    tangent_solar_zenith_angle : float | None, optional
        Solar zenith angle in [degrees], by default None indicating it will be calculated from the observation time
    tangent_solar_azimuth_angle : float | None, optional
        Relative solar azimuth angle in [degrees] where 0 degrees is forward scatter, by default None indicating it will be
        calculated from the observation time

    Returns
    -------
    ObservationContainer
    """
    tan_alts = viewing_tangent_altitudes
    obs_time = time

    if (
        tangent_solar_zenith_angle is not None
        and tangent_solar_azimuth_angle is not None
    ):
        # Forced angles
        solar_handler = sk.solar.SolarGeometryHandlerForced(
            tangent_solar_zenith_angle, tangent_solar_azimuth_angle
        )
        viewing_azimuth = 0.0
    else:
        # Time angles
        solar_handler = sk.solar.SolarGeometryHandlerAstropy()

        if observer_latitude is None or observer_longitude is None:
            msg = "Have to specify observer_latitude and observer_longitude when using time based solar angles"
            raise ValueError(msg)

        # Here we also have to calculate the viewing azimuth angle
        tangent_geo = sk.geodetic.WGS84()
        tangent_geo.from_lat_lon_alt(tangent_latitude, tangent_longitude, 25000.0)

        obs_geo = sk.geodetic.WGS84()
        obs_geo.from_lat_lon_alt(
            observer_latitude, observer_longitude, observer_altitude
        )

        viewing_ray = tangent_geo.location - obs_geo.location
        viewing_ray = viewing_ray / np.linalg.norm(viewing_ray)

        north = -1 * tangent_geo.local_south  # x axis
        east = -1 * tangent_geo.local_west  # y axis

        viewing_azimuth = np.arctan2(
            np.dot(east, viewing_ray), np.dot(north, viewing_ray)
        )

    viewing_geo = sk.viewinggeo.LimbVertical.from_tangent_parameters(
        solar_handler=solar_handler,
        tangent_altitudes=tan_alts,
        tangent_latitude=tangent_latitude,
        tangent_longitude=tangent_longitude,
        time=obs_time,
        observer_altitude=observer_altitude,
        viewing_azimuth=viewing_azimuth,
    )

    obs_sza = np.rad2deg(np.arccos(viewing_geo.recommended_cos_sza()))
    if obs_sza > maximum_allowed_sza:
        msg = f"Observation SZA: {obs_sza} is greater than the allowed maximum: {maximum_allowed_sza}"
        raise ValueError(msg)

    return ObservationContainer(
        SimulatedObservationGeometry(
            viewing_geo=viewing_geo,
            sample_wavel=sample_wavelengths,
        ),
        obs_time,
    )
