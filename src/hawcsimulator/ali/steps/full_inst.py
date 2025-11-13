from __future__ import annotations

from copy import copy

import pandas as pd
from aliprocessing.l1b.data import L1bImage, L1bSpectra
from skretrieval.core.sasktranformat import SASKTRANRadiance

from hawcsimulator.datastructures.viewinggeo import ObservationContainer


def l1b(
    observation: ObservationContainer,  # noqa: ARG001
    front_end_radiance: SASKTRANRadiance,
    l1b_mode: str,
    l1b_cfg_defaults: dict,
    l1b_cfg: dict | None = None,
) -> L1bImage:
    if l1b_cfg is None:
        l1b_cfg = {}

    try:
        import ali_l1
        from ali_l1.frontendradiance import ALIFER
    except ImportError:
        msg = "The ali_l1 package must be installed to use this simulator. Try pip install ali_l1 -f https://arg.usask.ca/wheels/"
        raise OSError(msg)  # noqa: B904

    model_options = copy(l1b_cfg_defaults)
    model_options.update(l1b_cfg)

    alifer = ALIFER(front_end_radiance._data)
    l1a = ali_l1.calculate_l1a_signals(alifer, mode=l1b_mode, **model_options)
    l1b = ali_l1.calculate_l1b_signals(l1a)
    l1b = ali_l1.l1b_to_xarray(l1b)

    l1b = l1b.isel(observation_set=0, across_track=2)
    l1b = l1b.interp(reference_altitude=l1b.altitude)

    geo = l1b.isel(wavelength=0)

    snr = l1b.radiance.values / l1b.radiance_uncertainty.values
    snr[snr > 200] = 200

    I = L1bSpectra.from_np_arrays(  # noqa: E741
        l1b.radiance.values,
        l1b.radiance.values / snr,
        l1b.altitude.values,
        geo.tangent_latitude.values,
        geo.tangent_longitude.values,
        l1b.wavelength.values,
        pd.to_datetime(geo.time.values).to_pydatetime(),
        float(geo.spacecraft_latitude),
        float(geo.spacecraft_longitude),
        float(geo.spacecraft_altitude),
        geo.tangent_solar_zenith_angle.values,
        geo.tangent_solar_azimuth_angle.values,
        geo.tangent_viewing_azimuth_angle.values,
    )

    dolp_error = l1b.degree_of_linear_polarization_uncertainty.values
    dolp_error[dolp_error < 0.003] = 0.003

    dolp = L1bSpectra.from_np_arrays(
        l1b.degree_of_linear_polarization.values,
        l1b.degree_of_linear_polarization_uncertainty.values,
        l1b.altitude.values,
        geo.tangent_latitude.values,
        geo.tangent_longitude.values,
        l1b.wavelength.values,
        pd.to_datetime(geo.time.values).to_pydatetime(),
        float(geo.spacecraft_latitude),
        float(geo.spacecraft_longitude),
        float(geo.spacecraft_altitude),
        geo.tangent_solar_zenith_angle.values,
        geo.tangent_solar_azimuth_angle.values,
        geo.tangent_viewing_azimuth_angle.values,
    )

    return L1bImage({"I": I, "dolp": dolp})
