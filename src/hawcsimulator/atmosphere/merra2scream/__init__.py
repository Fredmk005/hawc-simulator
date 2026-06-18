from __future__ import annotations

import xarray 
import numpy as np
from pathlib import Path
from astropy.coordinates import EarthLocation, get_sun, AltAz
from astropy.time import Time
import astropy.units as u

from hawcsimulator.appconfig import load_user_config

def _merra2scream_folder():
  cfg = load_user_config()

  if "merra2scream_folder" in cfg:
      return Path(cfg["merra2scream_folder"]).expanduser()
  msg = "No merra-2 scream folder specified the user config. Add 'merra2scream_folder' to the user config."
  raise ValueError(msg)

def _access_MERRA2SCREAMDatasets(earthaccess_folder=None):
  if earthaccess_folder = None:
    earthaccess_folder = _merra2scream_folder()
  files = sorted(
      f for f in Path(earthaccess_folder).iterdir()
      if f.is_file()
  )

  datasets = [
      xr.open_dataset(f, engine="h5netcdf")
       for f in files
   ]

   times = [
      pd.to_datetime(
           f.name.split(".")[-1],
          format="%Y%m%d_%H%Mz",
          utc=True,
      )
      for f in files
  ]

  return {"ds": datasets, "times": times}

def _latlon(ref_lat, ref_lon, ds):
  """
  Returns nearest lattitude, longitude from user input
  """
  lon_grid = ds.coords["lon"].values
  lat_grid = ds.coords["lat"].values

  lon = ((lon + 180) % 360) - 180

  lon_idx = np.argmin(np.abs(lon_grid - lon))
  lat_idx = np.argmin(np.abs(lat_grid - lat))

  return {
      "lon_idx": int(lon_idx),
      "lat_idx": int(lat_idx),
      "lon": lon_grid[lon_idx],
      "lat": lat_grid[lat_idx]
  }  

def _PTA_grid(longitude_idx,latitude_idx,dataset):
    """
    Creates Pressure, temperature, altitude grid. returns dict with pressure, temp and altitude
    """
    altitude_m_arr = dataset["H"].isel(lat=latitude_idx,lon=longitude_idx, lev=slice(None,None,-1)).values
    pressure_Pa_arr = dataset["PL"].isel(lat=latitude_idx,lon=longitude_idx, lev=slice(None,None,-1)).values
    temperature_K_arr = dataset["T"].isel(lat=latitude_idx,lon=longitude_idx, lev=slice(None,None,-1)).values
    pressure_Pa = np.ascontiguousarray(np.squeeze(pressure_Pa_arr), dtype=np.float64)
    temperature_K = np.ascontiguousarray(np.squeeze(temperature_K_arr), dtype=np.float64)
    altitude_m = np.ascontiguousarray(np.squeeze(altitude_m_arr), dtype=np.float64)
    return dict(pressure=pressure_Pa, temp=temperature_K, altitude=altitude_m)

def _species_vmr(longitude_idx,latitude_idx,dataset):
    """
    Creates ozone, water vapor, nitric acid, hydrochloric and Nitrous Oxide vmr profiles as O3, HNO3, N2O, HCL, and H2O
    """
    spec_hum_arr = dataset["QV"].isel(lat=latitude_idx,lon=longitude_idx, lev=slice(None,None,-1)).values
    spec_hum = np.ascontiguousarray(np.squeeze(spec_hum_arr), dtype=np.float64)
    M_v=18.01528
    M_q=28.9644
    epsilon = M_v/M_q
    vmr_profile_h2o = spec_hum /(epsilon + spec_hum*(1.0-epsilon))
    o3_vmr_arr = dataset["O3"].isel(lat=latitude_idx,lon=longitude_idx, lev=slice(None,None,-1)).values
    hno3_vmr_arr = dataset["HNO3"].isel(lat=latitude_idx,lon=longitude_idx, lev=slice(None,None,-1)).values
    hcl_vmr_arr = dataset["HCL"].isel(lat=latitude_idx,lon=longitude_idx, lev=slice(None,None,-1)).values
    n2o_vmr_arr = dataset["N2O"].isel(lat=latitude_idx,lon=longitude_idx, lev=slice(None,None,-1)).values
    o3_vmr = np.ascontiguousarray(np.squeeze(o3_vmr_arr), dtype=np.float64)
    hno3_vmr = np.ascontiguousarray(np.squeeze(hno3_vmr_arr), dtype=np.float64)
    hcl_vmr = np.ascontiguousarray(np.squeeze(hcl_vmr_arr), dtype=np.float64)
    n2o_vmr = np.ascontiguousarray(np.squeeze(n2o_vmr_arr), dtype=np.float64)
    return dict(O3=o3_vmr*1e-6,HNO3=hno3_vmr,N2O=n2o_vmr,HCL=hcl_vmr,H2O=vmr_profile_h2o)
