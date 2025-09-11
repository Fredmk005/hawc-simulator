from __future__ import annotations

import abc

import numpy as np
import xarray as xr


class NoiseModel(abc.ABC):
    @abc.abstractmethod
    def calc_noise(self, signal: xr.DataArray, **kwargs):
        pass


class ConstantNoise(NoiseModel):
    def __init__(self, noise_level: float):
        self._noise_level = noise_level

    def calc_noise(self, signal: xr.DataArray, **kwargs):

        noise_sigma = signal * self._noise_level

        add_noise = np.random.default_rng().normal(0, noise_sigma.to_numpy())

        return (signal + add_noise, noise_sigma)


class ALINoiseModel(NoiseModel):
    def __init__(
        self,
        calibration_noise_level: float = 0.005,
        straylight_reference_alt: float = 45000,
        stray_light_reference_wavelength: float = 1020,
        straylight_fraction: float = 0.02,
        seed=None,
    ):
        self._calibration_noise_level = calibration_noise_level
        self._straylight_reference_alt = straylight_reference_alt
        self._straylight_reference_wavelength = stray_light_reference_wavelength
        self._straylight_fraction = straylight_fraction
        self._seed = seed

    def calc_noise(self, signal: xr.DataArray, **kwargs):
        noise_sigma = signal * self._calibration_noise_level

        fer = kwargs["fer"]

        fer_swapped = fer.swap_dims(
            {"los": "tangent_altitude", "spectral_grid": "wavelength_nm"}
        ).isel(stokes=0)

        straylight_amount = fer_swapped.interp(
            tangent_altitude=self._straylight_reference_alt
        )["radiance"]

        straylight_scale = (
            straylight_amount
            / fer_swapped.interp(tangent_altitude=500.0)["radiance"]
            / fer_swapped["wavelength_nm"] ** 2
        )
        straylight_scale /= straylight_scale.interp(
            wavelength_nm=self._straylight_reference_wavelength
        )

        straylight_signal = (
            straylight_amount / straylight_scale * self._straylight_fraction
        )

        signal_with_stray = (
            signal.swap_dims({"spectral_grid": "wavelength_nm"}) + straylight_signal
        ).swap_dims({"wavelength_nm": "spectral_grid"})
        add_noise = np.random.default_rng(self._seed).normal(0, noise_sigma.to_numpy())

        return (signal_with_stray + add_noise, noise_sigma)
