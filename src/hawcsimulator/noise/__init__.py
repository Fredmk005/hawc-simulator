from __future__ import annotations

import abc

import numpy as np
import xarray as xr


class NoiseModel(abc.ABC):
    @abc.abstractmethod
    def calc_noise(self, signal: xr.DataArray, **kwargs):
        pass

    def calc_systematic(self, signal: xr.DataArray, **kwargs):
        return (signal, xr.zeros_like(signal))


class ConstantNoise(NoiseModel):
    def __init__(self, noise_level: float, systematics: dict | None = None):
        if systematics is None:
            self._systematics = {}
        else:
            self._systematics = systematics
        self._noise_level = noise_level

    def calc_noise(self, signal: xr.DataArray, **kwargs):

        noise_sigma = signal * self._noise_level

        add_noise = np.random.default_rng().normal(0, noise_sigma.to_numpy())

        return (signal + add_noise, noise_sigma)

    def calc_systematic(self, signal: xr.DataArray, **kwargs):
        signal_name = kwargs.get("name")

        if signal_name is None:
            return super().calc_systematic(signal, **kwargs)

        if signal_name not in self._systematics:
            return super().calc_systematic(signal, **kwargs)

        systematic_sigma = self._systematics[signal_name] * xr.ones_like(signal)

        add_systematic = np.random.default_rng().normal(0, systematic_sigma.to_numpy())

        return (signal + add_systematic, systematic_sigma)


class ALINoiseModel(NoiseModel):
    def __init__(
        self,
        calibration_noise_level: float = 0.005,
        straylight_reference_alt: float = 45000,
        stray_light_reference_wavelength: float = 1020,
        straylight_fraction: float = 0.02,
        seed=None,
    ):
        # ALI MRD SNR, nominal from 600 - 1020
        self._snrs = np.array([400, 400, 200, 200, 100, 100, 50, 50, 20, 20])
        self._snr_alts = [
            -10000.0,
            20000,
            20000.001,
            25000,
            25000.001,
            30000,
            30000.001,
            35000,
            35000.001,
            100000,
        ]

        # outside range factor
        # will multiply noise by this factor outside the range
        self._outside_range_factor = 2.0

        # SNR applies to I, so on each image the SNR can be worse by a factor of
        # 1 / sqrt(num_meas)
        self._num_meas = 3

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

        alt_snr = (
            xr.DataArray(self._snrs, coords={"tangent_altitude": self._snr_alts})
            .interp(tangent_altitude=fer_swapped.tangent_altitude.values)
            .expand_dims({"spectral_grid": signal.spectral_grid})
            .copy()
        )

        alt_snr.values[signal.wavelength_nm < 600, :] /= self._outside_range_factor
        alt_snr.values[signal.wavelength_nm > 1050, :] /= self._outside_range_factor

        alt_snr /= np.sqrt(self._num_meas)

        alt_snr = alt_snr.rename({"tangent_altitude": "los"})
        alt_snr = alt_snr.assign_coords({"los": signal.los})

        noise_from_snr = signal / alt_snr

        # Add calibration noise in quadrature with SNR noise
        noise_sigma = np.sqrt(noise_sigma**2 + noise_from_snr**2)

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
