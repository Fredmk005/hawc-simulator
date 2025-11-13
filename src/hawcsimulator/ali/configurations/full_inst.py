from __future__ import annotations

import numpy as np
import sasktran2 as sk
from aliprocessing.l2.optical import aerosol_median_radius_db

import hawcsimulator.ali.steps.fer as fer
import hawcsimulator.ali.steps.full_inst as full_inst
import hawcsimulator.ali.steps.l2 as l2
import hawcsimulator.ali.steps.por as por
from hawcsimulator.simulator import Simulator


class ALIPhase0Simulator(Simulator):
    def __init__(self) -> None:
        super().__init__()
        self._modules.append(fer)
        self._modules.append(full_inst)
        self._modules.append(por)
        self._modules.append(l2)

    def _initialize_data(self) -> dict:
        data = {}

        data["calibration_database"] = None

        data["viewing_tangent_altitudes"] = np.arange(-500.0, 50001, 500.0)
        data["observer_altitude"] = 450000.0

        data["aerosol_optical_property"] = aerosol_median_radius_db()
        data["aerosol_kwargs"] = {"extinction_wavelength_nm": 745.0}

        data["constituents"] = {
            "solar_irradiance": sk.constituent.SolarIrradiance(
                mode="average", resolution=1.0
            )
        }

        data["sample_wavelengths"] = np.array(
            [610, 676, 755, 869, 950, 1022, 1080, 1225, 1360, 1450, 1560]
        )

        data["l1b_cfg_defaults"] = {
            "detector": "CARDINAL1280:gain=2",
            "throughput_model_version": "P0-PUBLIC-V0.0.5",
            "wavelength_names": [str(w) for w in data["sample_wavelengths"]],
            "num_horizontal_profiles": 5,
        }

        data["l1b_mode"] = "LOW"

        data["wavelength_resolution"] = 0.5

        return data
