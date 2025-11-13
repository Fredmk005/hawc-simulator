from __future__ import annotations

from showlib.cal_db import CalibrationDatabase
from showlib.l1a.data import L1AImage
from showlib.l1b.data import L1bDataSet
from showlib.processing.l1a_to_l1b import process_l1a_to_l1b_dataset


def l1b(
    l1a: L1AImage,
    calibration_database: CalibrationDatabase,
    l1b_cfg: dict | None = None,
) -> L1bDataSet:
    if l1b_cfg is None:
        l1b_cfg = {}

    image = process_l1a_to_l1b_dataset(l1a, calibration_database, **l1b_cfg)

    return L1bDataSet.from_image(image)
