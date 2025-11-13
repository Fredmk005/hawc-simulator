from __future__ import annotations

from showlib.l1a.data import L1AImage
from skretrieval.core.sasktranformat import SASKTRANRadiance


def l1a(
    front_end_radiance: SASKTRANRadiance,
    l1a_cfg: dict | None = None,
) -> L1AImage:
    if l1a_cfg is None:
        l1a_cfg = {}

    try:
        from showinstrument.run import generate_l1a_image
    except ImportError:
        msg = "To use the full SHOW simulator the showinstrument package must be installed"
        raise OSError(msg)  # noqa: B904

    return generate_l1a_image(front_end_radiance._data, **l1a_cfg)
