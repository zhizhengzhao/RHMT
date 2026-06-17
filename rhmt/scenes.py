"""Benchmark scene definitions and truth outlines (figures/scoring only)."""
import numpy as np

# One centered U-shaped block per scene (matches sim/g4/config/u.poly): a 150 mm
# outer square minus a 60 mm top-centre notch -- 45 mm arms and base, open at +y.
# 3 materials x 2 backgrounds = 6 object scenes.
_MATS = ["water", "pb", "al"]

SCENES = {                       # scene -> (background, [objects])
    "blank_sio2":     ("sio2", []),
    "blank_concrete": ("concrete", []),
}
for _bg in ("sio2", "concrete"):
    for _m in _MATS:
        SCENES[f"u_{_m}_{_bg}"] = (_bg, [dict(mat=_m, kind="ushape", x=0, y=0)])

MAT = {
    "sio2":     dict(s=1.158, inv_x0=1.0 / 116.6),   # G4_SILICON_DIOXIDE 2.32
    "concrete": dict(s=1.171, inv_x0=1.0 / 115.5),   # G4_CONCRETE 2.3
    "water":    dict(s=0.555, inv_x0=1.0 / 360.8),
    "pb":       dict(s=4.492, inv_x0=1.0 / 5.612),
    "al":       dict(s=1.301, inv_x0=1.0 / 88.97),
    "air":      dict(s=0.0015, inv_x0=1.0 / 303900.0),
}


def _mask_of(obj, gx, gy):
    xx, yy = np.meshgrid(gx, gy)
    if obj["kind"] == "square":
        return ((np.abs(xx - obj["x"]) <= obj["r"])
                & (np.abs(yy - obj["y"]) <= obj["r"]))
    if obj["kind"] == "ushape":
        x, y = xx - obj["x"], yy - obj["y"]
        outer = (np.abs(x) <= 75) & (np.abs(y) <= 75)
        notch = (np.abs(x) <= 30) & (y >= -30)       # open top -> remove top-centre
        return outer & ~notch
    raise ValueError(obj["kind"])


def object_masks(scene, gx, gy):
    """List of (material, boolean mask) for each object on grid centres gx/gy."""
    return [(o["mat"], _mask_of(o, gx, gy)) for o in SCENES[scene][1]]


def truth_mask(scene, gx, gy):
    """Footprint of the scene's object; None for blank scenes."""
    objs = SCENES[scene][1]
    if not objs:
        return None
    m = np.zeros((len(gy), len(gx)), bool)
    for _, om in object_masks(scene, gx, gy):
        m |= om
    return m
