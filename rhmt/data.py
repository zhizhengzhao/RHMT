"""Dataset loading with optional subsampling and position smearing."""
import numpy as np

DATA = "data"


def load(tier, scene, cap=0, smear=0.0, seed=7):
    """Load hits for the given tier/scene. Returns (hits, layer_z)."""
    d = np.load(f"{DATA}/{tier}_{scene}.npz")
    nlay = len(d["layer_z"])
    h = d["hits"][d["nhit"] == nlay].astype(np.float64)
    rng = np.random.default_rng(seed)
    if cap and cap < len(h):
        h = h[rng.choice(len(h), cap, replace=False)]
    if smear > 0:
        h = h + rng.normal(0.0, smear, h.shape)
    return h, d["layer_z"]
