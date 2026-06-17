"""Runs one (method, scene, cap) cell and saves the reconstruction to results/cells/.

Usage: python experiments/run_cell.py --method rht_s --scene u_pb_sio2 [--cap 50000] [--seed 7] [--device cuda:1]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from rhmt import data
from rhmt.methods import get
from rhmt.scenes import SCENES

X0S = {"sio2": 122.9, "concrete": 115.3}
SIGMA_POS = 0.05      # mm
SCORE = 6.0           # common grid: field pitch, scoring, display


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--cap", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vox", type=float, default=0.0)
    a = ap.parse_args()

    os.makedirs("cal", exist_ok=True)
    os.makedirs("results/cells", exist_ok=True)
    method = get(a.method, dev=a.device)
    bg = SCENES[a.scene][0]
    fvox = a.vox or SCORE

    calf = f"cal/{a.method}_{bg}.npz"
    if os.path.exists(calf):
        cal = dict(np.load(calf, allow_pickle=False))
    else:
        bh, blz = data.load(method.tier, f"blank_{bg}", seed=1)
        cal = method.calibrate(bh, blz, SIGMA_POS, X0S[bg])
        np.savez(calf, **{k: np.asarray(v) for k, v in cal.items()})
        cal = dict(np.load(calf, allow_pickle=False))

    hits, lz = data.load(method.tier, a.scene, cap=a.cap, seed=a.seed)
    r = method.reconstruct(hits, lz, SIGMA_POS, cal, fvox, x0_sample=X0S[bg],
                           img_vox=SCORE)
    tag = f"{a.method}_{a.scene}_cap{a.cap}_s{a.seed}"
    out = {"method": a.method, "scene": a.scene, "cap": a.cap,
           "seed": a.seed, "vox": fvox, "n": len(hits)}
    json.dump(out, open(f"results/cells/{tag}.json", "w"))
    arrs = {k: v for k, v in r.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(f"results/cells/img_{tag}.npz", **arrs)
    print(f"[{tag}] n={len(hits)} done")


if __name__ == "__main__":
    main()
