"""Reconstruct at one position-smear sigma and score mean ROC-AUC.
  python experiments/robustness.py --sigma 0.5 --device cuda:0
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from rhmt import data
from rhmt.methods import get
from rhmt.scenes import MAT, SCENES
from experiments.metrics import ROWS, roc_auc, MATS, BGS
from experiments.run_cell import SCORE, X0S

LABELS = [r[0] for r in ROWS]
SCENES_USED = [f"u_{m}_{bg}" for bg in BGS for m in MATS]


def scene_auc(img, gx, gy, scene, obs):
    bg = SCENES[scene][0]
    mat = SCENES[scene][1][0]["mat"]
    return roc_auc(img, gx, gy, scene, np.sign(MAT[mat][obs] - MAT[bg][obs]))


def run_sigma(sigma, dev):
    per = {l: [] for l in LABELS}
    for scene in SCENES_USED:
        bg = SCENES[scene][0]
        x0 = X0S[bg]
        recon = {}
        for label, mkey, imgkey, obs in ROWS:
            if mkey not in recon:
                m = get(mkey, dev=dev)
                bh, blz = data.load(m.tier, f"blank_{bg}", smear=sigma, seed=1)
                cal = m.calibrate(bh, blz, sigma, x0)
                h, lz = data.load(m.tier, scene, smear=sigma, seed=7)
                recon[mkey] = m.reconstruct(h, lz, sigma, cal, SCORE,
                                            x0_sample=x0, img_vox=SCORE)
            r = recon[mkey]
            per[label].append(scene_auc(r[imgkey], r["gx"], r["gy"], scene, obs))
    return {l: float(np.mean(v)) for l, v in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out = run_sigma(a.sigma, a.device)
    os.makedirs("results/robust", exist_ok=True)
    json.dump({"sigma": a.sigma, "auc": out},
              open(f"results/robust/sigma_{a.sigma:g}.json", "w"))
    print(f"sigma {a.sigma:g}: " + "  ".join(f"{l}={out[l]:.3f}" for l in LABELS), flush=True)


if __name__ == "__main__":
    main()
