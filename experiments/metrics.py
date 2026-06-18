"""Computes ROC-AUC and writes the LaTeX table bodies to results/tables/."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from experiments.matrix import CAPS
from rhmt.scenes import MAT, truth_mask

ROWS = [
    ("MLS-EM",          "mlsem",  "img",     "inv_x0"),
    ("PoCA",            "poca",   "img",     "inv_x0"),
    ("ASR",             "asr",    "img",     "inv_x0"),
    ("MLS-EM + momentum (6-plane)", "mlsem_p", "img", "inv_x0"),
    ("PoCA + momentum (6-plane)", "poca_p", "img", "inv_x0"),
    ("ASR + momentum (6-plane)", "asr_p", "img", "inv_x0"),
    ("RHMT-S (3-plane)", "rht_3p", "img",     "inv_x0"),
    ("RHMT-S (4-plane)", "rht_s",  "img",     "inv_x0"),
    ("RHMT-E (6-plane)", "rht_6p", "img_s",   "s"),
]
LABELS = [r[0] for r in ROWS]
GROUPS = [("Baselines (4-plane tracker)", LABELS[:3]),
          ("Scattering baselines + momentum", LABELS[3:6]),
          ("Ours\\,---\\,RHMT", LABELS[6:])]
# main-paper lineup: baselines + tracker scattering (3- and 4-plane) + energy
MAIN = ["PoCA", "ASR", "MLS-EM", "RHMT-S (3-plane)", "RHMT-S (4-plane)", "RHMT-E (6-plane)"]
MAIN_GROUPS = [("Baselines (4-plane tracker)", MAIN[:3]), ("Ours\\,---\\,RHMT", MAIN[3:])]
MATS = ["pb", "water", "al"]    # display order: lead, water, aluminium
BGS = ["sio2", "concrete"]


def _prep(img, gx, gy, scene, sgn):
    """Sign-oriented anomaly, U truth mask, and pure-background mask for one scene."""
    truth = truth_mask(scene, gx, gy)
    grow = np.zeros_like(truth)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            grow |= np.roll(np.roll(truth, i, 0), j, 1)
    bg = np.isfinite(img) & ~grow
    a = sgn * (img - np.median(img[bg]))
    return a, truth, bg


def roc_auc(img, gx, gy, scene, sgn):
    """Mann-Whitney AUC separating the U footprint from pure background."""
    a, truth, bg = _prep(img, gx, gy, scene, sgn)
    pos = a[truth & np.isfinite(a)]
    neg = a[bg]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = np.empty(allv.size)
    ranks[allv.argsort()] = np.arange(1, allv.size + 1)
    u = ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def cell(method, imgkey, mat, bg, cap, obs):
    scene = f"u_{mat}_{bg}"
    f = f"results/cells/img_{method}_{scene}_cap{cap}_s7.npz"
    if not os.path.exists(f):
        return float("nan")
    d = np.load(f)
    if imgkey not in d:
        return float("nan")
    sgn = np.sign(MAT[mat][obs] - MAT[bg][obs])
    return roc_auc(d[imgkey], d["gx"], d["gy"], scene, sgn)


def hl(vals):
    """Mark the top three distinct values: 1st \\best, 2nd \\snd, 3rd \\thd."""
    rank = sorted({v for v in vals if np.isfinite(v)}, reverse=True)
    b = rank[0] if len(rank) > 0 else None
    s = rank[1] if len(rank) > 1 else None
    t = rank[2] if len(rank) > 2 else None
    out = []
    for v in vals:
        if not np.isfinite(v):
            out.append("--")
        elif v == b:
            out.append(f"\\best{{{v:.2f}}}")
        elif v == s:
            out.append(f"\\snd{{{v:.2f}}}")
        elif v == t:
            out.append(f"\\thd{{{v:.2f}}}")
        else:
            out.append(f"{v:.2f}")
    return out


def plain(vals):
    return ["--" if not np.isfinite(v) else f"{v:.2f}" for v in vals]


def _decorate(label):
    """Annotate each method with its detector: a magnet-free tracker or the
    momentum-measuring magnetic spectrometer."""
    return (label.replace("(3-plane)", "(3-plane tracker)")
                 .replace("(4-plane)", "(4-plane tracker)")
                 .replace("(6-plane)", "(6-plane spectrometer)"))


def _auc_body(labels, groups, g):
    """Per-material AUC body for the given row set; mean column ranked within the set."""
    span = 2 * len(MATS) + 4
    means = [np.mean([g(lab, mat, bg, 0) for bg in BGS for mat in MATS]) for lab in labels]
    meanhl = hl(means)
    body = []
    for gname, glabels in groups:
        body.append(f"\\multicolumn{{{span}}}{{@{{}}l}}{{\\textit{{{gname}}}}} \\\\")
        for lab in glabels:
            i = labels.index(lab)
            s = plain([g(lab, mat, "sio2", 0) for mat in MATS])
            c = plain([g(lab, mat, "concrete", 0) for mat in MATS])
            body.append(f"\\quad {_decorate(lab)} & " + " & ".join(s) + " & & "
                        + " & ".join(c) + " & & " + meanhl[i] + r" \\")
    return "\n".join(body) + "\n"


def _ladder_body(labels, groups, g):
    """Mean-AUC vs exposure body for the given row set; each column ranked within the set."""
    capcol = {cap: hl([np.mean([g(lab, mat, bg, cap) for bg in BGS for mat in MATS])
                       for lab in labels]) for cap in CAPS}
    body = []
    for gname, glabels in groups:
        body.append(f"\\multicolumn{{{1 + len(CAPS)}}}{{@{{}}l}}{{\\textit{{{gname}}}}} \\\\")
        for lab in glabels:
            i = labels.index(lab)
            body.append(f"\\quad {_decorate(lab)} & "
                        + " & ".join(capcol[cap][i] for cap in CAPS) + r" \\")
    return "\n".join(body) + "\n"


def main():
    full = {}
    for label, m, imgkey, obs in ROWS:
        for bg in BGS:
            for mat in MATS:
                for cap in CAPS:
                    v = cell(m, imgkey, mat, bg, cap, obs)
                    if np.isfinite(v):
                        full[(label, mat, bg, cap)] = v
    os.makedirs("results/tables", exist_ok=True)

    def g(label, mat, bg, cap):
        return full.get((label, mat, bg, cap), float("nan"))

    print("=== AUC (full exposure) ===")
    for label in LABELS:
        row = "   ".join(f"{mat[:2]}.{bg[:2]}:{g(label, mat, bg, 0):.2f}"
                         for bg in BGS for mat in MATS)
        print(f"{label:16s} {row}")

    T = "results/tables"
    open(f"{T}/auc_main_body.tex", "w").write(_auc_body(MAIN, MAIN_GROUPS, g))
    open(f"{T}/auc_body.tex", "w").write(_auc_body(LABELS, GROUPS, g))
    open(f"{T}/ladder_main_body.tex", "w").write(_ladder_body(MAIN, MAIN_GROUPS, g))
    open(f"{T}/ladder_body.tex", "w").write(_ladder_body(LABELS, GROUPS, g))

    caps_hdr = {0: "full ($\\sim$125k)", 50000: "50k", 15000: "15k"}
    open(f"{T}/ladder_header.tex", "w").write(
        "method & " + " & ".join(caps_hdr.get(c, f"{c}") for c in CAPS) + r" \\" + "\n")
    print("-> wrote auc_main / auc / ladder_main / ladder bodies + ladder_header")


if __name__ == "__main__":
    main()
