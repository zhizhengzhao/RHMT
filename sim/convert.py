"""NAMT converter: ROOT -> NPZ keeping raw RPC crossings on ALL planes (any count).

Per event: hits (N,nlay,2) = muon crossing (x,y) per plane, nhit, src_p4 (TRUTH,
eval-only, never a method input), layer_z. Works for the 4-layer (tier A) and
6-layer (tier B) instruments alike.
"""
import argparse
import glob as globmod
from multiprocessing import Pool

import numpy as np
import uproot


def _get_layerz(f):
    lz = f["params"]["Params/Params.LayerZ"].array(library="np")[0]
    while hasattr(lz, "__len__") and len(lz) == 1 and hasattr(lz[0], "__len__"):
        lz = lz[0]
    out = []
    for v in lz:
        out.extend([float(x) for x in v] if hasattr(v, "__len__") else [float(v)])
    return np.sort(np.array(out))[::-1]  # top -> bottom


def process_one(fp):
    try:
        fo = uproot.open(fp)
        lz = _get_layerz(fo)
        t = fo["tree"]
        n = t.num_entries
        nlay = len(lz)
        if n == 0 or nlay < 4:
            return None
        a = t.arrays(["Edeps.Id", "Edeps.X", "Edeps.Y", "Edeps.trackID",
                      "Event.E", "Event.Px", "Event.Py", "Event.Pz"], library="np")
    except Exception as e:  # noqa: BLE001
        return f"ERR {fp}: {e}"
    hits = np.full((n, nlay, 2), np.nan, np.float32)
    nhit = np.zeros(n, np.int8)
    src_p4 = np.zeros((n, 4), np.float32)
    keep = np.zeros(n, bool)
    for i in range(n):
        if len(a["Event.E"][i]) == 0:
            continue
        ids = np.asarray(a["Edeps.Id"][i], np.int64)
        tid = np.asarray(a["Edeps.trackID"][i], np.int64)
        xs = np.asarray(a["Edeps.X"][i], np.float64)
        ys = np.asarray(a["Edeps.Y"][i], np.float64)
        cnt = 0
        for lay in range(nlay):
            sel = np.where((ids == lay) & (tid == 1))[0]
            if len(sel):
                hits[i, lay, 0] = xs[sel[0]]
                hits[i, lay, 1] = ys[sel[0]]
                cnt += 1
        if cnt == 0:
            continue
        nhit[i] = cnt
        src_p4[i] = [a["Event.E"][i][0], a["Event.Px"][i][0],
                     a["Event.Py"][i][0], a["Event.Pz"][i][0]]
        keep[i] = True
    idx = np.where(keep)[0]
    if len(idx) == 0:
        return None
    return dict(hits=hits[idx], nhit=nhit[idx], src_p4=src_p4[idx], layer_z=lz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    files = sorted(globmod.glob(a.glob))
    assert files, "no files match"
    with Pool(min(a.workers, len(files))) as pool:
        res = pool.map(process_one, files)
    errs = [r for r in res if isinstance(r, str)]
    res = [r for r in res if isinstance(r, dict)]
    for e in errs[:5]:
        print(e)
    assert res, "nothing converted"
    lz = res[0]["layer_z"]
    merged = {k: np.concatenate([r[k] for r in res]) for k in ["hits", "nhit", "src_p4"]}
    merged["layer_z"] = lz
    full = int((merged["nhit"] == len(lz)).sum())
    np.savez_compressed(a.out, **merged)
    print(f"wrote {len(merged['nhit'])} events ({full} full-{len(lz)}) -> {a.out}")
    print("layer z:", lz)


if __name__ == "__main__":
    main()
