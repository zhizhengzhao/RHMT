"""Blank-scan flat-field calibration for the NAMT-E energy channel."""
import numpy as np
import torch

NPBIN = 8
FF_VOX = 25.0
HALF = 200.0
HUBER_DELTA = 1.345


def _huber_loc(v, sig, delta=HUBER_DELTA, iters=12):
    """Huber-M location of a 1-D sample at fixed scale sig (IRLS)."""
    loc = np.median(v)
    for _ in range(iters):
        r = np.abs(v - loc)
        w = np.minimum(1.0, delta * sig / np.maximum(r, 1e-12))
        loc = np.sum(w * v) / np.sum(w)
    return loc


def _huber_loc_cells(v, cell, ncell, sig, init, delta=HUBER_DELTA, iters=12):
    """Per-cell Huber-M locations, vectorised IRLS at fixed scale."""
    loc = init.copy()
    for _ in range(iters):
        r = np.abs(v - loc[cell])
        w = np.minimum(1.0, delta * sig / np.maximum(r, 1e-12))
        sw = np.bincount(cell, weights=w, minlength=ncell)
        swv = np.bincount(cell, weights=w * v, minlength=ncell)
        upd = sw > 0
        loc[upd] = swv[upd] / sw[upd]
    return loc


def lam_table(fit, pq=None):
    """Build per-(p-bin, x, y) Huber location and MAD width table from a blank fit."""
    ok = fit["ok"].cpu().numpy()
    lam0 = fit["lam"].cpu().numpy()[ok]
    p = fit["p_in"].cpu().numpy()[ok]
    xy = fit["xy"][ok].cpu().numpy()
    good = (p > 300) & (p < 60000) & np.isfinite(lam0) & (lam0 < p - 50)
    lam = np.log(p[good] / (p[good] - lam0[good]))
    p, xy = p[good], xy[good]
    if pq is None:
        pq = np.quantile(p, np.linspace(0, 1, NPBIN + 1))
        pq[0], pq[-1] = 0.0, 1e9
    nb = int(2 * HALF / FF_VOX)
    tabs = np.zeros((NPBIN, nb, nb))
    sigs = np.zeros(NPBIN)
    for b in range(NPBIN):
        m = (p >= pq[b]) & (p < pq[b + 1])
        ix = np.clip(((xy[m, 0] + HALF) / FF_VOX).astype(int), 0, nb - 1)
        iy = np.clip(((xy[m, 1] + HALF) / FF_VOX).astype(int), 0, nb - 1)
        v = lam[m]
        med = np.median(v)
        sigs[b] = max(1.4826 * np.median(np.abs(v - med)), 1e-6)
        cell = iy * nb + ix
        cnt = np.bincount(cell, minlength=nb * nb)
        glob = _huber_loc(v, sigs[b])
        loc = _huber_loc_cells(v, cell, nb * nb, sigs[b],
                               np.full(nb * nb, glob))
        tabs[b] = np.where(cnt >= 20, loc, glob).reshape(nb, nb)
    return dict(pq=pq, tabs=tabs, sigs=sigs, nb=nb)


def flat_lookup(table, p, xy):
    pb = np.clip(np.searchsorted(table["pq"], p) - 1, 0, NPBIN - 1)
    nb = int(table["nb"])
    ix = np.clip(((xy[:, 0] + HALF) / FF_VOX).astype(int), 0, nb - 1)
    iy = np.clip(((xy[:, 1] + HALF) / FF_VOX).astype(int), 0, nb - 1)
    return table["tabs"][pb, iy, ix], table["sigs"][pb]


def chord_nodes(fit_sub, nq=12):
    """Sample-chord (x, y) positions at nq Gauss-Legendre nodes."""
    ins = fit_sub["ins"]
    bot, top = ins["sample"]
    xg, wg = np.polynomial.legendre.leggauss(nq)
    zq = 0.5 * (top + bot) + 0.5 * (top - bot) * xg
    wq = 0.5 * (top - bot) * wg
    st = fit_sub["state_mb"]
    zq_t = torch.as_tensor(zq, dtype=torch.float64, device=st.device)
    x = st[:, 0:1] - st[:, 2:3] * (fit_sub["z_mb"] - zq_t)[None, :]
    y = st[:, 1:2] - st[:, 3:4] * (fit_sub["z_mb"] - zq_t)[None, :]
    return torch.stack([x, y], -1), torch.as_tensor(
        wq, dtype=torch.float64, device=st.device)
