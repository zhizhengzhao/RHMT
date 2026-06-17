"""Classical scattering-tomography baselines: PoCA, ASR, MLS-EM."""
import numpy as np

from rhmt.inversion import grid_axes
from rhmt.methods.base import Method

HALF = 200.0


def derived(h, lz):
    """Per-muon squared scattering angle and PoCA point."""
    t_up = (h[:, 1] - h[:, 0]) / (lz[1] - lz[0])     # (N,2)
    t_dn = (h[:, 3] - h[:, 2]) / (lz[3] - lz[2])
    th2 = ((np.arctan(t_dn) - np.arctan(t_up)) ** 2).sum(1)
    num = (h[:, 2] - h[:, 1]) + t_dn * (lz[1] - lz[2])
    dt = t_up - t_dn
    with np.errstate(divide="ignore", invalid="ignore"):
        zpoca = num / np.where(np.abs(dt) < 1e-12, np.nan, dt)
    zp = lz[1] + np.nanmean(zpoca, axis=1)
    zp = np.clip(np.nan_to_num(zp, nan=0.5 * (lz[1] + lz[2])), lz[2], lz[1])
    xy_poca = h[:, 1] + t_up * (zp[:, None] - lz[1])
    return th2, xy_poca


def grid_index(xy, vox):
    n = int(2 * HALF / vox)
    ix = np.clip(((xy[:, 0] + HALF) / vox).astype(int), 0, n - 1)
    iy = np.clip(((xy[:, 1] + HALF) / vox).astype(int), 0, n - 1)
    return iy * n + ix, n


def img_from_stat(idx, n, vals, min_cov=25, stat="median"):
    order = np.argsort(idx)
    idx_s, val_s = idx[order], vals[order]
    bounds = np.searchsorted(idx_s, np.arange(n * n + 1))
    img = np.full(n * n, np.nan)
    for k in range(n * n):
        a, b = bounds[k], bounds[k + 1]
        if b - a >= min_cov:
            seg = val_s[a:b]
            img[k] = np.median(seg) if stat == "median" else seg.mean()
    return img.reshape(n, n)


def poca_img(h, lz, vox):
    th2, xy = derived(h, lz)
    idx, n = grid_index(xy, vox)
    return img_from_stat(idx, n, th2, stat="median"), n


def asr_img(h, lz, vox):
    """Angle Statistics Reconstruction."""
    th2, _ = derived(h, lz)
    v = np.sqrt(th2)
    v = np.clip(v, None, np.quantile(v, 0.995))
    t_up = (h[:, 1] - h[:, 0]) / (lz[1] - lz[0])
    ks = 8
    zs = np.linspace(lz[2] + 10, lz[1] - 10, ks)
    imgs = []
    idxs = []
    vals = []
    for z in zs:
        xy = h[:, 1] + t_up * (z - lz[1])
        idx, n = grid_index(xy, vox)
        idxs.append(idx)
        vals.append(v)
    idx = np.concatenate(idxs)
    vals = np.concatenate(vals)
    return img_from_stat(idx, n, vals, min_cov=25 * ks // 2, stat="median"), n


def mlsem_img(h, lz, vox, iters=40):
    """MLS-EM (Schultz et al.) in 2D with nominal momentum."""
    th2, _ = derived(h, lz)
    t_up = (h[:, 1] - h[:, 0]) / (lz[1] - lz[0])
    nvox = int(2 * HALF / vox)
    ks = 8
    zs = np.linspace(lz[2] + 5, lz[1] - 5, ks)
    seg = (lz[1] - lz[2]) / ks
    idx_all = np.empty((len(h), ks), np.int64)
    for j, z in enumerate(zs):
        xy = h[:, 1] + t_up * (z - lz[1])
        idx_all[:, j], _ = grid_index(xy, vox)
    dth2 = th2 / 2.0
    lam = np.full(nvox * nvox, 1e-6)
    cov = np.bincount(idx_all.ravel(), minlength=nvox * nvox)
    for _ in range(iters):
        denom = lam[idx_all].sum(1) * seg + 1e-12
        w = dth2 / denom
        upd = np.bincount(idx_all.ravel(),
                          weights=np.repeat(w, ks), minlength=nvox * nvox)
        cnt = np.maximum(cov, 1)
        lam = lam * upd / cnt
    img = np.where(cov.reshape(nvox, nvox) >= 25 * ks // 2,
                   lam.reshape(nvox, nvox), np.nan)
    return img, nvox


class _Family(Method):
    tier = "A"
    fn = None

    def __init__(self, dev="cpu"):
        self.dev = dev

    def calibrate(self, blank_hits, layer_z, sigma_pos, x0_sample):
        return {}

    def reconstruct(self, hits, layer_z, sigma_pos, cal, vox, **kw):
        img, n = type(self).fn(hits, layer_z, vox)
        gx = grid_axes(HALF, vox)
        return {"img": img, "gx": gx, "gy": gx}


class PoCA(_Family):
    name = "poca"
    fn = staticmethod(poca_img)


class ASR(_Family):
    name = "asr"
    fn = staticmethod(asr_img)


class MLSEM(_Family):
    name = "mlsem"
    fn = staticmethod(mlsem_img)
