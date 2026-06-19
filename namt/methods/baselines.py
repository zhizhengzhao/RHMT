"""Scattering-tomography baselines: PoCA, ASR, MLS-EM."""
import numpy as np

from namt.inversion import grid_axes
from namt.methods.base import Method

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


def poca_img(h, lz, vox, min_cov=25):
    th2, xy = derived(h, lz)
    idx, n = grid_index(xy, vox)
    return img_from_stat(idx, n, th2, min_cov=min_cov, stat="median"), n


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


def mlsem_full_img(h, lz, vox, sigma_pos=0.05, iters=60, ks=8, pr2=None):
    """MLS-EM (Schultz et al.): per-projection (dtheta, displacement) datum,
    PoCA-kinked path, 3D path length, weight matrix W, and a variance-components EM
    update with a median correction. pr2 (N,) scales each muon's scattering
    covariance by the measured 1/(p*beta)^2 when momentum is available; None uses a
    nominal momentum."""
    z0, z1, z2, z3 = (float(lz[0]), float(lz[1]), float(lz[2]), float(lz[3]))
    t_up = (h[:, 1] - h[:, 0]) / (z1 - z0)
    t_dn = (h[:, 3] - h[:, 2]) / (z3 - z2)
    dth = t_dn - t_up
    dx = h[:, 2] - (h[:, 1] + t_up * (z2 - z1))
    num = (h[:, 2] - h[:, 1]) + t_dn * (z1 - z2)
    dt = t_up - t_dn
    with np.errstate(divide="ignore", invalid="ignore"):
        zpc = num / np.where(np.abs(dt) < 1e-12, np.nan, dt)
    zp = np.clip(z1 + np.nan_to_num(np.nanmean(zpc, 1), nan=0.5 * (z1 + z2)), z2, z1)
    H = z1 - z2; seg = H / ks
    sec = np.sqrt(1.0 + t_up[:, 0] ** 2 + t_up[:, 1] ** 2)
    L = seg * sec
    zc = z1 - (np.arange(ks) + 0.5) * seg
    nvox = int(2 * HALF / vox); nv2 = nvox * nvox
    idx = np.empty((len(h), ks), np.int64)
    for k in range(ks):
        xin = h[:, 1] + t_up * (zc[k] - z1)
        xout = h[:, 2] + t_dn * (zc[k] - z2)
        xy = np.where((zc[k] >= zp)[:, None], xin, xout)
        idx[:, k], _ = grid_index(xy, vox)
    Tk = (ks - 1 - np.arange(ks))[None, :] * L[:, None]
    Lf = L[:, None] * np.ones(ks)[None, :]
    W = np.zeros((len(h), ks, 2, 2))
    W[:, :, 0, 0] = Lf
    W[:, :, 0, 1] = W[:, :, 1, 0] = Lf ** 2 / 2 + Lf * Tk
    W[:, :, 1, 1] = Lf ** 3 / 3 + Lf ** 2 * Tk + Lf * Tk ** 2
    if pr2 is not None:
        W = W * np.asarray(pr2, dtype=float).reshape(-1, 1, 1, 1)
    duz, ddz = z1 - z0, z3 - z2; rr = (z2 - z1) / duz
    Cm = np.array([[1 / duz, -1 / duz, -1 / ddz, 1 / ddz], [rr, -(1 + rr), 1.0, 0.0]])
    Nmat = sigma_pos ** 2 * (Cm @ Cm.T)
    lam = np.full(nv2, max(np.median((dth ** 2).sum(1)) / (2 * H), 1e-9))
    cov = np.bincount(idx.ravel(), minlength=nv2)
    Dx = np.stack([dth[:, 0], dx[:, 0]], 1); Dy = np.stack([dth[:, 1], dx[:, 1]], 1)
    flat = idx.ravel()
    for _ in range(iters):
        Sig = np.einsum('nk,nkij->nij', lam[idx], W) + Nmat
        Si = np.linalg.inv(Sig)
        trSW = np.einsum('nab,nkba->nk', Si, W)
        Cs = []
        for Dp in (Dx, Dy):
            v = np.einsum('nij,nj->ni', Si, Dp)
            Cs.append((np.einsum('ni,nkij,nj->nk', v, W, v) - trSW).ravel())
        vox_all = np.concatenate([flat, flat]); C_all = np.concatenate(Cs)
        o = np.argsort(vox_all, kind="stable")
        vs, cs = vox_all[o], C_all[o]
        b = np.searchsorted(vs, np.arange(nv2 + 1))
        corr = np.zeros(nv2)
        for j in np.nonzero(cov > 0)[0]:
            corr[j] = np.median(cs[b[j]:b[j + 1]])
        lam = np.maximum(lam + lam * lam * corr, 1e-12)
    return np.where(cov.reshape(nvox, nvox) >= 25 * ks // 2,
                    lam.reshape(nvox, nvox), np.nan), nvox


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

    def reconstruct(self, hits, layer_z, sigma_pos, cal, vox, min_cov=25, **kw):
        img, n = poca_img(hits, layer_z, vox, min_cov=min_cov)
        gx = grid_axes(HALF, vox)
        return {"img": img, "gx": gx, "gy": gx}


class ASR(_Family):
    name = "asr"
    fn = staticmethod(asr_img)


class MLSEM(_Family):
    name = "mlsem"

    def reconstruct(self, hits, layer_z, sigma_pos, cal, vox, **kw):
        img, n = mlsem_full_img(hits, layer_z, vox, sigma_pos=sigma_pos)
        gx = grid_axes(HALF, vox)
        return {"img": img, "gx": gx, "gy": gx}
