"""Energy-loss statistic: a derived-observable baseline for the energy channel.

Per muon, the momentum-corrected log loss l * p*beta (proportional to the path
integral of s = rho Z/A) is binned to the voxel column at the track's sample
crossing and reduced by the per-column median. It uses the spectrometer's
measured momentum but no blank calibration.
"""
import numpy as np

from namt.field import HALF
from namt.inversion import grid_axes
from namt.methods.base import Method
from namt.physics import MU
from namt.trackfit import fit_free


def _grid_index(xy, vox):
    n = int(2 * HALF / vox)
    ix = np.clip(((xy[:, 0] + HALF) / vox).astype(int), 0, n - 1)
    iy = np.clip(((xy[:, 1] + HALF) / vox).astype(int), 0, n - 1)
    return iy * n + ix, n


def _median_image(idx, n, vals, min_cov):
    order = np.argsort(idx)
    idx_s, val_s = idx[order], vals[order]
    bounds = np.searchsorted(idx_s, np.arange(n * n + 1))
    img = np.full(n * n, np.nan)
    for k in range(n * n):
        a, b = bounds[k], bounds[k + 1]
        if b - a >= min_cov:
            img[k] = np.median(val_s[a:b])
    return img.reshape(n, n)


class EnergyLossStat(Method):
    name = "eloss"
    tier = "B"

    def __init__(self, dev="cuda:0"):
        self.dev = dev

    def reconstruct(self, hits, layer_z, sigma_pos, cal, vox, x0_sample=122.9,
                    min_cov=25, img_vox=6.0, **kw):
        fit = fit_free(hits, layer_z, sigma_pos=sigma_pos, dev=self.dev)
        ok = fit["ok"].cpu().numpy()
        pin = fit["p_in"].cpu().numpy()
        pout = fit["p_out"].cpu().numpy()
        xy = fit["xy"].cpu().numpy()
        good = (ok & (pin > 300) & (pin < 60000) & (pout > 0)
                & ((pin - pout) < pin - 50))
        ell = np.log(pin[good] / pout[good])
        pm = 0.5 * (pin[good] + pout[good])
        pbeta = pm * pm / np.sqrt(pm * pm + MU ** 2)
        idx, n = _grid_index(xy[good], img_vox)
        img = _median_image(idx, n, ell * pbeta, min_cov)
        gx = grid_axes(HALF, img_vox)
        return {"img": img, "gx": gx, "gy": gx}
