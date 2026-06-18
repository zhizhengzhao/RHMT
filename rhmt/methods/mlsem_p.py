"""MLS-EM with the spectrometer's measured momentum: the MLS-EM of baselines.py on
the inner four planes, with the per-muon scattering covariance scaled by the
measured 1/(p*beta)^2."""
import numpy as np

from rhmt.inversion import grid_axes
from rhmt.methods.base import Method
from rhmt.methods.baselines import HALF, mlsem_full_img
from rhmt.physics import MU
from rhmt.trackfit import fit_free


class MLSEMMomentum(Method):
    name = "mlsem_p"
    tier = "B"

    def __init__(self, dev="cuda:0"):
        self.dev = dev

    def calibrate(self, blank_hits, layer_z, sigma_pos, x0_sample):
        return {}

    def reconstruct(self, hits, layer_z, sigma_pos, cal, vox, **kw):
        fit = fit_free(hits, layer_z, sigma_pos=sigma_pos, dev=self.dev)
        ok = fit["ok"].cpu().numpy()
        p = fit["p_in"].cpu().numpy()
        good = ok & (p > 300) & (p < 60000)
        h = np.asarray(hits)[good][:, 1:5]
        lz = np.asarray(layer_z)[1:5]
        pg = p[good]
        pb = pg * (pg / np.sqrt(pg ** 2 + MU ** 2))
        pr2 = (np.median(pb) / pb) ** 2
        img, n = mlsem_full_img(h, lz, vox, sigma_pos=sigma_pos, pr2=pr2)
        gx = grid_axes(HALF, vox)
        return {"img": img, "gx": gx, "gy": gx}
