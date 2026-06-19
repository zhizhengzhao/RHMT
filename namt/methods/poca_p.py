"""PoCA with the spectrometer's measured momentum: closest-approach assignment of
the per-muon squared scattering angle weighted by the measured (p*beta)^2, on the
inner four planes."""
import numpy as np

from namt.inversion import grid_axes
from namt.methods.base import Method
from namt.methods.baselines import HALF, derived, grid_index, img_from_stat
from namt.physics import MU
from namt.trackfit import fit_free


class PoCAMomentum(Method):
    name = "poca_p"
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
        th2, xy = derived(h, lz)
        v = th2 * pb ** 2
        v = np.clip(v, None, np.quantile(v, 0.995))
        idx, n = grid_index(xy, vox)
        img = img_from_stat(idx, n, v, min_cov=25, stat="median")
        gx = grid_axes(HALF, vox)
        return {"img": img, "gx": gx, "gy": gx}
