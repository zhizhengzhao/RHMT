"""ASR given the spectrometer's measured momentum: angle statistics on the inner
four planes, the per-muon scattering angle normalised by the measured p*beta."""
import numpy as np

from namt.inversion import grid_axes
from namt.methods.base import Method
from namt.methods.baselines import HALF, derived, grid_index, img_from_stat
from namt.physics import MU
from namt.trackfit import fit_free


class ASRMomentum(Method):
    name = "asr_p"
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
        h = hits[good][:, 1:5]                       # inner 4 planes straddle the sample
        lz = np.asarray(layer_z)[1:5]
        pg = p[good]
        beta = pg / np.sqrt(pg ** 2 + MU ** 2)
        th2, _ = derived(h, lz)
        v = np.sqrt(th2) * pg * beta                 # momentum-normalised scattering angle
        v = np.clip(v, None, np.quantile(v, 0.995))
        t_up = (h[:, 1] - h[:, 0]) / (lz[1] - lz[0])
        ks = 8
        zs = np.linspace(lz[2] + 10, lz[1] - 10, ks)
        idxs, vals = [], []
        for z in zs:
            xy = h[:, 1] + t_up * (z - lz[1])
            idx, n = grid_index(xy, vox)
            idxs.append(idx)
            vals.append(v)
        img = img_from_stat(np.concatenate(idxs), n, np.concatenate(vals),
                            min_cov=25 * ks // 2, stat="median")
        gx = grid_axes(HALF, vox)
        return {"img": img, "gx": gx, "gy": gx}
