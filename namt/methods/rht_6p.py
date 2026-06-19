"""NAMT-E: stopping-power field s = rho Z/A from the momentum loss of muons
crossing the 6-plane magnetic spectrometer."""
import numpy as np
import torch

from namt.calib import flat_lookup, lam_table
from namt.field import HALF, VoxelField3D, column_eval, tv3d
from namt.inversion import coverage, grid_axes, huber
from namt.methods.base import Method
from namt.physics import MU, bethe_dedx
from namt.trackfit import fit_free


class NAMTSpectro(Method):
    name = "rht_6p"
    tier = "B"

    def __init__(self, dev="cuda:0"):
        self.dev = dev

    def calibrate(self, blank_hits, layer_z, sigma_pos, x0_sample):
        fb = fit_free(blank_hits, layer_z, sigma_pos=sigma_pos, dev=self.dev)
        t = lam_table(fb)
        return {"pq": t["pq"], "tabs": t["tabs"], "sigs": t["sigs"],
                "nb": np.array([t["nb"]])}

    def _zq(self, ins, nq=12):
        bot, top = ins["sample"]
        xg, _ = np.polynomial.legendre.leggauss(nq)
        return 0.5 * (top + bot) + 0.5 * (top - bot) * xg

    def reconstruct(self, hits, layer_z, sigma_pos, cal, vox, x0_sample=122.9,
                    s_steps=1500, lr_s=3e-2, pri_tv_s=0.3, nz=6, wz=30.0,
                    min_cov=10, img_vox=6.0, **kw):
        dev = self.dev
        table = dict(pq=cal["pq"], tabs=cal["tabs"], sigs=cal["sigs"],
                     nb=int(np.asarray(cal["nb"]).ravel()[0]))
        fit = fit_free(hits, layer_z, sigma_pos=sigma_pos, dev=dev)
        ok = fit["ok"]
        pn = fit["p_in"][ok].cpu().numpy()
        lam_np = fit["lam"][ok].cpu().numpy()
        xy_mid = fit["xy"][ok].cpu().numpy()
        good = (pn > 300) & (pn < 60000) & np.isfinite(lam_np) & (lam_np < pn - 50)
        gsel = torch.as_tensor(good, device=dev)

        ell = np.log(pn[good] / (pn[good] - lam_np[good]))
        flat, sig = flat_lookup(table, pn[good], xy_mid[good])
        r_E = torch.as_tensor(ell - flat, dtype=torch.float64, device=dev)
        wE = 1.0 / torch.as_tensor(sig, dtype=torch.float64, device=dev)
        sub = {k: (v[ok][gsel] if torch.is_tensor(v) and v.dim() and
                   v.shape[0] == fit["ok"].shape[0] else v)
               for k, v in fit.items()}
        sub["ins"] = fit["ins"]
        sub["z_mb"] = fit["z_mb"]
        zqE = torch.as_tensor(self._zq(fit["ins"]), dtype=torch.float64, device=dev)
        st = sub["state_mb"]
        xE = st[:, 0:1] - st[:, 2:3] * (sub["z_mb"] - zqE)[None, :]
        yE = st[:, 1:2] - st[:, 3:4] * (sub["z_mb"] - zqE)[None, :]
        xyzE = torch.stack([xE, yE, zqE[None, :].expand(xE.shape[0], -1)], -1)
        wqE = torch.as_tensor(0.5 * (fit["ins"]["sample"][1] - fit["ins"]["sample"][0])
                              * np.polynomial.legendre.leggauss(12)[1],
                              dtype=torch.float64, device=dev)
        kap = bethe_dedx(sub["p_in"], torch.tensor(1.0, dtype=torch.float64, device=dev))
        beta_in = sub["p_in"] / torch.sqrt(sub["p_in"] ** 2 + MU ** 2)
        coefE = kap / (sub["p_in"] * beta_in)
        secE = sub["sec"]
        field_s = VoxelField3D(nx=int(2 * HALF / vox), nz=nz, dev=dev)
        opt = torch.optim.Adam(field_s.parameters(), lr=lr_s)
        for _ in range(s_steps):
            opt.zero_grad()
            sline = field_s(xyzE)
            pred = coefE * secE * torch.einsum("nq,q->n", sline, wqE)
            (huber((pred - r_E) * wE).mean() + pri_tv_s * tv3d(field_s, wz=wz)).backward()
            opt.step()

        img_s, _ = column_eval(field_s, vox=img_vox)
        cov = coverage(xyzE[:, :, :2], xyzE.shape[1], HALF, img_vox)
        gx = grid_axes(HALF, img_vox)
        img_s = np.where(cov >= min_cov, img_s, np.nan)
        return {"img_s": img_s, "gx": gx, "gy": gx, "field_s": field_s}
