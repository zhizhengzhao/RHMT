"""MDMT-S: 4-plane scattering tomograph reconstructed as a penalised-MLE 3D voxel field."""
import numpy as np
import torch

from mdmt.contrast import ScatterT
from mdmt.field import HALF, VoxelField3D, column_eval, tv3d
from mdmt.inversion import coverage, grid_axes
from mdmt.methods.base import Method


class MDMTScatter(Method):
    name = "rht_s"
    tier = "A"

    def __init__(self, dev="cuda:0"):
        self.dev = dev

    def calibrate(self, blank_hits, layer_z, sigma_pos, x0_sample):
        m = ScatterT(layer_z, sigma_pos=sigma_pos, x0_sample=x0_sample, dev=self.dev)
        m.calibrate(blank_hits, steps=400, max_n=250000)
        return {"log_gref": m.log_gref.item(), "log_nu": m.log_nu.item()}

    def reconstruct(self, hits, layer_z, sigma_pos, cal, vox, x0_sample=122.9,
                    steps=1500, lr=3e-2, pri_tv=16.0, nz=6, wz=30.0,
                    min_cov=10, img_vox=6.0, **kw):
        m = ScatterT(layer_z, sigma_pos=sigma_pos, x0_sample=x0_sample, dev=self.dev)
        m.log_gref = torch.tensor(float(cal["log_gref"]), dtype=torch.float64,
                                  device=self.dev)
        m.log_nu = torch.tensor(float(cal["log_nu"]), dtype=torch.float64,
                                device=self.dev)
        prep = m.prepare(hits)
        zq = torch.as_tensor(m.zq, dtype=torch.float64, device=self.dev)
        xyz = torch.cat([prep["xy"],
                         zq[None, :, None].expand(prep["n"], -1, 1)], -1)  # (N,nq,3)
        field = VoxelField3D(nx=int(2 * HALF / vox), nz=nz, dev=self.dev)
        opt = torch.optim.Adam(field.parameters(), lr=lr)
        n = prep["n"]
        for _ in range(steps):
            opt.zero_grad()
            lam = field(xyz)                                   # (N,nq)
            loss = m.nll_t(prep, lam_line=lam) / n + pri_tv * tv3d(field, wz=wz)
            loss.backward()
            opt.step()
        img, _ = column_eval(field, vox=img_vox)
        cov = coverage(prep["xy"], prep["xy"].shape[1], HALF, img_vox)
        gx = grid_axes(HALF, img_vox)
        return {"img": np.where(cov >= min_cov, img, np.nan),
                "gx": gx, "gy": gx, "field": field}
