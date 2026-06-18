"""4-plane REML scattering contrast and its Student-t likelihood (MDMT-S)."""
import numpy as np
import torch

from mdmt.physics import ES, instrument_from_layers
from mdmt.contrast_moments import region_moment_matrix

DEV = "cuda:2" if torch.cuda.is_available() else "cpu"


def contrast_basis(layer_z, dev):
    z = torch.as_tensor(layer_z, dtype=torch.float64, device=dev)
    X = torch.stack([torch.ones_like(z), z - z.mean()], 1)
    q, _ = torch.linalg.qr(torch.cat([X, torch.eye(len(z), dtype=torch.float64,
                                                    device=dev)], 1))
    return q[:, 2:]


class ScatterT:
    def __init__(self, layer_z, sigma_pos, x0_sample=122.9, nq=12, dev=DEV):
        self.dev = dev
        self.ins = instrument_from_layers(layer_z)
        self.c = contrast_basis(layer_z, dev)                  # (n, n-2)
        self.cTc = (self.c.T @ self.c)                         # (n-2, n-2)
        Kbg = region_moment_matrix(self.ins, x0_sample=None)
        self.cKbg = torch.as_tensor(self.c.cpu().numpy().T @ Kbg @ self.c.cpu().numpy(),
                                    dtype=torch.float64, device=dev)
        bot, top = self.ins["sample"]
        xg, wg = np.polynomial.legendre.leggauss(nq)
        zq = 0.5 * (top + bot) + 0.5 * (top - bot) * xg
        self.wq = torch.as_tensor(0.5 * (top - bot) * wg, dtype=torch.float64, device=dev)
        lz = self.ins["layer_z"]
        lever = np.zeros((len(lz), nq))
        for k, zk in enumerate(lz):
            if zk < bot:
                lever[k] = zk - zq
        lever = torch.as_tensor(lever, dtype=torch.float64, device=dev)
        self.clever = torch.einsum("kc,kq->cq", self.c, lever)
        self.zq = zq
        self.sig2 = float(max(sigma_pos, 0.05)) ** 2
        self.lam_bg = 1.0 / float(x0_sample)
        self.log_gref = torch.tensor(np.log((ES / 2000.) ** 2), dtype=torch.float64,
                                     device=dev, requires_grad=True)
        self.log_nu = torch.tensor(np.log(3.0), dtype=torch.float64, device=dev,
                                   requires_grad=True)

    def prepare(self, hits, max_n=None):
        h = torch.as_tensor(hits, dtype=torch.float64, device=self.dev)
        if max_n:
            h = h[:max_n]
        lz = torch.as_tensor(self.ins["layer_z"], dtype=torch.float64, device=self.dev)
        zc = lz - lz.mean()
        slope = torch.einsum("nkc,k->nc", h, zc) / (zc * zc).sum()
        inter = h.mean(1)
        w = torch.einsum("nkc,kj->ncj", h, self.c)             # (N,2,n-2)
        zq_t = torch.as_tensor(self.zq, dtype=torch.float64, device=self.dev)
        xy = inter[:, None, :] + slope[:, None, :] * (zq_t - lz.mean())[None, :, None]
        sec = torch.sqrt(1 + slope[:, 0] ** 2 + slope[:, 1] ** 2)
        return dict(w=w, xy=xy, sec=sec, n=h.shape[0])

    def _scale(self, prep, lam_line=None):
        sec = prep["sec"]
        if lam_line is None:
            total = self.lam_bg * torch.ones(prep["n"], self.wq.shape[0],
                                             dtype=torch.float64, device=self.dev)
        else:
            total = self.lam_bg + lam_line
        total = torch.clamp(total, min=1e-7)
        powq = total * self.wq[None, :] * (sec ** 3)[:, None]
        cKfield = torch.einsum("cq,nq,dq->ncd", self.clever, powq, self.clever)
        return self.cKbg[None] * (sec ** 3)[:, None, None] + cKfield

    def nll_t(self, prep, lam_line=None, g_per=None, nu_override=None):
        if g_per is not None:
            g = g_per[:, None, None]
            nu = nu_override if nu_override is not None else 8.0
        else:
            g = torch.exp(self.log_gref)
            nu = torch.exp(self.log_nu) + 1.0                    # nu > 1
        cK = self._scale(prep, lam_line)
        S = self.sig2 * self.cTc[None] + g * cK
        chol = torch.linalg.cholesky(S)
        logdet = 2 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        w = prep["w"].permute(0, 2, 1)
        sol = torch.cholesky_solve(w, chol)
        delta = (w * sol).sum(-2)
        p = w.shape[-2]
        nll = ((nu + p) * torch.log1p(delta / nu)).sum(-1) + 2 * logdet
        return nll.sum()

    def calibrate_nu_measured(self, blank_hits, g_per, steps=300, lr=0.05,
                              max_n=200000, lo=2.0, hi=200.0):
        """Calibrate tail nu on blank with per-muon momentum fixed."""
        prep = self.prepare(blank_hits, max_n=max_n)
        g = torch.as_tensor(g_per, dtype=torch.float64,
                            device=self.dev)[:prep["n"]]
        raw = torch.zeros((), dtype=torch.float64, device=self.dev,
                          requires_grad=True)
        opt = torch.optim.Adam([raw], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            nu = lo + (hi - lo) * torch.sigmoid(raw)
            (self.nll_t(prep, lam_line=None, g_per=g, nu_override=nu)
             / prep["n"]).backward()
            opt.step()
        return float(lo + (hi - lo) * torch.sigmoid(raw).item())

    def calibrate(self, blank_hits, steps=400, lr=0.05, max_n=200000, verbose=False):
        prep = self.prepare(blank_hits, max_n=max_n)
        opt = torch.optim.Adam([self.log_gref, self.log_nu], lr=lr)
        for it in range(steps):
            opt.zero_grad()
            nll = self.nll_t(prep) / prep["n"]
            nll.backward()
            opt.step()
            if verbose and (it % 100 == 0 or it == steps - 1):
                g = torch.exp(self.log_gref).item()
                nu = torch.exp(self.log_nu).item() + 1
                print(f"  calib {it}: nll={nll.item():.4f} p_eff={ES/np.sqrt(g)/1000:.2f}GeV nu={nu:.2f}")
        self.log_gref.requires_grad_(False)
        self.log_nu.requires_grad_(False)
        return float(nll.item())
