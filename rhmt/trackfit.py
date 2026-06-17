"""Batched Levenberg-Marquardt track fits."""

import torch

from rhmt.physics import (arc_map, drift, fermi_eyges_cov,
                          instrument_from_layers, scatter_regions)

DEV = "cuda:1" if torch.cuda.is_available() else "cpu"


def init_theta(hits, lz):
    """Initialize theta from arm lines and chord bend."""
    x = hits[:, :, 0]
    y = hits[:, :, 1]
    tx = (x[:, 1] - x[:, 2]) / (lz[1] - lz[2])
    ty_up = (y[:, 1] - y[:, 2]) / (lz[1] - lz[2])
    ty_top = (y[:, 0] - y[:, 1]) / (lz[0] - lz[1])
    bend = 2.0 * (torch.atan(ty_top) - torch.atan(ty_up))
    kb = 0.3 * 3.0 * 400.0  # MeV·rad
    p0 = torch.clamp(kb / torch.clamp(bend.abs(), min=1e-4), 300.0, 60000.0)
    q0 = torch.sign(bend) * (1.0 / p0)
    x0 = x[:, 0]
    y0 = y[:, 0]
    return torch.stack([x0, tx, y0, ty_up, q0], dim=-1)


def forward_hits6_free(theta, ins, b_tesla, sign_b=1.0):
    """Forward model with independent arm curvatures. theta (N,6) = [x0,tx,y0,ty,Qin,Qout]."""
    lz = torch.as_tensor(ins["layer_z"], dtype=theta.dtype, device=theta.device)
    x, tx, y, ty, qin, qout = theta.unbind(-1)
    out = torch.empty(theta.shape[0], 6, 2, dtype=theta.dtype, device=theta.device)
    out[:, 0, 0], out[:, 0, 1] = x, y
    mag_bot, mag_top = ins["mag_up"]
    x, y, tx, ty = drift(x, y, tx, ty, lz[0] - mag_top)
    x, y, tx, ty = arc_map(x, y, tx, ty, sign_b * qin, b_tesla, mag_top - mag_bot)
    x, y, tx, ty = drift(x, y, tx, ty, mag_bot - lz[1])
    out[:, 1, 0], out[:, 1, 1] = x, y
    x, y, tx, ty = drift(x, y, tx, ty, lz[1] - lz[2])
    out[:, 2, 0], out[:, 2, 1] = x, y
    x, y, tx, ty = drift(x, y, tx, ty, lz[2] - lz[3])
    out[:, 3, 0], out[:, 3, 1] = x, y
    x, y, tx, ty = drift(x, y, tx, ty, lz[3] - lz[4])
    out[:, 4, 0], out[:, 4, 1] = x, y
    mag_bot2, mag_top2 = ins["mag_dn"]
    x, y, tx, ty = drift(x, y, tx, ty, lz[4] - mag_top2)
    x, y, tx, ty = arc_map(x, y, tx, ty, sign_b * qout, b_tesla, mag_top2 - mag_bot2)
    x, y, tx, ty = drift(x, y, tx, ty, mag_bot2 - lz[5])
    out[:, 5, 0], out[:, 5, 1] = x, y
    return out


def _jac6(theta, ins, b, sign_b):
    n = theta.shape[0]
    cols = []
    for k in range(6):
        v = torch.zeros_like(theta)
        v[:, k] = 1.0
        _, jvp = torch.autograd.functional.jvp(
            lambda t: forward_hits6_free(t, ins, b, sign_b).reshape(n, -1),
            (theta,), (v,), create_graph=False)
        cols.append(jvp)
    return torch.stack(cols, dim=-1)


def _lm6(theta, obs, wmat, ins, b_tesla, sign_b, iters):
    n = theta.shape[0]

    def cost(t):
        r = obs - forward_hits6_free(t, ins, b_tesla, sign_b).reshape(n, -1)
        wr = torch.einsum("nij,nj->ni", wmat, r)
        return (r * wr).sum(-1), r, wr

    c_old, r, wr = cost(theta)
    mu = torch.full((n,), 1e-3, dtype=theta.dtype, device=theta.device)
    eye = torch.eye(6, dtype=theta.dtype, device=theta.device).expand(n, 6, 6)
    for _ in range(iters):
        jac = _jac6(theta, ins, b_tesla, sign_b)
        wj = torch.einsum("nij,njk->nik", wmat, jac)
        jtj = torch.einsum("nij,nik->njk", jac, wj)
        jtr = torch.einsum("nij,ni->nj", jac, wr)
        step = torch.linalg.solve(jtj + mu[:, None, None] * eye, jtr[:, :, None])[:, :, 0]
        cand = theta + step
        c_new, r_new, wr_new = cost(cand)
        ok = c_new < c_old
        theta = torch.where(ok[:, None], cand, theta)
        r = torch.where(ok[:, None], r_new, r)
        wr = torch.where(ok[:, None], wr_new, wr)
        c_old = torch.where(ok, c_new, c_old)
        mu = torch.where(ok, mu / 3.0, mu * 5.0).clamp(1e-7, 1e3)
    return theta, c_old


def fit_free(hits, layer_z, b_tesla=3.0, sigma_pos=1.0, iters=12,
             sign_b=1.0, x0_rpc_eff=140.0, dev=DEV):
    """Free-curvature two-pass fit; returns per-muon state, Lambda, and chord point."""
    ins = instrument_from_layers(layer_z)
    h = torch.as_tensor(hits, dtype=torch.float64, device=dev)
    n = h.shape[0]
    obs = h.reshape(n, -1)
    lz = torch.as_tensor(layer_z, dtype=torch.float64, device=dev)
    t5 = init_theta(h, lz).to(torch.float64)
    theta = torch.cat([t5, t5[:, 4:5]], dim=1)
    sig2 = max(sigma_pos, 0.05) ** 2
    eye12 = torch.eye(12, dtype=torch.float64, device=dev)
    w1 = (eye12 / sig2).expand(n, -1, -1)
    theta, _ = _lm6(theta, obs, w1, ins, b_tesla, sign_b, iters)
    regs = scatter_regions(ins, x0_rpc_eff=x0_rpc_eff)
    t5b = torch.cat([theta[:, :4], theta[:, 4:5]], dim=1)
    c = fermi_eyges_cov(t5b.detach(), ins, regs)
    sigma = torch.zeros(n, 12, 12, dtype=torch.float64, device=dev)
    sigma[:, 0::2, 0::2] = c
    sigma[:, 1::2, 1::2] = c
    sigma += eye12 * sig2
    wmat = torch.linalg.inv(sigma)
    theta, chi2 = _lm6(theta, obs, wmat, ins, b_tesla, sign_b, iters)

    p_in = 1.0 / theta[:, 4].abs().clamp(min=1e-7)
    p_out = 1.0 / theta[:, 5].abs().clamp(min=1e-7)
    lam = p_in - p_out
    sign_ok = torch.sign(theta[:, 4]) == torch.sign(theta[:, 5])
    mag_bot, mag_top = ins["mag_up"]
    x, tx, y, ty = theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3]
    x, y, tx, ty = drift(x, y, tx, ty, lz[0] - mag_top)
    x, y, tx, ty = arc_map(x, y, tx, ty, sign_b * theta[:, 4], b_tesla,
                           mag_top - mag_bot)
    smp_bot, smp_top = ins["sample"]
    zmid = 0.5 * (smp_bot + smp_top)
    x_mid = x - tx * (mag_bot - zmid)
    y_mid = y - ty * (mag_bot - zmid)
    sec = torch.sqrt(1.0 + tx ** 2 + ty ** 2)
    return dict(theta=theta, chi2=chi2, p_in=p_in, p_out=p_out, lam=lam,
                ok=sign_ok, xy=torch.stack([x_mid, y_mid], 1), sec=sec,
                state_mb=torch.stack([x, y, tx, ty], 1), z_mb=float(mag_bot),
                ins=ins)
