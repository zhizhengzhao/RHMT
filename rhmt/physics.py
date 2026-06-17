"""Single-muon transport and scattering moments (momenta MeV, lengths mm, B Tesla)."""
import numpy as np
import torch

MU = 105.6583755  # MeV

RPC_FULL = 30.0  # mm


def instrument_from_layers(layer_z):
    """Derive magnet/sample slab boundaries from plane positions (top->bottom)."""
    lz = np.asarray(layer_z, float)
    h = RPC_FULL / 2.0
    ins = {"layer_z": lz, "n": len(lz)}
    if len(lz) == 6:
        ins["mag_up"] = (lz[1] + h, lz[0] - h)    # (bottom, top) of upper magnet
        ins["mag_dn"] = (lz[5] + h, lz[4] - h)
        ins["sample"] = (lz[3] + h, lz[2] - h)
    elif len(lz) == 4:
        ins["sample"] = (lz[2] + h, lz[1] - h)
    elif len(lz) == 3:
        ins["sample"] = (lz[2] + h, lz[1] - h)    # 2 planes above the sample, 1 below
    else:
        raise ValueError(f"unsupported plane count {len(lz)}")
    return ins


K_BB = 0.307075  # MeV cm^2/mol
ME = 0.51099895  # MeV


def bethe_dedx(p, s_zoa, i_ev=135.0):
    """Bethe-Bloch mean -dE/dx [MeV/mm]."""
    e = torch.sqrt(p * p + MU * MU)
    beta2 = (p / e) ** 2
    gamma = e / MU
    wmax = 2 * ME * beta2 * gamma**2 / (1 + 2 * gamma * ME / MU + (ME / MU) ** 2)
    i_mev = i_ev * 1e-6
    arg = 2 * ME * beta2 * gamma**2 * wmax / i_mev**2
    dedx_cm = K_BB * s_zoa / beta2 * (0.5 * torch.log(arg) - beta2)
    return dedx_cm / 10.0  # MeV/mm


def sample_ploss(p_in, path_mm, s_zoa, nsub=4):
    """Bethe-Bloch momentum loss via nsub midpoint sub-steps."""
    h = path_mm / nsub
    p = p_in
    for _ in range(nsub):
        e = torch.sqrt(p * p + MU * MU)
        dedx = bethe_dedx(p, s_zoa)
        p_mid = p - 0.5 * h * dedx * e / p
        e_mid = torch.sqrt(p_mid * p_mid + MU * MU)
        dpdz = bethe_dedx(p_mid, s_zoa) * e_mid / p_mid
        p = p - h * dpdz
    return torch.clamp(p, min=120.0)


def mpv_loss(p_in, path_mm, s_zoa, i_ev=135.0, nsub=2):
    """Most-probable momentum loss (Landau location) through a slab (PDG eq. 34.12)."""
    i_mev = i_ev * 1e-6
    h = path_mm / nsub
    p = p_in
    for _ in range(nsub):
        e = torch.sqrt(p * p + MU * MU)
        beta2 = (p / e) ** 2
        gam2 = (e / MU) ** 2
        xi = 0.5 * K_BB * s_zoa * (h / 10.0) / beta2   # MeV
        dmp = xi * (torch.log(2 * ME * beta2 * gam2 / i_mev)
                    + torch.log(xi / i_mev) + 0.2 - beta2)
        dmp = torch.clamp(dmp, min=0.0)
        de = dmp * e / p
        p = p - de
    return torch.clamp(p, min=120.0)


def arc_map(x, y, tx, ty, qsign_over_p, b_tesla, lz_mm):
    """Circular-arc propagation through a magnet slab (B ∥ +x)."""
    n2 = 1.0 + tx * tx + ty * ty
    sec_yz = torch.sqrt((1.0 + ty * ty) / n2)
    kappa = 0.3 * b_tesla * qsign_over_p / sec_yz
    u_in = -ty / torch.sqrt(1.0 + ty * ty)
    u_out = u_in + kappa * lz_mm
    u_out = torch.clamp(u_out, -0.985, 0.985)
    c_in = torch.sqrt(1.0 - u_in * u_in)
    c_out = torch.sqrt(1.0 - u_out * u_out)
    ty_out = -u_out / c_out
    dphi = torch.asin(u_out) - torch.asin(u_in)
    inv_k = 1.0 / kappa
    y_out = y + inv_k * (c_in - c_out)
    x_out = x - (tx / torch.sqrt(1.0 + ty * ty)) * inv_k * dphi
    tx_out = tx * torch.sqrt((1.0 + ty_out * ty_out) / (1.0 + ty * ty))
    return x_out, y_out, tx_out, ty_out


def drift(x, y, tx, ty, dz):
    """Field-free straight propagation downward by dz."""
    return x - tx * dz, y - ty * dz, tx, ty


ES = 13.6  # MeV
X0_AIR = 303900.0  # mm
X0_SIO2 = 123.2    # mm


def scatter_regions(ins, x0_rpc_eff=140.0, x0_sample=X0_SIO2):
    """Piecewise-constant scattering regions along z. Returns [(z_top, z_bot, inv_x0)]."""
    lz = ins["layer_z"]
    h = RPC_FULL / 2.0
    regs = []
    for k, zc in enumerate(lz):
        regs.append((zc + h, zc - h, 1.0 / x0_rpc_eff))
        if k + 1 < len(lz):
            top, bot = zc - h, lz[k + 1] + h
            smp_bot, smp_top = ins["sample"]
            if abs(top - smp_top) < 1e-6 and abs(bot - smp_bot) < 1e-6:
                regs.append((top, bot, 1.0 / x0_sample))
            else:
                regs.append((top, bot, 1.0 / X0_AIR))
    return regs


def fermi_eyges_cov(theta, ins, regs, dtype=None, dev=None):
    """Fermi-Eyges scattering covariance of per-plane crossings. Returns C (N, nlay, nlay)."""
    lz = torch.as_tensor(ins["layer_z"], dtype=theta.dtype, device=theta.device)
    n, nlay = theta.shape[0], len(ins["layer_z"])
    tx, ty, q = theta[:, 1], theta[:, 3], theta[:, 4]
    p = 1.0 / q.abs().clamp(min=1e-6)
    e = torch.sqrt(p * p + MU * MU)
    beta = p / e
    sec = torch.sqrt(1.0 + tx * tx + ty * ty)
    t2 = (ES / (p * beta)) ** 2 * sec ** 3
    c = torch.zeros(n, nlay, nlay, dtype=theta.dtype, device=theta.device)
    for (ztop, zbot, invx0) in regs:
        below = torch.nonzero(lz < zbot - 1e-9).flatten()
        if len(below) == 0:
            continue
        a, b = zbot, ztop
        for i in below:
            zi = lz[i]
            for j in below:
                zj = lz[j]
                def F(z, zi=zi, zj=zj):
                    return (zi * zj * z - (zi + zj) * z * z / 2.0 + z ** 3 / 3.0)
                c[:, i, j] += t2 * invx0 * (F(b) - F(a))
    return c


S_AIR = 0.001496   # g/cm^3
S_RPC = 1.25       # g/cm^3


def forward_hits6(theta, ins, b_tesla, s_zoa_bg, sign_b=1.0,
                  s_rpc=S_RPC, nsub_sample=4):
    """Forward hit prediction for the 6-plane spectrometer. Returns (N,6,2)."""
    lz = torch.as_tensor(ins["layer_z"], dtype=theta.dtype, device=theta.device)
    h = RPC_FULL / 2.0
    x, tx, y, ty, q = theta.unbind(-1)
    out = torch.empty(theta.shape[0], 6, 2, dtype=theta.dtype, device=theta.device)
    out[:, 0, 0], out[:, 0, 1] = x, y

    qsign = torch.sign(q)
    p = 1.0 / torch.abs(q)

    def slant():
        return torch.sqrt(1.0 + tx * tx + ty * ty)

    def decay(p_now, dz_mm, s_med, nsub=1):
        return sample_ploss(p_now, dz_mm * slant(), s_med, nsub=nsub)

    smp_bot, smp_top = ins["sample"]
    mag_bot, mag_top = ins["mag_up"]
    mag_bot2, mag_top2 = ins["mag_dn"]
    s_bg = s_zoa_bg

    p = decay(p, h, s_rpc)
    p = decay(p, (lz[0] - h) - mag_top, S_AIR)
    x, y, tx, ty = drift(x, y, tx, ty, lz[0] - mag_top)
    x, y, tx, ty = arc_map(x, y, tx, ty, sign_b * qsign / p, b_tesla, mag_top - mag_bot)
    p = decay(p, mag_top - mag_bot, S_AIR)
    x, y, tx, ty = drift(x, y, tx, ty, mag_bot - lz[1])
    out[:, 1, 0], out[:, 1, 1] = x, y
    p = decay(p, mag_bot - (lz[1] + h), S_AIR)
    p = decay(p, RPC_FULL, s_rpc)
    x, y, tx, ty = drift(x, y, tx, ty, lz[1] - lz[2])
    out[:, 2, 0], out[:, 2, 1] = x, y
    p = decay(p, (lz[1] - h) - (lz[2] + h), S_AIR)
    p = decay(p, RPC_FULL, s_rpc)
    p = decay(p, smp_top - smp_bot, s_bg, nsub=nsub_sample)
    x, y, tx, ty = drift(x, y, tx, ty, lz[2] - lz[3])
    out[:, 3, 0], out[:, 3, 1] = x, y
    p = decay(p, RPC_FULL, s_rpc)
    p = decay(p, (lz[3] - h) - (lz[4] + h), S_AIR)
    x, y, tx, ty = drift(x, y, tx, ty, lz[3] - lz[4])
    out[:, 4, 0], out[:, 4, 1] = x, y
    p = decay(p, RPC_FULL, s_rpc)
    p = decay(p, (lz[4] - h) - mag_top2, S_AIR)
    x, y, tx, ty = drift(x, y, tx, ty, lz[4] - mag_top2)
    x, y, tx, ty = arc_map(x, y, tx, ty, sign_b * qsign / p, b_tesla, mag_top2 - mag_bot2)
    x, y, tx, ty = drift(x, y, tx, ty, mag_bot2 - lz[5])
    out[:, 5, 0], out[:, 5, 1] = x, y
    return out
