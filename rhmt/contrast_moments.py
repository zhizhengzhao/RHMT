"""Closed-form Fermi-Eyges lever-moment integrals of the background scattering budget."""
import numpy as np

from rhmt.physics import scatter_regions


def region_moment_matrix(ins, x0_rpc_eff=140.0, x0_sample=None):
    """Background moment matrix K_bg (n×n) for per-plane scattering covariance."""
    lz = ins["layer_z"]
    n = len(lz)
    regs = scatter_regions(ins, x0_rpc_eff=x0_rpc_eff,
                           x0_sample=(x0_sample if x0_sample else 1e30))
    k = np.zeros((n, n))
    for (ztop, zbot, invx0) in regs:
        below = np.where(lz < zbot - 1e-9)[0]
        for i in below:
            for j in below:
                zi, zj = lz[i], lz[j]

                def f(z):
                    return zi * zj * z - (zi + zj) * z * z / 2.0 + z ** 3 / 3.0

                k[i, j] += invx0 * (f(ztop) - f(zbot))
    return k
