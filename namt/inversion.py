"""Shared MAP-inversion utilities: bilinear sampling, Huber loss, TV."""
import torch

HALF = 200.0  # [mm]


def bilinear(grid, xy, half=HALF, vox=16.0):
    """Bilinear sample of grid (n,n) at xy (...,2) [mm]; zero outside."""
    gx = (xy[..., 0] + half) / vox - 0.5
    gy = (xy[..., 1] + half) / vox - 0.5
    nx = grid.shape[0]
    ix = torch.clamp(gx.floor().long(), 0, nx - 2)
    iy = torch.clamp(gy.floor().long(), 0, nx - 2)
    fx = torch.clamp(gx - ix, 0.0, 1.0)
    fy = torch.clamp(gy - iy, 0.0, 1.0)
    inside = (gx > -0.5) & (gx < nx - 0.5) & (gy > -0.5) & (gy < nx - 0.5)
    v = (grid[iy, ix] * (1 - fx) * (1 - fy) + grid[iy, ix + 1] * fx * (1 - fy)
         + grid[iy + 1, ix] * (1 - fx) * fy + grid[iy + 1, ix + 1] * fx * fy)
    return v * inside


def huber(x, delta=1.345):
    ax = x.abs()
    return torch.where(ax <= delta, 0.5 * x * x, delta * (ax - 0.5 * delta))


def tv(g):
    return (g[1:] - g[:-1]).abs().mean() + (g[:, 1:] - g[:, :-1]).abs().mean()


def coverage(xy, nq, half=HALF, vox=16.0, device=None):
    """Chords-per-voxel map."""
    nx = int(2 * half / vox)
    fxy = xy.reshape(-1, 2)
    ix = torch.clamp(((fxy[:, 0] + half) / vox).long(), 0, nx - 1)
    iy = torch.clamp(((fxy[:, 1] + half) / vox).long(), 0, nx - 1)
    cov = torch.zeros(nx, nx, dtype=torch.float64, device=fxy.device)
    cov.index_put_((iy, ix), torch.ones_like(fxy[:, 0]), accumulate=True)
    return (cov / nq).cpu().numpy()


def grid_axes(half=HALF, vox=16.0):
    import numpy as np
    nx = int(2 * half / vox)
    return -half + vox * (np.arange(nx) + 0.5)
