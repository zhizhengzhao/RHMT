"""3-D voxel field for RHMT reconstruction."""
import numpy as np
import torch
import torch.nn as nn

HALF = 200.0      # [mm]
ZHALF = 234.5     # [mm]


class VoxelField3D(nn.Module):
    """Trilinear 3-D voxel field, vox[z, y, x]."""

    def __init__(self, nx=25, nz=6, half=HALF, zhalf=ZHALF, dev="cuda:0"):
        super().__init__()
        self.nx, self.nz = int(nx), int(nz)
        self.half, self.zhalf = float(half), float(zhalf)
        self.vox = nn.Parameter(torch.zeros(self.nz, self.nx, self.nx,
                                            dtype=torch.float64, device=dev))
        self.to(dev)

    def forward(self, xyz):
        """xyz (..., 3) [mm] -> field values (...,); zero outside the volume."""
        shp = xyz.shape[:-1]
        p = xyz.reshape(-1, 3)
        nx, nz = self.nx, self.nz
        gx = (p[:, 0] + self.half) / (2 * self.half) * nx - 0.5
        gy = (p[:, 1] + self.half) / (2 * self.half) * nx - 0.5
        gz = (p[:, 2] + self.zhalf) / (2 * self.zhalf) * nz - 0.5
        inside = ((gx > -0.5) & (gx < nx - 0.5) & (gy > -0.5) & (gy < nx - 0.5)
                  & (gz > -0.5) & (gz < nz - 0.5))
        ix = gx.floor().long().clamp(0, nx - 2)
        iy = gy.floor().long().clamp(0, nx - 2)
        iz = gz.floor().long().clamp(0, max(nz - 2, 0))
        fx = (gx - ix).clamp(0.0, 1.0)
        fy = (gy - iy).clamp(0.0, 1.0)
        fz = (gz - iz).clamp(0.0, 1.0) if nz > 1 else torch.zeros_like(gx)
        dz1 = 1 if nz > 1 else 0
        v = self.vox

        def g(dz, dy, dx):
            return v[iz + dz, iy + dy, ix + dx]

        val = (g(0, 0, 0) * (1 - fz) * (1 - fy) * (1 - fx)
               + g(0, 0, 1) * (1 - fz) * (1 - fy) * fx
               + g(0, 1, 0) * (1 - fz) * fy * (1 - fx)
               + g(0, 1, 1) * (1 - fz) * fy * fx
               + g(dz1, 0, 0) * fz * (1 - fy) * (1 - fx)
               + g(dz1, 0, 1) * fz * (1 - fy) * fx
               + g(dz1, 1, 0) * fz * fy * (1 - fx)
               + g(dz1, 1, 1) * fz * fy * fx)
        return (val * inside).reshape(shp)

    def tv(self, wz=30.0):
        """Anisotropic TV regularizer on voxel parameters."""
        v = self.vox
        tx = (v[:, :, 1:] - v[:, :, :-1]).abs().mean()
        ty = (v[:, 1:, :] - v[:, :-1, :]).abs().mean()
        tz = (v[1:, :, :] - v[:-1, :, :]).abs().mean() if self.nz > 1 else 0.0
        return tx + ty + wz * tz

    def column(self, img_vox=6.0):
        """Top-down column projection (integral over z) on a 2-D grid [mm]."""
        n = int(2 * self.half / img_vox)
        ax = (np.arange(n) + 0.5) * img_vox - self.half
        K = max(2 * self.nz, 12)
        dz = 2 * self.zhalf / K
        zs = -self.zhalf + (np.arange(K) + 0.5) * dz
        xx, yy = np.meshgrid(ax, ax)
        acc = np.zeros_like(xx)
        dev = self.vox.device
        with torch.no_grad():
            for zv in zs:
                pts = torch.tensor(np.stack([xx, yy, np.full_like(xx, zv)], -1),
                                   dtype=torch.float64, device=dev)
                acc += self(pts).cpu().numpy()
        return acc * dz, ax

    def slab(self, img_vox=6.0):
        """Per-depth-slab (x,y) images. Returns (slabs [nz, n, n], ax, z_centres)."""
        n = int(2 * self.half / img_vox)
        ax = (np.arange(n) + 0.5) * img_vox - self.half
        zc = (np.arange(self.nz) + 0.5) * (2 * self.zhalf / self.nz) - self.zhalf
        xx, yy = np.meshgrid(ax, ax)
        dev = self.vox.device
        out = []
        with torch.no_grad():
            for zv in zc:
                pts = torch.tensor(np.stack([xx, yy, np.full_like(xx, zv)], -1),
                                   dtype=torch.float64, device=dev)
                out.append(self(pts).cpu().numpy())
        return np.stack(out), ax, zc


def tv3d(field, wz=30.0):
    return field.tv(wz=wz)


def column_eval(field, vox=6.0):
    return field.column(img_vox=vox)
