"""MDMT-S on a 3-plane tracker: the scattering method with the bottom plane dropped
(2 planes above the sample, 1 below). Scattering is read from the displacement at the
single exit plane rather than the in/out kink, so one contrast dimension instead of two."""
from mdmt.methods.rht_s import MDMTScatter


class MDMTScatter3(MDMTScatter):
    name = "rht_3p"
    tier = "A"

    def calibrate(self, blank_hits, layer_z, sigma_pos, x0_sample):
        return super().calibrate(blank_hits[:, :3], layer_z[:3], sigma_pos, x0_sample)

    def reconstruct(self, hits, layer_z, sigma_pos, cal, vox, **kw):
        return super().reconstruct(hits[:, :3], layer_z[:3], sigma_pos, cal, vox, **kw)
