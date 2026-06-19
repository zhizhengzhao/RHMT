"""Base class for NAMT reconstruction methods."""


class Method:
    name = "base"
    tier = "A"

    def calibrate(self, blank_hits, layer_z, sigma_pos, x0_sample):
        return {}

    def reconstruct(self, hits, layer_z, sigma_pos, cal, vox, **kw):
        raise NotImplementedError
