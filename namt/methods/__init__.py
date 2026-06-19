"""NAMT method registry."""
from namt.methods.baselines import ASR, MLSEM, PoCA
from namt.methods.rht_s import NAMTScatter
from namt.methods.rht_3p import NAMTScatter3
from namt.methods.rht_6p import NAMTSpectro
from namt.methods.asr_p import ASRMomentum
from namt.methods.poca_p import PoCAMomentum
from namt.methods.mlsem_p import MLSEMMomentum
from namt.methods.eloss import EnergyLossStat

REGISTRY = {m.name: m for m in (PoCA, ASR, MLSEM, ASRMomentum, PoCAMomentum,
                                MLSEMMomentum, NAMTScatter3, NAMTScatter, NAMTSpectro,
                                EnergyLossStat)}


def get(name, dev="cuda:0"):
    return REGISTRY[name](dev=dev)
